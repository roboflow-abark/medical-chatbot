import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.graph import app
from graph.chains.generation import generation_chain


def _build_conversation_text() -> str:
    """Build textual transcript of the conversation for the LLM."""
    lines = []
    for msg in st.session_state.messages:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


def main():
    st.title("Medical Assistant")
    st.write(
        ""
    )

    # Initialize chat history in the Streamlit session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat-style user input
    user_input = st.chat_input("Share how you're feeling or describe your symptoms:")

    if user_input:
        # Add the new user message to the history immediately
        st.session_state.messages.append({"role": "user", "content": user_input})

    # Render the conversation so far
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                source = msg.get("context_source")
                context = msg.get("context")
                if source:
                    st.caption(f"Source: {source}")
                if context:
                    with st.expander(
                        f"Show medical information used (Source: {source or 'Unknown Source'})"
                    ):
                        st.write(context)

    # Determine if the last message still needs an assistant response
    needs_answer = (
        len(st.session_state.messages) > 0
        and st.session_state.messages[-1]["role"] == "user"
    )

    # If we have a new user message without an answer yet, generate one
    if needs_answer:
        conversation_text = _build_conversation_text()
        inputs = {"question": conversation_text}

        with st.spinner("Thinking..."):
            try:
                state = {}
                # Run the workflow using app.stream and collect outputs
                for output in app.stream(
                    inputs, config={"configurable": {"thread_id": "therapy-thread"}}
                ):
                    state.update(output)
                    print(f"Output from node: {output}")
                    print(f"Updated state: {state}")
                print(f"Final state: {state}")

                # Access the 'generate' node's outputs
                generate_output = state.get("generate", {})
                retrieve_output = state.get("retrieve", {})

                generation = generate_output.get("generation")
                if not generation:
                    generation = (
                        "I'm sorry, I couldn't generate a helpful response right now. "
                        "Please try adding a bit more detail or rephrasing your message."
                    )

                # Optionally show the underlying context used for the answer
                context = (
                    generate_output.get("context")
                    or retrieve_output.get("context")
                    or state.get("context")
                )
                source = (
                    generate_output.get("context_source")
                    or retrieve_output.get("context_source")
                    or state.get("context_source", "Unknown Source")
                )

                # Append assistant message (including context + source) to session history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": generation,
                        "context": context,
                        "context_source": source,
                    }
                )

            except Exception as e:
                # If the graph fails (e.g., recursion limit), fall back to a
                # single LLM call so the user still gets a response.
                print(f"Error while running graph workflow: {e}")
                try:
                    fallback_answer = generation_chain.invoke(
                        {"context": "", "question": conversation_text}
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": fallback_answer,
                            "context": "",
                            "context_source": "Fallback LLM (no retrieved context)",
                        }
                    )
                except Exception as inner_exc:
                    print(f"Fallback generation also failed: {inner_exc}")
                    st.error(
                        "An error occurred while processing your question. "
                        "Please try again in a moment."
                    )

        # After appending the assistant answer, rerun so the latest chat messages
        # are rendered cleanly without duplicate content.
        st.rerun()

    if not st.session_state.messages:
        st.info(
            ""
        )


if __name__ == "__main__":
    main()
