# main.py

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.graph import app
from graph.chains.generation import generation_chain

def main():
    st.title("Medical Assistant")
    st.write(
        "This assistant will first ask follow-up questions like a human clinician "
        "to understand your situation before offering personalized guidance."
    )

    # Initialize chat history in the Streamlit session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat-style user input
    user_input = st.chat_input("Share how you're feeling or describe your symptoms:")

    if user_input:
        # Add the new user message to the history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Build a conversation transcript to send into the graph
        conversation_text_lines = []
        for msg in st.session_state.messages:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            conversation_text_lines.append(f"{speaker}: {msg['content']}")
        conversation_text = "\n".join(conversation_text_lines)

        inputs = {"question": conversation_text}

        with st.chat_message("assistant"):
            with st.spinner("Thinking about your situation..."):
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

                    # Show assistant reply in the chat bubble
                    st.markdown(generation)

                    # Append assistant message to session history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": generation}
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

                    if context:
                        with st.expander(
                            f"Show medical information used (Source: {source})"
                        ):
                            st.write(context)

                except Exception as e:
                    # If the graph fails (e.g., recursion limit), fall back to a
                    # single LLM call so the user still gets a response.
                    print(f"Error while running graph workflow: {e}")
                    try:
                        fallback_answer = generation_chain.invoke(
                            {"context": "", "question": conversation_text}
                        )
                        st.markdown(fallback_answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": fallback_answer}
                        )
                    except Exception as inner_exc:
                        print(f"Fallback generation also failed: {inner_exc}")
                        st.error(
                            "An error occurred while processing your question. "
                            "Please try again in a moment."
                        )

    if not st.session_state.messages:
        st.info(
            "Start by briefly sharing what you're experiencing. "
            "The assistant will respond with clarifying questions first, "
            "then offer personalized guidance once it has enough information."
        )

if __name__ == "__main__":
    main()
