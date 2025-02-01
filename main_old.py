import streamlit as st
from toolkit import embed_pdf, retrieve_documents, generate_response
from langchain.memory import ConversationBufferMemory


# Set up the Streamlit app
st.set_page_config(
    page_title="RAG-based Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Chat with the RAG-based Assistant")


# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for memory
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# Display the chat messages
for message in st.session_state.messages:
    if message["is_user"]:
        st.markdown(f"**You:** {message['text']}")
    else:
        st.markdown(f"**Assistant:** {message['text']}")

# Create a form for user input and send button
with st.form(key='message_form', clear_on_submit=True):
    user_input = st.text_input("Type your message here...", key='input')
    submit_button = st.form_submit_button(label='Send')

# When the user submits a message
if submit_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"is_user": True, "text": user_input})

    assistant_response = generate_response(
        query=user_input
    )

    # Add assistant's response to chat history
    st.session_state.messages.append({"is_user": False, "text": assistant_response})

    # The input field is automatically cleared due to clear_on_submit=True
    # No need to manually reset st.session_state or user_input

    # Rerun the app to update the displayed messages
    st.rerun()
