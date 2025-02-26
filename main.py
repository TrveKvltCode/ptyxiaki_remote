import streamlit as st
from toolkit import embed_pdf, retrieve_documents, generate_response
from langchain.memory import ConversationBufferWindowMemory
#import pysqlite3
import sys
import os
from dotenv import load_dotenv


#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

APIKEY = os.getenv("OPENAI_API_KEY")


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
    st.session_state.memory = ""

# Display the chat messages
for message in st.session_state.messages:
    if message["is_user"]:
        st.markdown(f"**You:** {message['text']}")
    else:
        st.markdown(f"**Assistant:** {message['text']}")

# Temporary variable to handle user input
user_input = st.text_input("Type your message here...", key="input")

# When the user submits a message
if st.button("Send"):
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"is_user": True, "text": user_input})

        # Generate the assistant's response
        assistant_response = generate_response(
            query=user_input,
            memoryarg= st.session_state.memory
        )

        # Add assistant's response to chat history
        st.session_state.messages.append({"is_user": False, "text": assistant_response})

        # add chat history for further uses: TO REVIEW AND SAWP WITH MEMORYOBJ

        chat_history = f"Human Message: {user_input} \n\n AI message:{assistant_response}\n\n"
        st.session_state.memory = chat_history
        # Clear the input field by updating its key
        user_input = None

        # Rerun the app to update the displayed messages
        st.rerun()

if st.button("Clear Memory"):
    st.session_state.memory = ""