# ptyxiaki_toolkit.py

import os
from typing import List

from langchain.chains.llm import LLMChain
# Import necessary classes from LangChain and OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
#import pysqlite3
import sys


# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

APIKEY = os.getenv("OPENAI_API_KEY")
CHROMADB_DIR = r"/app/testembeddingsdir"

def embed_pdf(pdf_path: str, vectorstore_dir: str) -> bool:
    """
    Parses a PDF file, embeds its content into a vectorstore using ChromaDB, and saves it.

    Args:
        pdf_path (str): Full path to the PDF file.
        vectorstore_dir (str): Target directory to save the vectorstore.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Load the PDF and split it into documents
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=APIKEY)

        # Create a Chroma vectorstore from the documents and embeddings
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=vectorstore_dir
        )

        # Persist the vectorstore to disk
        vectorstore.persist()

        return True
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        return False


def retrieve_documents(query: str, vectorstore_dir: str, top_k: int = 5) -> List:
    """
    Retrieves documents from the vectorstore based on vector similarity to the query.

    Args:
        query (str): User's search query.
        vectorstore_dir (str): Directory where the vectorstore is stored.
        top_k (int): Number of top documents to retrieve.

    Returns:
        List[str]: A list of retrieved document contents.
    """
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=APIKEY)

        # Load the existing Chroma vectorstore
        vectorstore = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=embeddings
        )

        # Create a retriever object
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Retrieve relevant documents
        docs = retriever.invoke(query)
        with open("testdocumentsexported.txt", "w") as file:
            content = f"LENGTH OF DOCS {len(docs)} CONTENT: {[doc.page_content for doc in docs]}"
            file.write(content)
        # Extract and return the content from the documents
        return [doc.page_content for doc in docs]

    except Exception as e:
        with open("testdocumentsexported.txt", "w") as file:
            file.write(f"An error occurred during retrieval: {e}")
        return []


def generate_response(query: str, memoryarg ,vectorstore_dir=CHROMADB_DIR) -> str:
    """
    Generates a response to the user's query using retrieved documents and an LLM.

    Args:
        query (str): User's query.
        vectorstore_dir (str): Directory where the vectorstore is stored.

    Returns:
        str: Generated response from the assistant.
    """
    print("GENERATE RTESPONSE BEING CALLED")
    try:
        # Check for the API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")

        # Retrieve relevant documents using the existing function
        retrieved_docs = retrieve_documents(query, vectorstore_dir, top_k=5)

        # Combine the documents into a single context string
        context = "\n\n".join(retrieved_docs)

        # Define the prompt template
        # prompt_template = PromptTemplate(
        #     input_variables=["context", "question"],
        #     template=(
        #         "You are a knowledgeable assistant. Use the following context to answer the question. MAKE SURE TO MENTION THE DOCUMENTS PROVIDED\n\n"
        #         "Context:\n{context}\n\n"
        #         "Question: {question}\n\n"
        #         "Answer:"
        #     )
        # )
        prompt_template = PromptTemplate(
            input_variables=["chat_history","context", "question"],
            template=(
                "You are a knowledgeable assistant. Use the following context to answer the question. MAKE SURE TO MENTION THE DOCUMENTS PROVIDED\n\n"
                "Chat History:\n{chat_history}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                
                "Answer:"
            )
        )

        # Initialize the language model with the API key
        llm = ChatOpenAI(model_name="gpt-4o",
                     temperature=0.7,
                     max_tokens=1500,
                     openai_api_key=openai_api_key,
                     )

        # Create an LLMChain with the prompt template and the LLM
        llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
        #print(f"INPUT KEYS {llm_chain.input_keys}")

        # Generate the response by running the chain
        response = llm_chain.invoke({"context": context, "question": query, "chat_history": memoryarg})
        #print(f"GENERATE RTESPONSE RETURNING {response}")

        return str(response['text'])

    except Exception as e:
        print(f"An error occurred during response generation: {e}")
        return "I'm sorry, I couldn't process your request at this time."
