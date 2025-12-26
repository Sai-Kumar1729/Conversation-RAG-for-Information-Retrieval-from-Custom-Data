import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

@st.cache_resource
def create_vector_db_from_pdf(file_path):
    """Processes the PDF and creates a searchable vector database."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Switch to a smaller, efficient embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = FAISS.from_documents(texts, embeddings)
    return db

def get_conversational_chain(vector_store):
    """Creates the conversational RAG chain."""
    llm = GoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.3)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return chain

# --- STREAMLIT UI ---
st.set_page_config(page_title="Conversational RAG Agent", layout="wide")
st.title("🗣️ Conversational Agent for Disaster Reports")
st.markdown("This agent can answer questions about the **2023 Sikkim Flash Floods**, based on the official NDMA situation report.")

# Path to the PDF
pdf_path = "sikkim_report.pdf"

# Create vector store and conversational chain
vector_db = create_vector_db_from_pdf(pdf_path)
conversation_chain = get_conversational_chain(vector_db)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if user_question := st.chat_input("Ask a question about the Sikkim flood report"):
    # Add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Get and display bot response
    with st.spinner("Thinking..."):
        result = conversation_chain({"question": user_question})
        bot_response = result['answer']
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)