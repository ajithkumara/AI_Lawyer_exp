import streamlit as st
import os
import fitz  # PyMuPDF for fast PDF text extraction
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }

    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }

    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }

    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }

    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdf/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-embedding")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-lawyer")

if not os.path.exists(PDF_STORAGE_PATH):
    os.makedirs(PDF_STORAGE_PATH)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

## Since 1GB contains ~100,000 pages, you need an optimized text extraction method.
#  PyMuPDF is 10x faster than PyPDF2 and works well for legal PDFs.
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF efficiently."""
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

## Since retrieving 1GB at once is slow, break the text into smaller parts.
## Reduces query time by retrieving only relevant parts instead of scanning 1GB.
def chunk_text(text):
    """Split text into smaller chunks for efficient storage and retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)


## Instead of reading 1GB each time, store embeddings in FAISS and retrieve only relevant sections.
## Speeds up document retrieval 10x compared to reprocessing the full 1GB.
def initialize_vector_store(chunks):
    # Generate embeddings for the text chunks
    embeddings = EMBEDDING_MODEL.embed_documents(chunks)

    # Initialize the FAISS vector store
    vector_db = FAISS.from_embeddings(
        text_embeddings=list(zip(embeddings, chunks)),
        embedding=EMBEDDING_MODEL
    )

    # Store the vector store in the session state
    st.session_state.vector_db = vector_db

def index_documents(chunks):
    vector_db = st.session_state.vector_db  # This will now access the correctly initialized vector store
    vector_db.add_texts(chunks)

def find_related_documents(query):
    vector_db = st.session_state.vector_db
    return vector_db.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

uploaded_pdfs = st.file_uploader("Upload Research Documents (PDFs)", type="pdf", accept_multiple_files=True)

if uploaded_pdfs:
    for uploaded_pdf in uploaded_pdfs:
        saved_path = save_uploaded_file(uploaded_pdf)
        text = extract_text_from_pdf(saved_path)
        chunks = chunk_text(text)
    initialize_vector_store(chunks)  # Initialize the vector store first
    index_documents(chunks)  # Now index the documents
    st.success("✅ Documents processed successfully! Ask your questions below.")

user_input = st.chat_input("Enter your question about the documents...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    with st.spinner("Analyzing documents..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)
    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)
