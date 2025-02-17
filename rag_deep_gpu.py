import streamlit as st
import logging
import time
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.markdown("""
    <style>
    .stApp {
        background-color: black;
        color: white;
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

# Enable GPU for Embeddings and LLM
try:
    start_time = time.time()
    EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
    DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
    LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b", options={"num_gpu_layers": 32})
    logging.info(f"Models initialized successfully in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    logging.error(f"Error initializing models: {e}")
    raise

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    try:
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        logging.info(f"File saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        raise
    return file_path

def load_pdf_documents(file_path):
    try:
        start_time = time.time()
        document_loader = PDFPlumberLoader(file_path)
        docs = document_loader.load()
        logging.info(f"Loaded {len(docs)} pages in {time.time() - start_time:.2f} seconds.")
        return docs
    except Exception as e:
        logging.error(f"Error loading document: {e}")
        raise

def chunk_documents(raw_documents):
    try:
        start_time = time.time()
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_processor.split_documents(raw_documents)
        logging.info(f"Chunked into {len(chunks)} pieces in {time.time() - start_time:.2f} seconds.")
        return chunks
    except Exception as e:
        logging.error(f"Error chunking documents: {e}")
        raise

def index_documents(document_chunks):
    try:
        start_time = time.time()
        DOCUMENT_VECTOR_DB.add_documents(document_chunks)
        logging.info(f"Indexed documents in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error indexing documents: {e}")
        raise

def find_related_documents(query):
    try:
        start_time = time.time()
        results = DOCUMENT_VECTOR_DB.similarity_search(query)
        logging.info(f"Retrieved {len(results)} related documents in {time.time() - start_time:.2f} seconds.")
        return results
    except Exception as e:
        logging.error(f"Error retrieving related documents: {e}")
        raise

def generate_answer(user_query, context_documents):
    try:
        start_time = time.time()
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        response_chain = conversation_prompt | LANGUAGE_MODEL
        response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
        logging.info(f"Generated answer in {time.time() - start_time:.2f} seconds.")
        return response
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        raise

# Streamlit UI
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)
if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    st.success("✅ Document processed successfully! Ask your questions below.")

    user_input = st.chat_input("Enter your question about the document...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
