import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(page_title="RAG Application with Llama2", layout="wide")

# App title
st.title("📚 RAG Application with Llama2")

# Sidebar for document upload and settings
with st.sidebar:
    st.header("Settings")

    # Document upload
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["pdf", "txt"])

    # Model selection
    model_name = st.selectbox("Select Ollama Model", ["llama2:7b", "llama3.2:1b", "codellama:latest"])

    # Temperature setting
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # Process button
    process_button = st.button("Process Documents")

# Main content area
main_container = st.container()

# Initialize session state variables
if "processed" not in st.session_state:
    st.session_state.processed = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

# Function to save uploaded files
def save_uploaded_files(uploaded_files):
    # Create a documents directory if it doesn't exist
    if not os.path.exists("documents"):
        os.makedirs("documents")

    # Clear the directory
    for file in os.listdir("documents"):
        os.remove(os.path.join("documents", file))

    # Save the uploaded files
    for file in uploaded_files:
        with open(os.path.join("documents", file.name), "wb") as f:
            f.write(file.getbuffer())

    return "documents"

# Function to load documents
def load_documents(directory):
    documents = []

    # Load PDF files
    if any(file.endswith(".pdf") for file in os.listdir(directory)):
        pdf_loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())

    # Load text files
    if any(file.endswith(".txt") for file in os.listdir(directory)):
        text_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(text_loader.load())

    return documents

# Function to process documents
def process_documents(documents, model_name, temperature):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Create prompt template
    prompt_template = """
    You are a helpful AI assistant. Use the following context to answer the user's question.
    If you don't know the answer, just say you don't know. Don't make up an answer.

    Context: {context}

    Question: {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Initialize Ollama
    llm = OllamaLLM(model=model_name, temperature=temperature)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

# Process documents when button is clicked
if process_button and uploaded_files:
    with st.spinner("Processing documents..."):
        # Save uploaded files
        doc_dir = save_uploaded_files(uploaded_files)

        # Load documents
        documents = load_documents(doc_dir)

        if documents:
            # Process documents
            st.session_state.qa_chain = process_documents(documents, model_name, temperature)
            st.session_state.docs_processed = True
            st.success(f"Processed {len(documents)} documents")
        else:
            st.error("No documents were loaded. Please check the file formats.")

# Query input and response
with main_container:
    if st.session_state.docs_processed:
        st.header("Ask a question about your documents")
        query = st.text_input("Enter your question:")

        if query:
            with st.spinner("Generating answer..."):
                response = st.session_state.qa_chain.invoke(query)

                st.subheader("Answer")
                st.write(response["result"])
    else:
        st.info("Please upload documents and click 'Process Documents' to start.")
