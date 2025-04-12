import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

def process_documents(documents, model_name="llama2:7b", temperature=0.5):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split documents into {len(chunks)} chunks")
    
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

def main():
    print("RAG Application with Llama2")
    print("---------------------------")
    
    # Load documents
    print("Loading documents from 'documents' directory...")
    documents = load_documents("documents")
    
    if not documents:
        print("No documents found. Please add PDF or TXT files to the 'documents' directory.")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    # Process documents
    print("Processing documents...")
    qa_chain = process_documents(documents)
    
    # Interactive Q&A
    print("\nYou can now ask questions about your documents. Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
        
        print("Generating answer...")
        response = qa_chain.invoke(query)
        
        print("\nAnswer:")
        print(response["result"])

if __name__ == "__main__":
    main()
