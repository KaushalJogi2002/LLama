# RAG Application with Llama2

This is a Retrieval-Augmented Generation (RAG) application built with Python, LangChain, and Ollama using the Llama2 model. It features a minimalist UI built with Streamlit.

## Features

- Upload PDF and text documents
- Process documents for retrieval
- Ask questions about your documents
- Get AI-generated answers based on the content of your documents

## Requirements

- Python 3.8+
- Ollama installed with Llama2 model
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure Ollama is installed and the Llama2 model is available:

```bash
ollama list
```

If the model is not available, you can pull it:

```bash
ollama pull llama2:7b
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload your documents (PDF or TXT files)

4. Click "Process Documents" to index your documents

5. Ask questions in the text input field and get answers based on your documents

## How it Works

This application uses LangChain to:

1. Load and split documents into chunks
2. Create embeddings using HuggingFace's sentence-transformers
3. Store the embeddings in a FAISS vector database
4. Retrieve relevant document chunks based on the user's query
5. Generate a response using the Llama2 model through Ollama

## Customization

You can customize the application by:

- Changing the model in the dropdown menu
- Adjusting the temperature slider to control the randomness of responses
- Modifying the code to support additional document types
