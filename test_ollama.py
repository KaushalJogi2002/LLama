from langchain_ollama import OllamaLLM

# Initialize Ollama
llm = OllamaLLM(model="llama2:7b")

# Test with a simple query
response = llm.invoke("What is RAG (Retrieval-Augmented Generation)?")
print(response)
