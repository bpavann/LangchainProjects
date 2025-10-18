# Generative AI Powered Applications with LangChain: A Comprehensive Series

This repository contains the code and projects developed throughout the updated LangChain series, designed to guide users from scratch to advanced concepts in building Generative AI applications. The tutorials emphasize using the entire LangChain ecosystem, including associated libraries like LangSmith and LangServe.

The projects demonstrate how to build applications utilizing both paid LLM APIs (e.g., OpenAI) and various open-source LLM models.

## Key Features and Projects Covered

The repository explores several facets of modern AI engineering:

### 1. LangChain Ecosystem & Foundation Models
*   **Chatbot Creation:** Projects include simple chatbot applications built using paid LLMs (like `ChatOpenAI` and GPT-3.5 Turbo) and open-source models (like Llama 2 and Mistral).
*   **Local LLM Execution:** Demonstrations utilize **Ollama** to run large language models (LLMs) locally, which requires a system with good configuration (e.g., 64GB RAM).
*   **Monitoring and Tracing:** **LangSmith** is integrated throughout projects for monitoring, debugging, evaluation, and annotation. Environment variables are set to enable LangSmith tracking and trace LLM calls.

### 2. Deployment and APIs
*   **Production-Grade APIs:** Creating APIs for LLM models using **LangServe** (for deployment) and **FastAPI**.
*   **Route Integration:** APIs are designed with routes to allow interaction with multiple LLM models (e.g., OpenAI, Llama 2) based on the request path (e.g., `/essay`, `/poem`).
*   **Swagger UI Documentation:** The API setup provides auto-generated Swagger UI documentation for interaction.

### 3. Retrieval Augmented Generation (RAG) Pipelines
*   **Data Injection:** Techniques are shown for loading various data sources, including text files, PDFs (using `PyPDFLoader`), and web pages (using `WebBaseLoader`).
*   **Transformation and Chunking:** Documents are broken down into smaller chunks using the `RecursiveCharacterTextSplitter` to manage LLM context windows.
*   **Vector Embeddings and Storage:** Chunks are converted into vectors using embedding models (e.g., `OpenAIEmbeddings`, `OllamaEmbeddings`, `HuggingFaceBgeEmbeddings`) and stored in Vector Store databases like **ChromaDB** and **FAISS** (developed by Meta).
*   **Advanced RAG Concepts:** Implementation of the RAG pipeline using **Retrievers** (an interface that fetches relevant documents from the vector store) and **Chains** (specifically `create_stuff_document_chain` or `retrieval_QA`) to integrate the LLM, prompt, and context.

### 4. Agentic Applications and Inferencing Speed
*   **Multi-Search Agents:** Creation of advanced RAG applications using **Agents** and **Tools** to determine a sequence of actions and interact with multiple external data sources (like Wikipedia, ArXiv/RAG, and custom indices).
*   **Groq Inferencing Engine:** Utilizing the **Groq API** and its LPU (Language Processing Unit) system to achieve extremely fast inference speeds when using open-source LLM models like Llama 3 8B and Mistral.
*   **Hugging Face Integration:** Demonstrations of building Q&A RAG apps using LangChain integrated with Hugging Face components for models (like Mistral) and embeddings.

## Setup and Dependencies

To run these projects, you must set up your environment and install necessary dependencies.

1.  **Environment Setup:**
    *   Create and activate a virtual environment (e.g., using Conda: `conda create -p venv python=3.11`).
    *   Install Python dependencies using the provided `requirements.txt` file (e.g., `pip install -r requirements.txt`). Required packages include `langchain`, `fastapi`, `uvicorn`, `groq`, `chromadb`, and various third-party integration libraries (like `wikipedia`, `arxiv`, `sentence-transformers`).

2.  **API Keys and Environment Variables:**
    *   Create a `.env` file to store required credentials, which will be loaded using `load_dotenv`.
    *   The following environment variables are frequently required:
        *   `OPENAI_API_KEY`: Required for using OpenAI models and embeddings.
        *   `LANGCHAIN_API_KEY`: Used for connection to LangSmith.
        *   `GROQ_API_KEY`: Required for utilizing the Groq inferencing engine.
        *   `LANGCHAIN_TRACING_V2=true`: Enables tracing/monitoring in LangSmith.
        *   `LANGCHAIN_PROJECT`: Defines the project name for monitoring in LangSmith (e.g., `tutorial_one`).

## How to Run

Specific project details (e.g., running Streamlit apps or API servers) can be found within the individual project folders.

*   **Streamlit Applications:** Run using `streamlit run <app_file>.py` (e.g., `streamlit run app.py` or `streamlit run local_llama.py`).
*   **FastAPI Servers (APIs):** Run using `uvicorn app:app --host 0.0.0.0 --port 8000` (The server port is typically set to 8,000).