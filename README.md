# 🤖 Generative AI Powered Applications with LangChain

This repository showcases a complete collection of **production-grade Generative AI applications** built using the **LangChain ecosystem** — including **LangSmith**, **LangServe**, and **FastAPI**.  
It is designed to help developers transition from foundational LangChain concepts to advanced **LLM engineering**, **deployment**, and **observability** using both **open-source** and **paid** large language models.

---

## 🚀 Overview

This repository serves as a **hands-on LangChain learning and development suite**, covering everything from **chatbot creation** to **retrieval-augmented generation (RAG)** and **agentic systems**.

Projects combine:
- **Paid APIs** such as `OpenAI` for cloud-based reasoning.  
- **Open-source LLMs** such as `Llama 3`, `Mistral`, and `Gemma` through **Ollama** and **Groq** for local or high-speed inference.

The primary goal is to bridge the gap between **LLM architecture**, **local inference**, and **enterprise-level API deployment**.

---

## 🧠 Key Features & Projects

### 1️⃣ LangChain Ecosystem & Foundation Models
- **Chatbot Development:**  
  Interactive bots built using open-source alternatives (`Ollama`, `gemma`).
- **Local LLM Execution:**  
  Run LLMs locally with **Ollama**, ensuring full privacy and offline capability (recommended: ≥ 64GB RAM).
- **Tracing and Monitoring:**  
  Integrated **LangSmith** for tracking, debugging, and evaluating model performance using LangChain Tracing V2.

---

### 2️⃣ Deployment & APIs
- **Production-Grade APIs:**  
  Built using **FastAPI** and **LangServe** to deploy multiple LLMs as independent API routes.
- **Multi-Model Routing:**  
  Define and serve multiple endpoints (e.g., `/poem`, `/essay`) linked to different models and prompts.
- **Interactive Swagger UI:**  
  Auto-generated API documentation for easy testing and exploration.

---

### 3️⃣ Retrieval-Augmented Generation (RAG) Pipelines
- **Data Loading:**  
  Support for PDFs, text files, and web pages using loaders like `PyPDFLoader` and `WebBaseLoader`.
- **Chunking & Preprocessing:**  
  Use `RecursiveCharacterTextSplitter` to split documents into context-friendly chunks.
- **Vector Embedding & Storage:**  
  Transform documents into embeddings using models like:
  - `OpenAIEmbeddings`  
  - `OllamaEmbeddings`  
  - `HuggingFaceBgeEmbeddings`  
  Store vectors in **ChromaDB** or **FAISS** for fast retrieval.
- **RAG Integration:**  
  Combine retrievers and chains (`create_stuff_document_chain`, `retrieval_QA`) for contextual reasoning.

---

### 4️⃣ Agentic Systems & Inferencing Speed
- **Multi-Source Agents:**  
  Implement agents that can query multiple data sources (Wikipedia, ArXiv, or custom vector stores).
- **Groq LPU Integration:**  
  Leverage **Groq**'s inferencing engine for ultra-fast response times on open LLMs like `Llama3` or `gemma`.
- **Hugging Face Integration:**  
  Integrate with Hugging Face models and embeddings to enhance retrieval-based applications.

---
### 5️⃣ MedicoAgent — Clinical ReAct RAG Assistant
- **Purpose:**
MedicoAgent is a ReAct (Reason + Act)–based medical assistant designed for clinical reasoning, knowledge retrieval, and metric computation using LangChain, FAISS, and a local LLM (Ollama Llama 3.1).
It demonstrates how retrieval-augmented reasoning pipelines can work entirely offline, maintaining both data privacy and explainability.
- **Features:**
- Built with Streamlit UI for an interactive experience.
  - Uses local LLM (Ollama – Llama 3.1) for private, offline inference.
  - Employs FAISS vector search and MiniLM-L6 embeddings for efficient document retrieval.
  - Integrates LangChain ReAct framework for reasoning-based action steps (calculations, retrieval, summarization).
  - Includes built-in safety disclaimers to ensure ethical use and educational purpose only.
- **Architecture:**
The system connects the Streamlit frontend, LangChain ReAct controller, FAISS retriever, and Ollama LLM, forming a fully local, privacy-focused pipeline.
---

## ⚙️ Setup & Installation

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
