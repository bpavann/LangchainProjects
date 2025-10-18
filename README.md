## LangChain Series: Generative AI Application Engineering

This repository encapsulates projects developed to build generative AI applications utilizing the updated LangChain framework, ranging from foundational chatbots to advanced RAG pipelines and multi-agent systems. The methodology emphasizes the integration of the complete LangChain ecosystem, including associated libraries like LangServe and LangSmith.

Projects utilize both commercial LLM APIs (e.g., OpenAI) and various open-source models (Llama 2, Mistral, Llama 3).

### Core Technical Modules

| Module | Key Functionality & Components | Citation |
| :--- | :--- | :--- |
| **LangChain Ecosystem** | **LangSmith** is integrated for comprehensive monitoring, debugging, tracing, evaluation, and annotation of all LLM calls. **LangServe** is used alongside **FastAPI** and **Uvicorn** for creating production-grade REST APIs and deployment. | |
| **Foundation Models** | Chatbots developed using paid LLMs (`ChatOpenAI`, GPT-3.5 Turbo) and open-source models. **Ollama** is implemented for running models (like Llama 2) locally, suitable for high-configuration systems (e.g., 64GB RAM). | |
| **Retrieval Augmented Generation (RAG)** | Implements the full RAG pipeline: **Data Ingestion** (PDF, TXT, web pages using `PiPDFLoader`, `WebBaseLoader`). **Transformation** (Chunking large documents via `RecursiveCharacterTextSplitter`). **Vectorization** using multiple embedding models (OpenAI, Ollama, HuggingFace BGE). | |
| **Vector Stores & Retrieval** | Vectors stored in databases such as **ChromaDB** and **FAISS** (developed by Meta). Retrieval uses the **Retriever** interface to fetch relevant documents based on an unstructured query. Integration uses **Chains** (`create_stuff_document_chain`, `create_retrieval_chain`) to pass context to the LLM. | |
| **Advanced Agents** | Development of **Multi-Search Agents** using **Tools** (e.g., Wikipedia, ArXiv, and custom RAG indices). Agents use the LLM as a reasoning engine to determine the optimal sequence of actions and tool usage based on the user's inquiry. | |
| **Performance & Optimization** | Utilizes the **Groq Inferencing Engine** (LPU) for high-speed inference of open-source models (Llama 3, Mistral). Applications demonstrate deployment routes (APIs) supporting multiple LLMs (e.g., OpenAI route `/essay` and Llama 2 route `/poem`). | |

### Setup and Configuration

To execute the projects in this repository, follow these steps:

1.  **Environment Setup:** Create a virtual environment (e.g., using Python 3.11) and install all required packages via `pip install -r requirements.txt`.
2.  **API Keys:** Essential credentials must be set up in a `.env` file and loaded using `load_dotenv`:
    *   `OPENAI_API_KEY`: Required for OpenAI models and embeddings.
    *   `GROQ_API_KEY`: Required for utilizing the high-speed Groq Inferencing Engine.
    *   **LangSmith Configuration** (For tracing and monitoring):
        *   `LANGCHAIN_API_KEY`
        *   `LANGCHAIN_PROJECT` (Defines the project name)
        *   `LANGCHAIN_TRACING_V2=true`

3.  **Local LLMs (Optional):** If running open-source models locally, install and run **Ollama**. Models must be pulled first (e.g., `ollama run llama2`).