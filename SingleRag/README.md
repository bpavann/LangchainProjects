# 🦙 **SingleRAG Agent**
SingleRAG is an interactive **Retrieval-Augmented Generation (RAG)** demo built with **LangChain, Ollama LLM, and Chroma embeddings**. It allows users to upload a PDF and ask questions, and the AI agent answers using context extracted from the document.
The project is wrapped in **Streamlit** for easy interaction.

## 🔹 **Features**
- Upload a PDF and process it into document chunks.
- Vectorize document chunks using HuggingFace embeddings.
- Interactive RAG agent powered by Ollama LLM.
- Streamlined live responses in Streamlit, updated dynamically.
- Professional dark-themed interface with centered layout.
- LangSmith integration for traceable context (optional via API key).

## 🛠 **Required Tools**
- Python 3.10+
- LangChain for document handling and agent creation
- Ollama LLM for local LLM inference
- Chroma as vector store
- Streamlit for interactive web app
- HuggingFace Embeddings for text embeddings
- PyMuPDF for PDF loading

# **RAG / Agent Logic**
This section prepares the AI agent that answers questions based on your PDF content:

**Environment Setup**
```bash 
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
```
**Loads the LangSmith API key from .env for logging and tracking.**
**Enables tracing of requests to LangSmith if configured.**
**PDF Loading**
```bash
loader = PyMuPDFLoader("../SingleRag/SSLLM.pdf", mode="single")
document = loader.load()
```
**Loads a PDF file into memory.**
**Converts it into a document object that can be split into chunks.**
**Document Splitting**
```bash
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)
```
**Splits large PDFs into smaller chunks (1,000 characters per chunk, 200-character overlap).**
**This improves retrieval and LLM understanding.**
**Vector Store Creation**
```bash
db = Chroma(collection_name="SSLLM_collection", embedding_function=HuggingFaceEmbeddings())
doc_id = db.add_documents(chunks)
```
**Converts chunks into vector embeddings using HuggingFace embeddings.**
**Stores vectors in Chroma for fast similarity search during retrieval.**
**LLM Initialization**
```bash
llm = OllamaLLM(model="llama3.1")
```
**Initializes the Ollama local LLM for generating responses.**
**Dynamic Prompt Middleware**
```bash
@dynamic_prompt
def prompt_context(request: ModelRequest) -> str:
    ...
```
**Injects document context into the agent’s messages dynamically.**
**Ensures each user question is answered with relevant PDF context.**
**The middleware builds a “system message” including all chunk content.**
**Agent Creation**
```bash
agent = create_agent(llm, tools=[], middleware=[prompt_context])
```
**Creates the AI agent with the LLM and middleware.**
**The agent can stream responses to queries using document context.**

**Streamlit Interface**