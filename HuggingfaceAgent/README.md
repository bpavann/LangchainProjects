# **🧠 HuggingFace Agent - Generative AI Course**
**🤖 AI-Powered Document Q&A using LangChain, Hugging Face & Streamlit**
* This project demonstrates how to build an interactive Generative AI Agent that can **read**, **embed**, and **answer questions** from PDFs in a directory using **Hugging Face models** integrated with **LangChain** and **Streamlit**.
* It dynamically retrieves the most relevant context from documents and generates responses using a Hugging Face text generation model (e.g., gpt2).

## 🚀 **Features**
- 📚 **Automatic PDF loading** — Reads all PDF files from a given directory.
- 🔍 **Text chunking and embeddings** — Splits text into manageable pieces and generates embeddings using sentence-transformers/all-mpnet-base-v2.
- 🧠 **Vector database storage** — Stores document embeddings using ChromaDB for similarity search.
- 💬 **Dynamic prompt middleware** — Injects retrieved document context into the agent’s prompt dynamically.
- ⚡ **Hugging Face LLM integration** — Uses Hugging Face’s text-generation pipeline (e.g., GPT-2) for response generation.
- 🎨 **Streamlit UI** — Interactive frontend for asking questions and viewing responses in real-time.

## 🧩 **Tech Stack**
| Component |	Purpose |
|-----------|------------|
| LangChain	| Orchestrates agents, embeddings, retrieval, and middleware |
| Hugging Face |	Provides LLM and embedding models |
| ChromaDB |	Vector database for document similarity search |
| Streamlit |	Frontend UI for interaction |
| dotenv |	Manages API keys and environment variables |

## 🧠 **Code Walkthrough**
1. Environment Setup
```bash
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
Loads Hugging Face API key from the .env file to authenticate model access.
```
2. PDF Loading and Splitting
```bash
loader = PyPDFDirectoryLoader("./GenAI")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)
Reads all PDF files from the GenAI/ directory.
Splits large documents into overlapping chunks for better retrieval.
```
3. Embedding and Vector Store
```bash
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(collection_name="HuggingfaceAgent", embedding_function=embeddings)
doc_ids = db.add_documents(docs)
Converts document chunks into embeddings.
Stores them in a Chroma vector database for semantic search.
```
4. Dynamic Prompt Middleware
```bash
@dynamic_prompt
def prompt_context(request: ModelRequest) -> str:
    retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.invoke(query) if query else []
    filtered_docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]
    context = "\n\n".join(doc.page_content for doc in filtered_docs[:3])
    return f"""You are a helpful AI assistant...
               Context: {context}
               Question: {query}"""
This middleware automatically:
Retrieves top 3 relevant chunks from the database.
Builds a context-aware prompt for the model dynamically.
```
5. Model Initialization
```bash
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 30},
)
agent = create_agent(llm, tools=[], middleware=[prompt_context])
Uses GPT-2 for text generation.
Creates an agent wrapped with middleware for dynamic contextual reasoning.
```
6. Interactive Streamlit Interface
```bash
query = st.text_input("Ask me question:", placeholder="e.g. Explain about Generative AI ?")

if query:
    for step in agent.stream(
         {"messages": [{"role": "user", "content": query}]},
         stream_mode="values",
    ):
        response_text += step["messages"][-1].content + " "
        output_placeholder.markdown(f"**🧠 Response:** {response_text}")
```
Accepts user input.
Streams model-generated responses in real time.

## 📊 **Example Usage**
**Input:**
| What is Generative AI?
**Output:**
| Generative AI refers to models capable of creating new data — such as text, images, or music — similar to the data they were trained on. It uses deep learning techniques like transformers and diffusion models to generate original outputs.