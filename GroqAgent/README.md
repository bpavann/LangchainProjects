# **🧠 Groq Agent on Hugging Face — LLM Course WebBaseLoader 📘**
- This Streamlit application demonstrates how to build a retrieval-augmented AI agent powered by LangChain, Groq, and Ollama - embeddings.
- It dynamically loads and indexes learning materials from the Hugging Face LLM Course, retrieves the most relevant context for any user query, and generates real-time answers using Groq’s high-speed inference API.

---

## 🚀 Key Features
* **WebBaseLoader Integration** – Automatically loads online learning content (e.g., Hugging Face LLM course chapters).
* **Vector Store with Chroma** – Stores document embeddings for efficient semantic retrieval.
* **Dynamic Prompt Middleware** – Retrieves top-3 relevant chunks to augment every LLM query dynamically.
* **Groq LLM Backend** – Uses llama-3.1-8b-instant via Groq API for ultra-fast reasoning.
* **Streamlit UI** – Interactive, clean, and beginner-friendly interface with live streaming responses.
* **End-to-End RAG Pipeline** – From document loading → chunking → embedding → retrieval → LLM response.

---

## 🧩 Tech Stack
| Component | Library/Service |
|----------|--------------|
| Frontend |	Streamlit |
| LLM Backend |	Groq API (llama-3.1-8b-instant)|
| Framework |	LangChain |
| Embeddings |	Ollama (llama3.1) |
| Vector Database |	Chroma |
| Document Loader |	WebBaseLoader |
| Environment Management |	python-dotenv|

---

## 📦 Installation
1️⃣ *Clone the repository*
```bash
git clone https://github.com/bpavann/LangchainProjects/tree/main/GroqAgent
cd GroqAgent
```
2️⃣ *Create and activate a virtual environment*
```bash
python3 -m venv venv
source venv/bin/activate    # for Linux/Mac
venv\Scripts\activate       # for Windows
```
3️⃣ *Install dependencies*
```bash
pip install -r requirements.txt
```
4️⃣ *Add your Groq API key*
```bash
Create a .env file in the project root:
GROQ_API_KEY=your_api_key_here
```
▶️ *Run the Application*
```bash
streamlit run app.py
Then open your browser at http://localhost:8501
```
---

## 💡 Usage
* The app automatically loads the Hugging Face LLM course content (Chapter 1.4).
* It splits the text into 1000-character chunks with 200-character overlaps.
* Each chunk is embedded using Ollama embeddings and stored in Chroma DB.
* When you enter a query, the app:
* Retrieves the top-3 most similar chunks.
* Injects them into a dynamic prompt using LangChain’s middleware.
* Sends the enriched prompt to Groq’s llama-3.1-8b-instant model.
* Streams the generated response live in the Streamlit UI.

---

## 🧠 Code Explanation
1️⃣ **Environment Setup**
```bash
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
Loads the Groq API key from .env.
```
2️⃣ **Document Loading & Chunking**
```bash
WebBaseLoader("https://huggingface.co/learn/llm-course/en/chapter1/4")
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
Fetches and splits the Hugging Face course material into small, context-rich text chunks.
```
3️⃣ **Embedding & Vector Storage**
```bash
embeddings = OllamaEmbeddings(model="llama3.1")
db = Chroma(collection_name="GroqAgent", embedding_function=embeddings)
db.add_documents(docs)
Encodes the text into numerical vectors and stores them in a Chroma vector database for retrieval.
```
4️⃣ **Dynamic Prompt Context Injection**
```bash
@dynamic_prompt
def prompt_context(request: ModelRequest) -> str:
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.invoke(query)
This middleware retrieves the top-3 relevant document chunks and injects them into the system prompt before sending it to the LLM — forming the backbone of the Retrieval-Augmented Generation (RAG) approach.
```
5️⃣ **LLM & Agent Creation**
```bash
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
agent = create_agent(llm, tools=[], middleware=[prompt_context])
Initializes the Groq LLM and attaches the dynamic prompt middleware to create a context-aware agent.
```
6️⃣ **Real-Time Query Interface**
```bash
query = st.text_input("Ask me question:")
for step in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
    ...
Streams responses token-by-token directly into the Streamlit UI for a smooth conversational feel.
```
7️⃣ **Document Similarity Display**
```bash
for doc in db.similarity_search(query, k=3):
    st.write(doc.page_content)
Shows the top-3 most relevant document snippets retrieved for transparency.
```

---

## 📊 Output Example

* User Query:
>What is a Large Language Model?
* Agent Response:
>A Large Language Model (LLM) is a type of deep neural network trained on vast text data to understand and generate human-like language...

---

## 🔒 Environment Variables
| Variable |	Description |
|----------|--------------|
| GROQ_API_KEY |	API key to access Groq inference services |