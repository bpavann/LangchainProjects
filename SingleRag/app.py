import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"]="true"
os.environ["LANGSMITH_PROJECT"]="SingleRAG"

# Dynamic prompt middleware
@dynamic_prompt
def prompt_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})
    docs_content = "\n\n".join(f"{doc.page_content}\nMetadata: {doc.metadata}" for doc in document)
    system_message = (
        """You are a helpful AI assistant. Use the following context to answer the question.
        Think step by step before providing a detailed answer."""
        f"\n\n{docs_content}"
    )
    return system_message


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SingleRAG Agent Demo",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="📘",
)

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    body { background-color: #0E1117; color: #F5F5F5; }
    .main .block-container { max-width: 800px; margin-left: auto; margin-right: auto; padding-top: 30px; }
    .stTextInput>div>div>input { background-color: #1E1E1E; color: #F5F5F5; border-radius: 6px; border: 1px solid #444; padding: 8px; }
    .stButton>button { background-color: #2563EB; color: white; border-radius: 6px; padding: 0.5em 1em; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #1D4ED8; cursor: pointer; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER ---
st.title("SingleRAG Agent Interactive Demo")
st.markdown("Upload a PDF and ask questions. The AI agent will answer using context from your document.")

# --- PDF UPLOAD ---
uploaded_file = st.file_uploader("📂 Upload your PDF", type=["pdf"])
if uploaded_file is not None:
    temp_path = "uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded successfully!")

    # Use existing loader and chunks
    loader = PyMuPDFLoader(temp_path, mode="single")
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)

    # Create vector store
    db = Chroma(collection_name="SSLLM_collection", embedding_function=HuggingFaceEmbeddings())
    doc_id = db.add_documents(chunks)

    # Reinitialize Ollama LLM and agent
    llm = OllamaLLM(model="llama3.1")
    agent = create_agent(llm, tools=[], middleware=[prompt_context])

    # --- QUESTION INPUT ---
    query = st.text_input("💬 Enter your question:")
    run_query = st.button("🔍 Ask")

    if run_query and query.strip():
        st.info("⏳ Running agent, please wait...")
        output_placeholder = st.empty()
        response_text = ""

        # Stream response on same line
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            latest_message = step["messages"][-1].content
            response_text += latest_message + " "
            output_placeholder.markdown(f"**🧠 Response:** {response_text}")

        st.success("✅ Response complete!")
else:
    st.info("👆 Upload a PDF to start interacting with the agent.")
