import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

load_dotenv()
st.title("🧠 Huggingface Agent - Generative AI Course")
st.info("Built with LangChain, Huggingface, and Streamlit.")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if "db" not in st.session_state:
    # PDF Directory Loader to load documents from GenAI directory
    st.session_state.loader=PyPDFDirectoryLoader("./GenAI")
    st.session_state.documents=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.docs=st.session_state.text_splitter.split_documents(st.session_state.documents)
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    st.session_state.db = Chroma(collection_name="HuggingfaceAgent", embedding_function=st.session_state.embeddings)
    st.session_state.doc_ids = st.session_state.db.add_documents(st.session_state.docs)


# Dynamic prompt middleware
@dynamic_prompt
def prompt_context(request: ModelRequest) -> str:
    """Add top-3 retrieved context and user query into system prompt."""
    # Retrieve top-3 relevant chunks
    retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.invoke(query) if query else []
    # Combine retrieved text
    # Filter retrieved docs
    filtered_docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]

    # Build context from text only
    context = "\n\n".join(doc.page_content for doc in filtered_docs[:3]) if filtered_docs else "No relevant context found."
    system_message = (
        f"""You are a helpful AI assistant. Use the following context to answer the question.
        Think step by step before providing a detailed answer.
        Context:
        {context}
        Question: {query}
        """
    )
    return system_message

# LLM Initialization
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 30},
)
agent = create_agent(llm, tools=[], middleware=[prompt_context])

query = st.text_input("Ask me question:", placeholder="e.g. Explain about Generative AI ?")

if query and query.strip():
    start = time.process_time()
    output_placeholder = st.empty()
    response_text = ""

    # Stream response on same line
    start = time.process_time()
    for step in agent.stream(
         {"messages": [{"role": "user", "content": query}]},
         stream_mode="values",
         ):
         latest_message = step["messages"][-1].content
         response_text += latest_message + " "
         output_placeholder.markdown(f"**🧠 Response:** {response_text} \n")
    print("Response time:", time.process_time() - start)
    
    
