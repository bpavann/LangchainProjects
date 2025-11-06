import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent

st.title("🧠 Groq Agent on Hugging Face - LLM Course - WebBaseLoader 📘 ")
st.info("Built with LangChain, Groq, and Streamlit.")

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if "db" not in st.session_state:
    #Web Base Loader to load documents from Huggingface LLM course
    st.session_state.loader=WebBaseLoader("https://huggingface.co/learn/llm-course/en/chapter1/4")
    st.session_state.documents=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.docs=st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
    st.session_state.embeddings=OllamaEmbeddings(model="llama3.1",)
    st.session_state.db=Chroma(collection_name="GroqAgent", embedding_function=st.session_state.embeddings)
    st.session_state.doc_ids=st.session_state.db.add_documents(st.session_state.docs)


# Dynamic prompt middleware
@dynamic_prompt
def prompt_context(request: ModelRequest) -> str:
    """Add top-3 retrieved context and user query into system prompt."""
    # Get user input
    query = request.state["messages"][-1].text.strip() if request.state["messages"] else "<no question>"
    # Retrieve top-3 relevant chunks
    retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.invoke(query) if query else []
    # Combine retrieved text
    context = "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant context found."
    system_message = (
        f"""You are a helpful AI assistant. Use the following context to answer the question.
        Think step by step before providing a detailed answer.
        Context:
        {context}
        Question:
        {query}
        """
    )
    return system_message

#LLM Initialization
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_retries=2,
)

#Agent creation and Query input
agent = create_agent(llm, tools=[], middleware=[prompt_context])
query = st.text_input("Ask me question:", placeholder="e.g. What is a Large Language Model?")

if query and query.strip():
    with st.spinner("⏳ Running agent, please wait..."):
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

        # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(st.session_state.db.similarity_search(query, k=3)):
            st.write(doc.page_content)
            st.write("--------------------------------")
    st.success("✅ Response complete!")