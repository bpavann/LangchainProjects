import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["ASTRA_DB_API_ENDPOINT"] = os.getenv("ASTRA_DB_API_ENDPOINT")
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

st.title("Gemma Model Document Q&A")

def vectore_embedding():
    if "db" not in st.session_state:
        st.session_state.loader= PyPDFDirectoryLoader("./GenAI")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.embeddings=OllamaEmbeddings(model="llama3.1")
        st.session_state.db=Chroma(collection_name="Ollama_gemma_docs",embedding_function=st.session_state.embeddings)
        st.session_state.db_ids=st.session_state.db.add_documents(st.session_state.documents)

system_prompt = """Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
llm=ChatGroq(model_name="llama-3.1-8b-instant")

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=system_prompt
)
if st.button("Documents Embedding"):
    vectore_embedding()
    st.write("Vector Store DB Is Ready")

query = st.text_input("Ask me question:", placeholder="e.g. What is GenAI?")

if query and query.strip():
    vectore_embedding()
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