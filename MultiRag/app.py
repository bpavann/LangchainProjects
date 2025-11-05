
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.tools.retriever import create_retriever_tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"]="true"
os.environ["LANGSMITH_PROJECT"]="MultiRAG"

#WIKIPEDIA BASED TOOLS:-1
api_wrapper=WikipediaAPIWrapper(top_k_results=5,doc_content_char_limit=500)
wikipedia=WikipediaQueryRun(api_wrapper=api_wrapper)

#ARXIV BASED TOOLS:-2
arxiv_wrapper=ArxivAPIWrapper(top_k_results=5,doc_content_char_limit=500)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

#WEBBASED LOADER TOOLS:-3
loader=WebBaseLoader("https://modelcontextprotocol.io/docs/getting-started/intro")
document=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(document)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

db=Chroma(collection_name="MultiRag",embedding_function=embeddings)
doc_id=db.add_documents(text_splitter)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})
web_tool = create_retriever_tool(
    retriever,
    "model context protocol",
    "Search for information about Model Context Protocol. For any questions about Model Context Protocol, you must use this tool!",
)

#ALL TOOLS COMBINED:-4
Tools=[wikipedia,arxiv,web_tool]

# Create Ollama LLM
llm=ChatOllama(model="llama3.1")

#Creating DYNAMIC PROMPT FUNCTION
@dynamic_prompt
def prompt_context(request: ModelRequest) -> str:
    """Add top-3 retrieved context and user query into system prompt."""
    # Get user input
    query = request.state["messages"][-1].text.strip() if request.state["messages"] else "<no question>"
    # Retrieve top-3 relevant chunks
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
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

agent=create_agent(llm,tools=Tools,middleware=[prompt_context])

# response = agent.invoke({"messages": [{"role": "user", "content": "Summarize the research papers about Transformer model "}]})
# print("Agent response:", response)

# Streamlit App Configuration
st.markdown("## 📘 MultiRAG Agent")
st.write("""
This intelligent agent retrieves and synthesizes information from:
- **Wikipedia** 🌐 for general knowledge  
- **Arxiv** 📚 for academic research papers  
- **Model Context Protocol (MCP)** 🧠 for AI framework details  
""")
st.info("Built with LangChain, Ollama, and Streamlit.")
st.divider()


# User input
query = st.text_input("Ask me anything:", placeholder="e.g. What is the meaning of MCP?")

if st.button("Ask") or query:
    if query.strip():
        with st.spinner("Thinking..."):
            response = agent.invoke({"messages": [{"role": "user", "content": query}]})

            # Extract tools used and clean AI response
            tool_names = []
            final_answer = ""

            for msg in response.get("messages", []):
                # Detect tools used
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool in msg.tool_calls:
                        tool_names.append(tool.get("name", "Unknown Tool"))

                # Get final AI response
                if getattr(msg, "content", None) and msg.__class__.__name__ == "AIMessage":
                    if msg.content.strip():
                        final_answer = msg.content.strip()

            tool_names = list(set(tool_names))  

          
            # Display results
            if tool_names:
                st.info(f"🧩 **Tool(s) Used:** {', '.join(tool_names)}")
            else:
                st.info("🧩 **Tool(s) Used:** None")

            st.success("🧠 **Agent Response:**")
            st.write(final_answer or "No response generated.")
    else:
        st.warning("Please enter a question to start.")
