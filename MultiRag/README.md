# 🤖 MultiRAG Agent
An intelligent Multi-Retrieval Augmented Generation (MultiRAG) system built with LangChain, Ollama, and Streamlit.
This agent retrieves and synthesizes information from multiple reliable knowledge sources — Wikipedia, Arxiv, and Model Context Protocol (MCP) — to generate precise and well-informed answers.

## 🚀 Overview
**The MultiRAG Agent combines multiple retrieval tools into a single intelligent assistant that:**
- Searches Wikipedia for general information 🌐
- Retrieves academic research papers from Arxiv 📚
- Extracts and summarizes AI framework documentation using Model Context Protocol (MCP) 🧠
It uses a Large Language Model (LLM) from Ollama (Llama3.1) and dynamically injects context to produce accurate, step-by-step answers.

##🧩 Key Features
- ✅ Multi-source Retrieval:
Fetches knowledge from Wikipedia, Arxiv, and web documents.
- ✅ Vector Database (Chroma):
Stores and searches text chunks for similarity-based retrieval.
- ✅ Dynamic Prompting:
Adds document context dynamically to improve LLM responses.
- ✅ Ollama LLM Integration:
Uses the open-source Llama3.1 model for reasoning and text generation.
- ✅ Streamlit Interface:
User-friendly frontend for live interaction and visualization.

## ⚙️ How It Works (Step-by-Step)
1. Load Environment and Libraries
- Imports required modules (LangChain, Ollama, Streamlit, etc.).
- Loads environment variables using dotenv.
2. Define Tools
- Wikipedia Tool → Uses WikipediaAPIWrapper for short summaries.
- Arxiv Tool → Fetches relevant research paper abstracts.
- Web Tool → Loads data from Model Context Protocol Docs, splits into text chunks, and stores in a Chroma vector database.
3. Create Retriever
- Uses HuggingFace sentence-transformer embeddings for semantic similarity.
- Converts documents into embeddings and stores them in Chroma DB.
4. Define Dynamic Prompt
- A custom function injects document context into the LLM prompt dynamically, helping the model produce contextually accurate answers.
5. Create Agent
- Combines all tools into one unified LangChain Agent using:
```bash
agent = create_agent(llm, tools=Tools, middleware=[prompt])
```
6. Streamlit Interface
- Displays a simple UI for user queries.
- Shows which tools were used and the AI’s response in real time.

