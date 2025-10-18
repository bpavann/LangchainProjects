# 🤖 Integrated Chatbot API with LangChain & FastAPI

A production-ready **AI chatbot** system with multiple LLM endpoints, powered by **LangChain**, **Ollama**, and **FastAPI**.  
Includes a sleek **Streamlit** interface for real-time interaction with two specialized bots:  

- **Poem ChatBot (llama)** – Generates responses in **poem format**.  
- **Essay ChatBot (gemma)** – Generates responses in **essay format**.  

---

## 🚀 Overview

This project demonstrates a **robust LLM deployment pipeline**: APIs for LLMs with structured prompts and parsing logic, plus a modern UI for end users.  

It combines **local model execution**, **LangChain prompt management**, and **Streamlit UI**, ensuring fast, interactive, and controlled chatbot responses.

---

## 🧠 Key Features

- **Multiple LLM Endpoints** – Poem and Essay bots with custom prompts.  
- **Streamlit Frontend** – Real-time chat interface with dark-themed UI and gradient buttons.  
- **Error Handling** – Graceful handling of API and connection errors.  
- **Environment Configuration** – Secure `.env` management for API keys.  
- **Customizable Prompts** – Easily modify bot roles or response styles.

---

## ⚙️ System Architecture

**Workflow:**  

User Query → Streamlit UI → LangChain Prompt → Ollama LLM → Output Parser → Response Display  

**Components:**

- **Streamlit:** User-facing interface.  
- **LangChain:** Prompt templates, output parsing, and chain logic.  
- **Ollama:** Local LLM engine (Llama 3.1, Gemma 3).  
- **FastAPI + Uvicorn:** Production-ready API endpoints.  
- **Python-dotenv:** Environment variable management.

---

## 🛠️ Setup Instructions

1. **Clone the repository**  
```bash
git clone <repo-url>
cd <repo-directory>
```
2. **Create a virtual environment** 
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```pip install -r requirements.txt```

4. **Configure environment variables**
Create a .env file in the root directory:
```LANGCHAIN_API_KEY=your_api_key_here```

5. **Run the API**
```python app.py```

6. **Start Streamlit interface**
```streamlit run streamlit_app.py```

Open http://localhost:8501
Select a bot, enter a question, and click Generate Response.

**✨ Pavan Kumar Boddupally ✨**
