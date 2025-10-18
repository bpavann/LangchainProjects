# 🤖 AI-Powered Chatbot

An intelligent, lightweight chatbot that brings the power of **local Large Language Models (LLMs)** to your fingertips.  
Built with **LangChain**, **Ollama**, and **Streamlit**, this project showcases how modern AI pipelines can run efficiently on local hardware — offering real-time responses, customizable prompts, and seamless integration with LangSmith for tracing and observability.

---

## 🚀 Overview

This project demonstrates an **AI-driven assistant** capable of understanding and responding to user queries through a sleek Streamlit interface.  
It connects a **local Ollama model (Llama 3.1)** with **LangChain’s prompt management and output parsing**, ensuring structured reasoning and professional-quality results.

The goal is to bridge **LLM reasoning**, **local execution**, and **user-friendly interaction**, creating a foundation for future agentic AI systems.

---

## 🧠 Key Features

- **Local Model Execution** – Runs entirely offline using Ollama’s Llama 3.1 or other compatible models.  
- **LangChain Integration** – Structured prompt templates, output parsing, and chain logic for controlled AI responses.  
- **LangSmith Tracing V2** – Visualize and debug LLM reasoning steps with full experiment tracking.  
- **Streamlit Frontend** – Interactive web UI with clean design and instant feedback.  
- **Environment-Based Configuration** – Secure setup using `.env` variables for API keys and project metadata.  
- **Customizable Prompts** – Adapt the chatbot’s role or domain (education, healthcare, coding, etc.) with minimal edits.

---

## ⚙️ System Architecture

**Workflow:**

User Query → Streamlit Interface → LangChain Prompt → Ollama LLM → Output Parser → Display Response

**Core Components:**
- **Streamlit:** UI layer for real-time chat interaction.  
- **LangChain:** Framework for chaining prompts, models, and outputs.  
- **Ollama:** Local LLM engine for efficient inference.  
- **LangSmith:** Observability platform for tracking and optimizing runs.  
- **Python-dotenv:** Secure environment management.

---

## 🛠️ Setup Instructions

1. **Clone the repository**  
   Download or clone this project from GitHub to your local machine.

2. **Install dependencies**  
   Ensure all Python dependencies are installed (see `requirements.txt`).

3. **Install and configure Ollama**  
   - Install Ollama for your operating system.  
   - Pull the model `llama3.1` (or another of your choice).  
   - Start the Ollama server before running the app.

4. **Set environment variables**  
   Create a `.env` file in the project root containing your LangChain and LangSmith credentials:

5. **Run the application**  
Launch the Streamlit app and open the local URL displayed in your terminal.

---

## 🧩 How It Works

The chatbot processes each query through a structured LangChain pipeline:
- The **prompt template** defines the system and user instructions.
- The **Ollama model** (Llama 3.1) generates a natural, context-aware response.
- The **output parser** cleans and structures the response before displaying it.
- **LangSmith tracing** logs every run for evaluation and refinement.

This architecture ensures both **clarity** and **control** over model behavior, suitable for educational and experimental use.

---

## 🪄 Customization Options

- **Switch Models:** Replace `llama3.1` with `mistral`, `phi3`, or any model supported by Ollama.  
- **Modify System Prompts:** Redefine the assistant’s behavior to act as a tutor, researcher, or content generator.  
- **Extend Functionality:** Add retrieval-based reasoning (RAG), memory modules, or domain-specific tool calls.  
- **Adjust Sampling Parameters:** Tune temperature, top-p, and top-k for balanced creativity and determinism.

---

## 📊 Observability with LangSmith

Integrated **LangSmith Tracing V2** allows for:
- Step-by-step visualization of LLM reasoning.
- Evaluation of output quality and latency.
- Debugging prompt and chain configurations.
- Experiment management across multiple sessions.

This provides a clear window into the chatbot’s decision-making process, enabling rapid improvements.

---


**✨ Pavan Kumar Boddupally ✨**
