import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes

load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"]="true"
os.environ["LANGSMITH_PROJECT"]="INTEGRATED_CHATBOT2"

app=FastAPI(
    title="Integrated Chatbot API",
    description="An API for a chatbot using LangChain and Ollama LLM",
    version="1.0.0"
)

#models
llm1=Ollama(model="llama3.1")
llm2=Ollama(model="gemma3")

#Prompt
prompt1=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps people find information."),
        ("user", "Please answer the following question in poem format: {question}")
     ]
)
prompt2=ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert assistant that provides detailed answers."),
        ("user", "Please answer the following question in essay format: {question}")
     ]
)

add_routes(app,prompt1|llm1,path="/PoemChatBot")
add_routes(app,prompt2|llm2,path="/EssayChatBot")

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)