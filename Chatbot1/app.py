import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

load_dotenv()

#LangSmith Tracing V2
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Chatbot1"


#Prompt
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps people find information."),
        ("user", "Question:{question}")
     ]
    )

#streamlit app
st.title("AI Powered Chatbot for Future")
input_text=st.text_input("Enter your question here")

#LLM
llm=Ollama(model="llama3.1",temperature=0.7,top_p=0.9,top_k=40)
outputparser=StrOutputParser()
chain=prompt|llm|outputparser

if input_text:
    st.write(chain.invoke({"question":input_text}))



