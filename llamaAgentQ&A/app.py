from langchain_ollama import ChatOllama
import streamlit as st

# Initialize local Llama 3.1 model
llm = ChatOllama(model="llama3.1")

# Function to get Llama response
def get_llama_response(prompt):
    response = llm.invoke(prompt)
    return response.content

# Streamlit UI
st.set_page_config(page_title="Llama 3.1 Chat Demo")
st.header("🦙 Local Llama 3.1 Chatbot")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input box
user_input = st.text_input("You:", key="input")
submit = st.button("Ask")

if submit and user_input:
    # Get response
    response = get_llama_response(user_input)

    # Save history
    st.session_state['chat_history'].append(("You", user_input))
    st.session_state['chat_history'].append(("Llama", response))

    # Display
    st.subheader("Response:")
    st.write(response)

# Display history
st.subheader("Chat History:")
for role, text in st.session_state['chat_history']:
    st.write(f"**{role}:** {text}")
