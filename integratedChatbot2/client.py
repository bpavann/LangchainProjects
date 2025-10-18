import streamlit as st
import requests

def get_llama_response_url(input_text):
    response=requests.post("http://localhost:8000/PoemChatBot/invoke",json={"input":{'question':input_text}})
    response.raise_for_status()
    return response.json()["output"]

def get_gemma_response_url(input_text):
    response=requests.post("http://localhost:8000/EssayChatBot/invoke",json={"input":{'question':input_text}})
    response.raise_for_status()
    return response.json()["output"]


# Streamlit
st.set_page_config(
    page_title="LangChain ChatBot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    body {
        background-color: #0A0F2A;
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
    }
    .css-1d391kg {  /* main container */
        max-width: 800px;
        margin: 0 auto;
        text-align: center;
    }
    h1 {
        color: #00BFFF;
        font-weight: 700;
    }
    h2, h3, h4 {
        color: #87CEFA;
    }
    .stTextArea>div>div>textarea {
        background-color: #1A2B5C;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1E3C72, #2A5298);
        color: #FFFFFF;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2A5298, #1E3C72);
        color: #FFFFFF;
    }
    .bot-response {
        background-color: #1A2B5C;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        font-size: 16px;
        text-align: left;
        color: #FFFFFF;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# UI Elements
# -----------------------------
st.title("AI Powered ChatBot For Future 🤖 ")
st.write("Interact with llama and gemma bots seamlessly.", unsafe_allow_html=True)

# Bot selection
bot_choice = st.radio(
    "Choose a Bot",
    ("Poem ChatBot (llama)", "Essay ChatBot (gemma)"),
    horizontal=False
)

# User input
user_input = st.text_area(
    "Enter your topic/question here:",
    height=150
)

# Generate response
if st.button("Generate Response"):
    if not user_input.strip():
        st.warning("Please enter a topic or question!")
    else:
        try:
            if bot_choice == "Poem ChatBot (llama)":
                output_text = get_llama_response_url(user_input)
            else:
                output_text = get_gemma_response_url(user_input)

            st.subheader("Bot Response:")
            st.markdown(f'<div class="bot-response">{output_text}</div>', unsafe_allow_html=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")