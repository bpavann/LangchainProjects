
# Required Libraries
import os
import re
import json
import faiss
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel,Field,AnyHttpUrl
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"]="true"
os.environ["LANGSMITH_PROJECT"]="MEDICO_AGENT"

# Safety Disclaimer and System Instructions
SAFETY_DISCLAIMER = (
    "This system is for educational and research purposes only. "
    "It does not provide medical advice, diagnosis, or treatment recommendations. "
    "Outputs are intended to support learning and hypothesis generation. "
    "Users should always verify findings with a qualified healthcare professional."
)
SYSTEM_INSTRUCTIONS = f"""
You are an educational diagnosis-support assistant designed for clinicians-in-training.
{SAFETY_DISCLAIMER}

Your role:
- Assist in reasoning through clinical information.
- Summarize objective parameters (e.g., BMI, MAP, Anion Gap) when relevant.
- Retrieve concise educational insights from the local RAG corpus.
- Use computational tools only when necessary. If a tool fails, acknowledge the limitation and continue reasoning.

Follow this structured output format:
1. Red Flags: Immediate concerns that require urgent attention.
2. Key Missing Questions: Up to 4 relevant clarifications or data points needed.
3. Calculated Parameters: Any computed results (BMI, MAP, etc.).
4. Hypotheses (Educational Only): Ranked differential possibilities; not definitive diagnoses.
5. Next-Step Considerations: Recommended lines of reasoning or evaluation.
6. Citations: Brief sources or corpus excerpts used for reference.
7. Safety Note: {SAFETY_DISCLAIMER}
"""

# Corpus
CORPUS = [
    # Body Temperature
    (
        "Normal adult body temperature ranges between 97°F (36.1°C) and 99°F (37.2°C). "
        "A fever is typically defined as a temperature above 100.4°F (38°C). "
        "Hypothermia occurs when body temperature falls below 95°F (35°C)."
    ),
    # Anion Gap
    (
        "Anion gap = Sodium (Na) - (Chloride (Cl) + Bicarbonate (HCO3)). "
        "A normal anion gap is typically between 8 and 16 mEq/L. "
        "An elevated anion gap may indicate metabolic acidosis."
    ),
    # MAP
    (
        "Mean Arterial Pressure (MAP) is estimated using: MAP = (SBP + 2 × DBP) / 3. "
        "A normal MAP is approximately 70–100 mmHg; MAP < 65 mmHg can indicate poor organ perfusion."
    ),
    # BMI
    (
        "Body Mass Index (BMI) is weight (kg) divided by height (m)^2. "
        "Categories: underweight <18.5, normal 18.5–24.9, overweight 25–29.9, obesity ≥30."
    ),
    # SpO2
    (
        "Oxygen saturation (SpO2) normally 95–100%. "
        "Values below 94% may indicate hypoxemia. SpO2 < 90% often considered severe hypoxia."
    ),
    # Tissue / cell injury context for tissue queries
    (
        "Tissue hypoxia results from inadequate oxygen delivery or extraction. "
        "Mechanisms of tissue injury include hypoxia, oxidative stress, inflammation, and ionic imbalance. "
        "Lactic acidosis often accompanies poor tissue oxygenation."
    ),
    (
        "Inflammation-mediated tissue damage involves cytokine release, increased vascular permeability, "
        "and leukocyte infiltration leading to further local injury."
    )
]


#RAG using FAISS
@st.cache_resource(show_spinner=False)
def get_faiss_rag(corpus):
    text = "\n\n".join(corpus)
    # Text Splitting
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=splitter.split_text(text)
    # Embeddings
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # FAISS Vector Store
    vectoredb=FAISS.from_texts(chunks,embedding=embeddings)
    # Retriever
    retriever=vectoredb.as_retriever(search_type="similarity",search_kwargs={"k":2})
    return retriever,vectoredb

retriever,vectoredb=get_faiss_rag(CORPUS)
st.sidebar.success("RAG Setup Complete")

# Define Tools
class BMIPayload(BaseModel):
    height_m: float = Field(..., gt=0, description="Height in meters")
    weight_kg: float = Field(..., gt=0, description="Weight in kilograms")

class MAPPayload(BaseModel):
    sbp: float = Field(..., gt=0, description="Systolic blood pressure (mmHg)")
    dbp: float = Field(..., gt=0, description="Diastolic blood pressure (mmHg)")

class AnionGapPayload(BaseModel):
    na: float = Field(..., description="Sodium (mEq/L)")
    cl: float = Field(..., description="Chloride (mEq/L)")
    hco3: float = Field(..., description="Bicarbonate (mEq/L)")

class WebPayload(BaseModel):
    url: AnyHttpUrl = Field(..., description="HTTP(S) URL to fetch a short snippet from")

# ---------- Tools ----------
@tool("calc_bmi", args_schema=BMIPayload)
def calc_bmi(height_m: float, weight_kg: float) -> str:
    """
    BMI = kg / (m^2). Educational estimate; not for diagnosis.
    """
    try:
        if height_m <= 0:
            return "Height must be greater than 0."
        bmi = weight_kg / (height_m ** 2)
        return (
            f"BMI={bmi:.1f} (educational estimate). "
            "Categories: <18.5 underweight, 18.5-24.9 normal, 25-29.9 overweight, ≥30 obese."
        )
    except Exception as e:
        return f"Error computing BMI: {e}"

@tool("calc_map", args_schema=MAPPayload)
def calc_map(sbp: float, dbp: float) -> str:
    """
    Mean Arterial Pressure (MAP) ≈ (SBP + 2*DBP)/3. Educational estimate.
    """
    try:
        val = (sbp + 2 * dbp) / 3
        return f"MAP≈{val:.0f} mmHg (educational estimate). Normal ~70-100 mmHg."
    except Exception as e:
        return f"Error computing MAP: {e}"

@tool("calc_anion_gap", args_schema=AnionGapPayload)
def calc_anion_gap(na: float, cl: float, hco3: float) -> str:
    """
    Anion Gap = Na - (Cl + HCO3). Educational estimate.
    """
    try:
        ag = na - (cl + hco3)
        return f"Anion gap={ag:.1f} mEq/L (educational estimate). Normal ~8-16 mEq/L."
    except Exception as e:
        return f"Error computing anion gap: {e}"
    
@tool("web", args_schema=WebPayload)
def web(url: str) -> str:
    """
    Fetch a short web snippet (plain text best-effort). Only http(s) allowed.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        txt = resp.text
        # naive HTML strip + whitespace squish; cap preview
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt[:800]
    except Exception as e:
        return f"ERROR: {e}"

# Export for agents/graphs
TOOLS = [calc_bmi, calc_map, calc_anion_gap, web]

# LLM Setup
LOCAL_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="llama3.1"

llm=ChatOllama(
    model=OLLAMA_MODEL,
    base_url=LOCAL_BASE_URL,
    temperature=0.8,
    max_tokens=512,
    top_p=0.9,
    top_k=50
)

st.sidebar.success(f"Ollama LLM configured ({OLLAMA_MODEL})")

# LCEL ReAct helpers
ACTION_RE = re.compile(r"(?s)Action:\s*([A-Za-z0-9_\-]+)\s*\n\s*Action Input:\s*(\{.*?\})")
FINAL_RE = re.compile(r"(?s)Final Answer:\s*(.+)$")

def parse_action_or_final(text: str):
    m = FINAL_RE.search(text or "")
    if m:
        return None, None, m.group(1).strip()
    m = ACTION_RE.search(text or "")
    if not m:
        return None, None, None
    name, raw = m.group(1).strip(), m.group(2).strip()
    try:
        args = json.loads(raw)
    except Exception:
        args = None
    return name, args, None

def call_tool(name: str, args: dict | None):
    t = next((x for x in TOOLS if x.name == name), None)
    if not t:
        return f"ERROR: tool '{name}' not found."
    try:
        return t.invoke({} if args is None else args)
    except Exception as e:
        return f"ERROR: {e}"

def tool_summaries(tools):
    lines = []
    for t in tools:
        schema = getattr(t, "args", None)
        lines.append(f"- {t.name}: args schema = {schema}" if schema else f"- {t.name}")
    return "\n".join(lines)

# ReAct Prompt Template
react_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "{system}\n\nUse tools with:\nAction: <tool_name>\nAction Input: <valid JSON>\n\nFinish with:\nFinal Answer: <text>\n\nTools:\n{tool_list}"
    ),
    ("system", "RETRIEVED_CONTEXT:\n{context}"),
    ("human", "{question}"),
])

# One-step and controller (ReAct loop)
def one_step(question: str, retriever, debug: bool=False):
    # Retrieve context
    docs = retriever.invoke(question)
    

    context_text = "\n\n".join(d.page_content if hasattr(d, "page_content") else d for d in docs)
    # Build messages for LLM
    messages = react_prompt.format_messages(
        system=SYSTEM_INSTRUCTIONS,
        tool_list=tool_summaries(TOOLS),
        context=context_text,
        question=question,
    )
    # Invoke LLM
    response = llm.invoke(messages)
    ai_text = response.content if hasattr(response, "content") else str(response)
    if debug:
        st.text_area("Model raw output (one step)", ai_text, height=200, disabled=True)
    tool_name, args, final = parse_action_or_final(ai_text)
    return tool_name, args, final, ai_text, docs

def run_pipeline_react(user_prompt: str, retriever, debug: bool=False, max_steps: int=3):
    q = user_prompt
    tool_log = []
    for step in range(1, max_steps + 1):
        # Single-step ReAct iteration
        tool_name, args, final, ai_text, docs = one_step(q, retriever, debug)
        # If final produced
        if final:
            return {
                "steps": step,
                "final": final,
                "ai_text": ai_text,
                "tool_log": tool_log,
                "retrieved": docs
            }
        # If action produced
        if tool_name and args is not None:
            tool_result = call_tool(tool_name, args)
            tool_log.append({"tool": tool_name, "args": args, "result": tool_result})

            if debug:
                print(f"Tool call: {tool_name} with args: {args}")
                print(f"Tool result: {tool_result[:200]}")  # truncate long output

            # Append result for next LLM step
            q += f"\n\nTool Result ({tool_name}): {tool_result}"
            continue

        # No valid output -> ask model to be explicit
        q = q + "\n\nPlease respond with either a valid tool call (Action + Action Input JSON) or a Final Answer."
    # Reached max steps
    return {
        "final": None,
        "ai_text": ai_text if 'ai_text' in locals() else None,
        "tool_log": tool_log,
        "retrieved": docs if 'docs' in locals() else None
    }

# --- CUSTOM CSS FOR PROFESSIONAL DARK MODE (Centered, ChatGPT Style) ---

custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* 1. Root Variables & Base Setup */
:root {{
    --bg-color: #1A1A1A; /* Deep Charcoal/Black for background */
    --main-text-color: #FAFAFA; /* Lightest gray text */
    --accent-color: #00BFFF; /* Deep Sky Blue */
    --secondary-bg: #2C2C2C; /* Input/Output boxes and cards */
    --sidebar-bg: #121212; /* Even darker sidebar */
    --success-color: #38B000; /* Richer green for success */
    --warning-color: #FFC300; /* Amber for warning */
    --error-color: #FF4444; /* Red for error */
}}

/* Apply to the entire app */
.stApp {{
    background-color: var(--bg-color);
    color: var(--main-text-color);
    font-family: 'Inter', sans-serif;
}}

/* 2. Streamlit Headers & Text */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, label {{
    font-family: 'Inter', sans-serif;
    color: var(--main-text-color);
}}

.stMarkdown h1 {{
    font-weight: 800;
    font-size: 2.5rem;
    color: var(--main-text-color);
    padding-bottom: 10px;
    border-bottom: 2px solid var(--accent-color);
    text-align: center; /* Centering the main title */
}}

/* Stylized Section Headers (Agent Output, Context) */
.stMarkdown h3 {{
    font-weight: 600;
    font-size: 1.25rem;
    color: var(--accent-color);
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid var(--accent-color);
    padding-left: 10px;
}}

/* 3. Input & Output Styling */
.stTextArea > label, .stNumberInput > label {{
    color: var(--main-text-color);
    font-weight: 600;
    font-size: 1.1rem;
    text-align: center; /* Center input labels */
    width: 100%;
}}

.stTextArea > div > div > textarea, 
.stTextInput > div > div > input, 
.stNumberInput input, 
.stTextarea div[data-testid="stTextarea"] {{
    background-color: var(--secondary-bg);
    border: 1px solid #3A3A3A;
    color: var(--main-text-color);
    border-radius: 8px;
    padding: 12px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
}}

/* General Output Boxes (Raw Model, Retrieved Context) */
.output-box {{
    background-color: #242424; /* Slightly different dark shade for distinction */
    border-radius: 8px;
    padding: 15px;
    white-space: pre-wrap;
    font-size: 0.9rem;
    line-height: 1.4;
    color: #CCCCCC;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}}

/* Final Answer Box */
.final-answer-box {{
    background-color: #213A2F; /* Dark success shade */
    border-left: 5px solid var(--success-color);
    padding: 20px;
    border-radius: 10px;
    margin-top: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
}}
.final-answer-box h3 {{
    color: var(--success-color) !important;
    border: none;
    padding-left: 0;
    margin-top: 0;
}}

/* Safety Disclaimer (Centered) */
.safety-box {{
    background-color: #2C2C2C;
    border-left: 5px solid var(--warning-color);
    padding: 15px;
    border-radius: 8px;
    font-size: 0.9rem;
    color: #DDDDDD;
    margin-bottom: 25px;
    text-align: center; /* Centering the disclaimer text */
}}

/* Streamlit Warnings/Errors */
.stAlert {{
    border-radius: 8px;
    padding: 15px;
}}

/* 4. Sidebar Styling */
[data-testid="stSidebar"] {{
    background-color: var(--sidebar-bg);
    border-right: 1px solid #3A3A3A;
}}
[data-testid="stSidebar"] h2 {{
    color: var(--main-text-color);
    font-weight: 700;
    border-bottom: 1px solid #3A3A3A;
    padding-bottom: 5px;
    margin-bottom: 15px;
}}

/* Status pills class is no longer used but kept for safety */
.status-pill {{
    padding: 6px 10px;
    margin: 5px 0;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 8px;
    background-color: #1A1A1A;
    border: 1px solid #3A3A3A;
    color: var(--main-text-color);
}}
.status-pill strong {{
    color: var(--accent-color);
}}

/* 5. Buttons (Centered button text is default, just need to ensure the button container is centered) */
.stButton button {{
    background-color: var(--accent-color);
    color: var(--bg-color);
    font-weight: 700;
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    transition: background-color 0.2s, transform 0.1s;
    box-shadow: 0 4px 8px rgba(0, 191, 255, 0.4);
}}
.stButton button:hover {{
    background-color: #0099CC; 
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 191, 255, 0.6);
}}
/* Ensure button container centers the button itself */
div.stButton > button {{
    display: block;
    margin: 0 auto;
    width: fit-content;
}}

/* 6. Personal Footer (Moved to bottom center of main page) */
.personal-footer {{
    /* Increased vertical padding */
    width: 100%;
    padding: 30px 0; 
    margin-top: 60px; /* Increased top margin for more separation */
    background-color: transparent;
    border-top: 2px solid #5A5A5A; /* Thicker, more visible border */
    font-size: 1.0rem; /* Increased font size to be more readable */
    color: #AAAAAA; /* Lighter gray for better contrast */
    text-align: center;
}}
.personal-footer a {{
    color: var(--accent-color);
    text-decoration: none;
    font-weight: 700; /* Made links bolder */
    margin: 0 10px; /* Increased margin between links */
}}
.personal-footer a:hover {{
    color: var(--main-text-color);
}}
</style>
"""

# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

st.set_page_config(page_title="MedicoAgent-(Ollama + FAISS)", layout="wide", page_icon="🔬")

# --- SIDEBAR ---
with st.sidebar:
    # Status Indicators
    st.markdown("## Agent Status")
    st.markdown('<div class="status-pill">✅ RAG Setup Complete</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-pill">🧠 LLM Configured ({OLLAMA_MODEL})</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Settings
    st.markdown("## Settings")
    debug = st.checkbox("Debug: show raw model output", value=False)
    max_steps = st.number_input("Max ReAct Steps", value=4, min_value=1, max_value=8, step=1)
    st.markdown("---")
    
    # Model & Tracing
    st.markdown("## Model & Tracing")
    st.markdown(f"**Ollama Base:** `{LOCAL_BASE_URL}`")
    st.markdown(f"**Model:** `{OLLAMA_MODEL}`")

    # ADDED: Quick Tip section to the sidebar
    st.markdown("---")
    st.markdown("## 💡Tip")
    st.info("Powered by a **ReAct loop** for tool use (BMI, MAP) or **RAG** search. Enable **Debug** to view reasoning steps.")


# Use columns to create a centered block of content (4/6th of screen width)
col1, col_center, col3 = st.columns([1, 4, 1])

with col_center:
    # 1. Title and Disclaimer (Centered)
    st.markdown("<h1>🩺 MedicoAgent - ReAct RAG Agent</h1>", unsafe_allow_html=True)
    st.markdown(f'<div class="safety-box">**Safety:** {SAFETY_DISCLAIMER}</div>', unsafe_allow_html=True)

    # 2. Input
    st.markdown("### Ask your medical or clinical question below:")
    user_question = st.text_area("Enter your question (tissue query, vitals, labs, etc.):", height=150, label_visibility="visible")

    if st.button("🚀 Run Agent"):
        if not user_question.strip():
            st.error("Please type a question.")
        else:
            with st.spinner("Running ReAct pipeline..."):
                out = run_pipeline_react(user_question.strip(), retriever, debug=debug, max_steps=max_steps)

            # Display results neatly
            st.markdown("### Agent Output")
            
            # 1. Final Answer
            if out["final"]:
                st.markdown(
                    f'<div class="final-answer-box"><h3>✅ Final Answer</h3>{out["final"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.warning("No Final Answer produced within step limit.")
                
            # 2. Tool Log
            if out["tool_log"]:
                st.markdown("### Tool Log")
                for t in out["tool_log"]:
                    st.json(t)
                    
            # 3. Retrieved Context
            st.markdown("### Retrieved RAG Context (top-k)")
            with st.expander(f"View {len(out['retrieved'])} Context Chunks", expanded=True):
                for i, d in enumerate(out["retrieved"], start=1):
                    st.markdown(f"**Knowledge Section {i}:**")
                    content = d.page_content if hasattr(d, "page_content") else str(d)
                    # Use custom styling for each chunk
                    st.markdown(f'<div class="output-box" style="font-size: 0.8rem; margin-bottom: 10px;">{content[:600]}...</div>', unsafe_allow_html=True)
                    
            # 4. Raw model output if debug
            if debug and out["ai_text"]:
                st.markdown("### Raw Model Output (Final Step)")
                st.markdown(f'<div class="output-box">{out["ai_text"]}</div>', unsafe_allow_html=True)

# --- PERSONAL FOOTER (Moved to bottom center of the page) ---
st.markdown(
    """
    <div class="personal-footer">
        Built by Pavan Kumar B
        <br>
        Connect with me : 
        <a href="https://www.linkedin.com/in/pavankumarb1025/" target="_blank">LinkedIn</a> |
        <a href="https://github.com/bpavann" target="_blank">GitHub</a>
    </div>
    """, 
    unsafe_allow_html=True
)
