import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt Template for the  GEN Z LINGO MODE
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are CHAT 'Z', an assistant that always replies in Gen Z lingo. "
            "Keep it casual, meme-ish, short, vibey, and friendly. "
            "Use slang like 'bet', 'say less', 'lowkey', 'no cap', 'fr', 'vibes', etc. "
            "Never break Gen Z tone but stay helpful and clear."
        ),
        ("user", "Question: {question}")
    ]
)

# =======================
# Header — CHAT "Z"
# =======================

st.markdown(
    """
    <style>
    .title {
      text-align: center;
      font-size: 3rem;
      font-weight: 900;
      margin: 0;
      background: linear-gradient(90deg, #ff6ec7, #7a5cff, #4ee1a1, #ffd166);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 4px 10px rgba(120,80,200,0.18));
    }
    .tagline {
      text-align: center;
      color: #9aa0a6;
      margin-top: 4px;
      margin-bottom: 22px;
      font-size: 1.05rem;
    }

    

    
    .input-col .stTextInput label {
        display: none !important;
    }


    .input-col .stTextInput > div {
        display: flex !important;
        align-items: center !important;   
        height: 42px !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    
    .input-col .stTextInput input {
        height: 42px !important;
        padding: 8px 10px !important;
        font-size: 1rem;
        border-radius: 10px;
        box-sizing: border-box;
    }

    
    .btn-col .send-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 42px;
    }

    .btn-col .send-btn button {
        margin: 0;
        padding: 6px 14px;
        font-size: 1.3rem;
        font-weight: 700;
        border-radius: 10px;
        cursor: pointer;
    }

    .footer {
        position: fixed;
        right: 10px;
        bottom: 10px;
        font-size: 10px;
        opacity: 0.75;
    }
    </style>

    <h1 class="title">CHAT "Z"</h1>
    <div class="tagline">for the Gen Z ✨</div>
    """,
    unsafe_allow_html=True,
)

# =======================
# Input + Button Row
# =======================

left, right = st.columns([12, 2])

with left:
    st.markdown('<div class="input-col">', unsafe_allow_html=True)
    input_text = st.text_input("", placeholder="Drop your question here, no cap...", key="q")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="btn-col">', unsafe_allow_html=True)
    st.markdown('<div class="send-btn">', unsafe_allow_html=True)
    submitted = st.button("➜", key="send")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# Model + Chain
# =======================

llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if submitted and input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)

# Bottom corner model tag
st.markdown('<div class="footer">gemma 2b</div>', unsafe_allow_html=True)
