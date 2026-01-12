import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With OPENAI"

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are CHAT 'Z', an assistant that always replies in Gen Z lingo. "
            "Keep it casual, meme-ish, short, vibey, and friendly. "
            "Use slang like 'bet', 'say less', 'lowkey', 'no cap', 'fr', 'vibes', etc. "
            "Never break Gen Z tone but stay helpful and clear."),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,engine,temperature,max_tokens):
    #openai.api_key=api_key

    llm=ChatOpenAI(model=engine,
        api_key=api_key,        
        temperature=temperature,
        max_tokens=max_tokens
        )
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

##Title of the app
st.title("CHAT Z With OpenAI")



##Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API Key:",type="password")

##Select the OpenAI model
engine=st.sidebar.selectbox("Select Open AI model",["gpt-5.1","gpt-5.1-mini","gpt-5.1-nano","gpt-4","gpt-4-turbo"])

##Adjust Temperature and Max Token parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

##user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input and api_key:
    response=generate_response(user_input,api_key,engine,temperature,max_tokens)
    st.write(response)

elif user_input:
    st.warning("Please enter the OPEN AI API Key in the sider bar")
else:
    st.write("Please provide the user input")


