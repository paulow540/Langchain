from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama


import streamlit as st
import os
from dotenv import load_dotenv



# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("OLLAMA2 CHATBOT", anchor=False)
input_text = st.chat_input("Ask me any question")

# LangChain Ollama model
llm = ChatOllama(model="llama3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Process user input
if input_text:
    st.write(chain.invoke({"question": input_text}))
