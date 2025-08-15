import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS vectorstore (trusted source)
db = FAISS.load_local(
    "product_faq_vectorstore",
    OllamaEmbeddings(),
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever()

# Initialize Llama2 via Ollama
llm = ChatOllama(model="llama2", allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("SmartHomeHub Support Chatbot")
question = st.text_input("Ask your question about SmartHomeHub:")

if question:
    response = qa_chain.run(question)
    st.markdown("**Answer:**")
    st.write(response)
