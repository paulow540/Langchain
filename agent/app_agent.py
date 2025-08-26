import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Wikipedia AI Agent", page_icon="📖")
st.title(" Wikipedia AI Agent")
st.write("Ask me any question! I’ll fetch knowledge from **Wikipedia** and summarize with AI.")

# Setup LLM & Wikipedia tool
llm = ChatOllama(model="llama2", allow_dangerous_deserialization=True)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Load a Wikipedia page into FAISS (demo)
@st.cache_resource
def build_vectorstore(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

vectorstore = build_vectorstore("https://en.wikipedia.org/wiki/Artificial_intelligence")
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Build Agent with Wikipedia tool
tools = [wiki_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# User Input
question = st.text_input(" Ask a question:", placeholder="e.g., Who is the father of AI?")

if question:
    st.write(" Searching Wikipedia...")
    
    # Run both retrieval and agent
    with st.spinner("Thinking..."):
        agent_answer = agent.run(question)
        retrieval_answer = qa_chain.run(question)

    st.subheader(" AI Agent Answer")
    st.write(agent_answer)

    st.subheader(" Retrieved Wikipedia Context Answer")
    st.write(retrieval_answer)
