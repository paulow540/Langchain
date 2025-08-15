import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
import os
from dotenv import load_dotenv


load_dotenv()
# print("OPENAI KEY LOADED:", os.getenv("OPENAI_API_KEY") is not None)


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Load data
df = pd.read_csv("product_faq.csv")

# Convert to LangChain Documents
docs = [Document(page_content=row['answer'], metadata={"question": row['question']}) for _, row in df.iterrows()]

# Split and Embed
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
db.save_local("product_faq_vectorstore")
