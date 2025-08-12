# from fastapi import FastAPI
# from langchain.prompts import ChatPromptTemplate
# # from langchain.chat_models import ChatOpenAI
# # from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI

# # from langchain_community.llms import ollama 
# # from langchain_community.chat_models import ChatOllama
# from langchain_ollama import ChatOllama


# from langserve import add_routes
# import uvicorn
# import os
# # from dotenv import load_dotenv 


# # load_dotenv()

# # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# app = FastAPI(
#     title="Langchain Server",
#     version="1.0",
#     description="A simple API Server"
# )

# # add_routes(
# #     app,
# #     ChatOpenAI(),
# #     path="/openai"
# # )

# # model = ChatOpenAI()

# # Ollama lllma

# llm= ChatOllama(model="llama2")



# prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
# prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words")

# # add_routes(
# #     app,
# #     prompt1|model,
# #     path="/essay"
# # )

# add_routes(
#     app,
#     prompt2|llm,
#     path="/poem",
# )



# if __name__ == "__main__":
#     uvicorn.run(app,host="127.0.0.1",port=8000)


from fastapi import FastAPI
from langserve import add_routes
from fastapi.responses import RedirectResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

import config
import os
from dotenv import load_dotenv
import uvicorn


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API Server",
)


# output = StrOutputParser()

prompts1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompts2 = ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words")


model = ChatOpenAI()
llm = ChatOllama(model="llama2")



@app.get("essay")
async def redirect_root_to_docs1():
    return RedirectResponse(add_routes(
        app,
        prompts1 | model,
        path="/essay",
    ))
    



@app.get("poem")
async def redirect_root_to_docs2():
    return RedirectResponse(add_routes(
    app,
    prompts2 | llm ,
    path="/poem",
    
))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)