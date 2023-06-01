import os
import logging
# from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.staticfiles import StaticFiles
# from pymongo import MongoClient
# from pydantic import BaseModel
# from typing import Annotated
# from utils import create_access_token
# from google.oauth2 import id_token
# from google.oauth2.credentials import Credentials
# from google.auth.transport import requests

from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import OpenAI
from langchain.chains.question_answering import load_qa_chain
load_dotenv()

# class User(BaseModel):
#     email: str
#     full_name: str
#     role: str

# client = MongoClient(os.getenv("MONGO_URI"))
# db = client.get_database('storybot')
# users = db['users']

app = FastAPI()

app.mount("/store", StaticFiles(directory="store"), name="static")

origins = ["https://storybot-three.vercel.app"]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
# app.add_middleware(HTTPSRedirectMiddleware)

# Create a logger object
logger = logging.getLogger(__name__)

# Set log level
logger.setLevel(logging.DEBUG)

# Define a handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Define a formatter to format log messages
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)


"""
@app.post("/api/auth/login/google")
async def login(token: Annotated[str, Body(embed=True)]):
    try:
        # Verify the access token with Google
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), os.getenv("GOOGLE_CLIENT_ID"))

        for info in idinfo:
            print(info, idinfo[info])
        print("token" + token)

        user = users.find_one({"email": idinfo["email"]})
        if not user:
            present = datetime.utcnow()
            users.insert_one({
                "email": idinfo["email"],
                "full_name": idinfo["name"],
                "is_verified": True,
                "role": "user",
                "created_at": present,
                "updated_at": present
            })
        user = users.find_one({"email": idinfo["email"]})
        access_token = create_access_token(data={"sub": user["email"]})
        return {"access_token": access_token, "user": User(**user)}
    except ValueError:
        #  Handle invalid tokens
        return {"error": "Invalid token"}
"""

# template = """You are an interesting storyteller interacting with a human.

# Given the following extracted parts of a story and an human's input, generate a next part of story.
# First of all, we need to gather details
template = """
{DOC_PROMPT}
{context}
{human_input}
"""

prompt = PromptTemplate(
    input_variables=["human_input", "context", "DOC_PROMPT"],
    template=template
)


@app.post('/{user}')
async def next(user: str, msg: str = Body(embed=True)):
    logger.info(user + '/' + msg)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len)
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(temperature=0.7)
    chain = load_qa_chain(llm=llm, chain_type="stuff",
                          verbose=True, prompt=prompt)
    if os.path.exists(f"./store/{user}/index.faiss"):
        docsearch = FAISS.load_local(f"./store/{user}", embeddings)
        # msg = msg + " What's next?"
        docs = docsearch.similarity_search(msg)
        # msg = "Human: " + msg
        contents = ""
        with open(f"./store/{user}/prompt.txt", 'r') as f:
            contents = f.read()
            f.flush()
        completion = chain.run(
            input_documents=docs, human_input=msg + "What's next?", DOC_PROMPT=contents)
        texts = text_splitter.split_text(msg + "\r\n" + completion + "\r\n")
        docsearch.add_texts(texts)
    else:
        completion = chain.run(
            input_documents=[], human_input="Now ask me one by one but don't say yes or sure at first. Just start with greetings.", DOC_PROMPT=msg)
        texts = text_splitter.split_text(
            "Now ask me one by one but don't say yes or sure at first. Just start with greetings.\r\n" + completion + "\r\n")
        docsearch = FAISS.from_texts(texts, embeddings)
        os.makedirs(f"./store/{user}")
        with open(f"./store/{user}/prompt.txt", 'w') as f:
            f.write(msg)
            f.flush()

    docsearch.save_local(f"./store/{user}")
    return {'msg': completion}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=9000)
