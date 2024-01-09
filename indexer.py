import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

import dotenv

dotenv.load_dotenv()

import sqlite_fix


loader = PyPDFLoader("./llm-ebook-part1.pdf")
docs = loader.load()

print(len(docs), docs[0])

# splitting documents to fix into the context window
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./book_db"
)
