from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain.llms import CTransformers
from unittest import loader

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
#etract data fromthe files
def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob=("*.pdf"),
                             loader_cls=PyPDFLoader)

                             
    documents = loader.load()
    
    return documents

#creating text chunks and dividing the text into small chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


#download the embedding model from huggingface
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings