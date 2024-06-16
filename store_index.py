from src.helper import PINECONE_API_KEY,text_split,download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_pinecone import PineconeVectorStore
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import time

def load_pdf(file_path):
    all_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            all_text += page.extract_text() + "\n"
    return all_text if all_text else None

# Define the text_split function
def text_split(text):
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_text(text)

load_dotenv()

pdf_file_path = "data/finance.pdf"
extracted_data = load_pdf(pdf_file_path)
if extracted_data is None:
    raise ValueError(" Please check the load_pdf function.")

print(f"Extracted Data: {extracted_data}")

text_chunks = text_split(extracted_data)
if text_chunks is None:
    raise ValueError(" Please check the text_split function.")

print(f"Text Chunks: {text_chunks}")

embeddings = download_hugging_face_embeddings()
if embeddings is None:
    raise ValueError("The embeddings is None. Please check the download_hugging_face_embeddings function.")

print(f"Embeddings: {embeddings}")

#if not os.environ.get("PINECONE_API_KEY"):
#    from pinecone_notebooks import authenticate
#    authenticate()

api_key=os.environ.get("PINECONE_API_KEY")



pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
#host = os.environ.get('PINECONE_HOST') or "https://project-994k0af.svc.aped-4627-b74a.pinecone.io"
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "project"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=spec
    )
    
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)

text_chunk_store = {}
    
def store_text_chunk(text_chunk):
    chunk_id = f"chunk_{len(text_chunk_store)}"
    text_chunk_store[chunk_id] = text_chunk
    return chunk_id
    
for i, text_chunk in enumerate(text_chunks):
    chunk_id = store_text_chunk(text_chunk)
    embedding = embeddings.embed_query(text_chunk) 
    index.upsert(
        vectors=[
            {
                "id": f"vec_{i}", 
                "values": embedding, 
                "metadata": {"chunk_id": chunk_id} 
            }
        ],
        namespace="ns1"
    )

print("Indexing completed successfully.")
