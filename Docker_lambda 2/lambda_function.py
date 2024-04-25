import json
import os
import tempfile
import requests
import time
import random
import logging
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
from openai import OpenAI

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def Create_Texts(url, file_name, file_type):
    if file_type == "pdf":
        # Load PDF file from the provided URL
        response = requests.get(url)
        pdf_content = response.content

        # Save PDF content to a temporary file in the /tmp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir='/tmp/') as temp_pdf:
            logger.info(f"Directory path: {dir}")
            temp_pdf.write(pdf_content)
            temp_pdf_path = temp_pdf.name

        # Use PyPDFLoader to load the PDF content from the file path
        loader = PyPDFLoader(temp_pdf_path)
        data = loader.load()

    else:
        presigned_url = url  # Replace with the actual way you get the pre-signed URL

        # Download document into /tmp folder
        File_Name = file_name + "." + file_type

        tmp_file_path = f"/tmp/{File_Name}"
        with open(tmp_file_path, 'wb') as tmp_file:
            response = requests.get(presigned_url)
            if response.status_code == 200:
                tmp_file.write(response.content)
            else:
                logger.error("Failed to download document from the pre-signed URL.")
                return []
        
        # Use UnstructuredFileLoader to load the document content from the file path
        loader = UnstructuredFileLoader(tmp_file_path)
        data = loader.load()

        # Remove the temporary file
        os.remove(tmp_file_path)                    
        

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(data)

    # Clean up temporary file
    if file_type == "pdf":
        os.unlink(temp_pdf_path)
    
    logger.info(f"There are total {len(texts)} chunks in the file")

    print(texts)

def main():
    url = ""
    file_name = ""
    file_type = ""

    Create_Texts(url, file_name, file_type):


    


if __name__ == "__main__":
    main()