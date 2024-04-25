import json
import os
import tempfile
import time
import requests
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone
from openai import OpenAI

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

limit = 3750

def lambda_handler(event, context):
    action = event.get('action')
    if action == 'GetInsights':
        file_id = event.get('file_id')
        file_name = event.get('file_name')
        file_type = event.get('file_type')
        user_id = event.get('user_id')
        url = event.get('url')
        return GetInsights(file_id, file_name, file_type, user_id, url)
    else:
        return {
            "statusCode": 400,
            "body": json.dumps(response)
        }

# Function to get Pinecone index : Currently Using my Personal Credentials
def get_pinecone_index():
    pc = Pinecone(api_key="54fd135d-6e81-43c5-8802-ddfb63c09947")
    index = pc.Index("avivo-vector-db")
    return index

# Function to retrieve relevant contexts


def retrieve(query, user_id, file_id):
    index = get_pinecone_index()
    
    limit = 3750
    OPENAI_API_KEY = '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'

    embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    res = embed.embed_query(query)
    
    # Retrieve from Pinecone
    xq = res[:1536]

    # Get relevant contexts
    contexts = []
    page_nos = []
    scores = []

    time_waited = 0
    while len(contexts) < 3 and time_waited < 60 * 12:
        res = index.query(
            vector=xq,
            top_k=3,
            include_metadata=True
        )
        for x in res['matches']:
            contexts.append(x['metadata']['text'])
            page_no = x['metadata']['page_no']
            score = x['score']
            page_nos.append(page_no)
            scores.append(score)
    
        logger.info(f"Retrieved {len(contexts)} contexts")
        time_waited += 0

    if time_waited >= 60 * 12:
        logger.info("Timed out waiting for contexts to be retrieved.")
        contexts = ["No contexts retrieved. Try to answer the question yourself!"]

    # Build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # Append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt, page_nos[0], scores[0]

# Function to complete the prompt and get the answer
def complete(prompt):
    OPENAI_API_KEY = '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'
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Instructions
    sys_prompt = "You are a helpful assistant that always answers questions."

    res = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content



# Function to retrieve insights from a PDF file
def GetInsights(file_id, file_name, file_type, user_id, url):
    try:
        # Function to create texts/chunks from a PDF
        def Create_Texts(url):
            # Load PDF file from the provided URL
            response = requests.get(url)
            pdf_content = response.content

            # Save PDF content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_content)
                temp_pdf_path = temp_pdf.name

            # Use PyPDFLoader to load the PDF content from the file path
            loader = PyPDFLoader(temp_pdf_path)
            data = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            texts = text_splitter.split_documents(data)

            # Clean up temporary file
            os.unlink(temp_pdf_path)

            logger.info("There are total %d chunks in the file", len(texts))

            return texts

        # Function to create vector representations and upsert to Pinecone index
        def createVector(item, chunk, file_id, file_name, file_type, user_id):
            OPENAI_API_KEY = os.getenv('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', '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')
            index = get_pinecone_index()

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            text = chunk.page_content[:1500]
            response = embeddings.embed_query(text)
            text = chunk

            metadata = {
                'user_id': user_id,
                'file_id': file_id,
                'page_no': (text.metadata["page"] + 1),
                'text': text.page_content[:1500],
                'source': text.metadata["source"],
                'file_name': file_name,
                'file_type': file_type,
                'char_length': len(text.page_content)
            }
            vector = []
            vector.append({"id": str("User1_") + str(item), "values": response[:1536], "metadata": metadata})
            index.upsert(vector, namespace="Integrated_DB")

        # Step 1: Create Texts (Chunks)
        texts = Create_Texts(url)

        # Step 2: To Upsert Vectors
        for item in range(len(texts)):
            text = texts[item]
            createVector(item, text, file_id, file_name, file_type, user_id)
            logger.info(item)
            time.sleep(20)

        query = ("What is the duration of the contract, and are there any termination clauses?")
        query_with_contexts, page_no, score = retrieve(query, user_id, file_id)
        Answer1 = complete(query_with_contexts)

        query = ("What are the parties involved, and what are their roles and responsibilities?")
        query_with_contexts, page_no, score = retrieve(query, user_id, file_id)
        Answer2 = complete(query_with_contexts)

        query = ("What are the terms of payment and any associated payment schedules?")
        query_with_contexts, page_no, score = retrieve(query, user_id, file_id)
        Answer3 = complete(query_with_contexts)

        # Prepare response data for success case
        response_data = [
            {"clause": "What is the duration of the contract, and are there any termination clauses?", "value1": Answer1},
            {"clause": "What are the parties involved, and what are their roles and responsibilities?", "value2": Answer2},
            {"clause": "What are the terms of payment and any associated payment schedules?", "value3": Answer3}
        ]

        # Create JSON response for success case
        response = {
            "success": True,
            "error": False,
            "data": response_data
        }

    except Exception as e:
        # Create JSON response for failure case
        response = {
            "success": False,
            "error": str(e),
            "data": []
        }

    return json.dumps(response)

def main():
    event = {
            "action": "GetInsights",
            "file_id": "84efeced-51a4-4511-9481-a5d8f4569hello",
            "file_name": "hardward college observatory.pdf",
            "file_type": "pdf",
            "user_id": "User1",
            "url": "https://s3.us-east-1.amazonaws.com/lambda.bucket1/GP9_2024.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHgaCXVzLWVhc3QtMSJHMEUCIQC6YZXCszlMZlnrPb%2B32o9ShhhjLJD7dYY2LkpZdbkG9gIgQack%2FIkZp7h%2Bmu5DGZhNqeYtW1mQOCWBeGqDDln5VBIq2gMIof%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw4NTE3MjUzNDcyOTMiDLvlFWpBetzDgCSQnyquA8%2BYjJmX%2FdTt8%2F7Ek8RtZqvIjvua4mjeKyfCCJyRkacO45p%2Fdr2dYVgbV8IqRij%2BWICSV3xZPwtpGi2m4WSx4KehnoNzK0FgyDGEGfo9KQcoA5VGHV65S%2BYbyBCDJLfuunADMCgrnhxlkCR26k3AqsRFdgLMQw929ocxHNwuwuSbJ4rtekgUHeZ5xO3R4tAM5iWcYl7kzqCy5edSZxCC7vzr8uTMd1fjSoo3xsIzvZjuTmsfUFocc%2ByPg2v1kH7VgFMSxW%2BgxxBkkSBsrOngiKVg1CEXa%2BtK2Br%2FIPHnZUWe5bg0MmY%2Be1yx4fz6QjL8lkcpCbf7EOyiQT7erypky%2Fr8kb2pau7VsPDVsakXuUGRl%2FWCge%2BBxCDcF7fB%2FpUnQUkgujLgE0XkqU2C2PVFvfE9b%2FuiLmhhstluML83O%2Be660bIE%2B1YoO55bBJaFETfl4Kj9IvVRZEJduJ4B8yfdUK1%2BEBQdsBOWhMj41wtKZr35mL7607BE7801Ivq4cRr%2FWF1p0kNaR%2BoQQZav%2BA9PBGWFY5KVogyi%2BrtAH19FVgakVpzWobO2F82eNWQQ5Qwnsa%2BsAY6lAJjWWFKPxqEu6cBAz%2BFcZMAqEIIIipkQDhOSW0Cv8Ui2IebSt3xTEc8N02CRQMKFjZUKmzzm3KLz02pVAY1TPC9sEKo3yMMZAvnjJjiERHN9NktMPt18kGtsHv8m64eo%2BCBavLqoB1C%2F1g%2Fur2EO35qJ6UqoIHESVKw0dnN2bmJH3uPyT5kgOKFFjpo7DhGEew%2FM6gvOWy3QI7GiX2ZwsSmF3LtdpbDfRNbrZ6KfbkWvDA%2FxxokDlyMPYArundzXZqYhJBP3k4XjMlURar65pw0wvh%2F69HvxPz%2FS39SN0ciJeUx8ll%2FZKO2taN99AbtqHQ6Dz%2FgI%2B10HUXVUR4iOVIERxpVlXvynQHvNFL5qCjRIwvpx40%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240405T074518Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIA4MTWJNHOUYVYY666%2F20240405%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e1ea400b5befc336c4121497b4c4d991a71bd6f70da01e6cfe2a3ce4b9ea3350"        
            }
    context = 1

    output = lambda_handler(event, context)
    print(output)

if __name__ == "__main__":
    main()
