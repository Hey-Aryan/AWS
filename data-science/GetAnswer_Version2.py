import json
import os
import tempfile
import time
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone
from openai import OpenAI

def lambda_handler(event, context):
    action = event.get('action')
    if action == 'GetAnswer':
        user_id = event.get('user_id')
        question = event.get('question')
        file_id = event.get('file_id')
        return GetAnswer(user_id, question, file_id)
    else:
        return {
            "statusCode": 400,
            "body": json.dumps("Invalid action specified!")
        }

# Function to get Pinecone index : Currently Using my Personal Credentials
def get_pinecone_index():
    pc = Pinecone(api_key="54fd135d-6e81-43c5-8802-ddfb63c09947")
    index = pc.Index("avivo-vector-db")
    return index

# Function to retrieve relevant contexts
def retrieve(query, user_id, file_ids):
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
            filter={"user_id": user_id, "file_id": file_id},
            include_metadata=True
        )
        for x in res['matches']:
            contexts.append(x['metadata']['text'])
            page_no = x['metadata']['page_no']
            score = x['score']
            page_nos.append(page_no)
            scores.append(score)
    
        print(f"Retrieved {len(contexts)} contexts")
        time_waited += 0

    if time_waited >= 60 * 12:
        print("Timed out waiting for contexts to be retrieved.")
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
    # Query text-davinci-003
    res = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content


# Function to get answer to a question
def GetAnswer(user_id, question, file_ids):
    try:
        query = str(question)
        query_with_contexts, page_no, score = retrieve(query, user_id, file_ids)
        answer = complete(query_with_contexts)

        # Create JSON response for success case
        response = {
            "success": True,
            "error": "",
            "data": [
                {
                    "page_no": page_no,
                    "paragraph_no": "5",
                    "score": score,
                    "answer": answer
                }
            ]
        }
    
    except Exception as e:
        # Create JSON response for failure case
        response = {
            "success": False,
            "error": f"Failed to get answer: {str(e)}",
            "data": []
        }
    
    return json.dumps(response)


'''
{
  "action": "GetAnswer",
  "user_id": "User1",
  "question": "What is Termination Clause",
  "file_ids": ["84efeced-51a4-4511-9481-a5d8f4569691"]
}
'''