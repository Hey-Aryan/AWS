# Import necessary libraries

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from pinecone import Pinecone
from openai import OpenAI
import tempfile
import requests
import json
import time
import os

# Function to get Pinecone index : Currently Using my Personal Credentials
def get_pinecone_index():
    pc = Pinecone(api_key="54fd135d-6e81-43c5-8802-ddfb63c09947")
    index = pc.Index("avivo-vector-db")
    return index

# Function to retrieve insights from a PDF file
def GetInsights(file_id, file_name, file_type, user_id, url):
    try:


        # Function to create texts/chunks from a PDF
        def Create_Texts(url):
            # Load PDF file from the provided URL
            response = requests.get(url)
            if response.status_code == 200:
                pdf_content = response.content
            else:
                print("Failed to fetch PDF content.")
                return []

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
            print("There are total",len(texts),"chuncks in the file")

            return texts 
        



        # Function to create vector representations and upsert to Pinecone index

        def createVector(item, texts, file_id, file_name, file_type, user_id):

            OPENAI_API_KEY = os.getenv('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', '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')
            index = get_pinecone_index()

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            text = texts[item].page_content[:1500]
            response = embeddings.embed_query(text)
            text = texts[item]


            metadata = {
            'char_length': len(text.page_content),
            'page_no': str(text.metadata["page"]),
            'text': text.page_content[:1500],
            'source': text.metadata["source"],
            'file_id': file_id,
            'file_name': file_name,
            'file_type': file_type,
            'user_id': user_id
            }

            vector = []
            vector.append({"id": str("User1_")+str(item), "values": response[:1536], "metadata": metadata})
            index.upsert(vector,namespace="AvivoDB1")

        # Step 1: Create Texts (Chunks)
        texts = Create_Texts(url)

        # Step 2: To Upsert Vectors
        index = get_pinecone_index()

        for item in range(len(texts)):
            print(item)
            createVector(item, texts, file_id, file_name, file_type, user_id)
            time.sleep(20)

        # Prepare response data for success case
        response_data = [
            {"clause": "Parties Involved", "value": "Party1 Name, Party2 Name"},
            {"clause": "Effective Date", "value": "March 21 2024"},
            {"clause": "Purpose/Objective", "value": "A clear statement outlining the purpose or objective of the agreement is often included."},
            {"clause": "Duration and Termination", "value": "On Going"},
            {"clause": "Dispute Resolution", "value": "The contract may specify mechanisms for resolving disputes that may arise during the term of the agreement."}
        ]
        # Create JSON response for success case
        response = {
            "success": True,
            "error": "",
            "data": response_data
        }

    except Exception as e:
        # Create JSON response for failure case
        response = {
            "success": False,
            "error": "Invalid S3 URL",
            "data": []
        }

    return json.dumps(response)




# Function 2 to get answer to a question
def GetAnswer(user_id, question, file_ids='0'):
    try:

        # Function to retrieve relevant contexts
        def retrieve(query):
            index = get_pinecone_index()
            
            limit = 3750
            OPENAI_API_KEY='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'

            embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            res = embed.embed_query(query)

            # retrieve from Pinecone
            xq = res[:1536]

            # get relevant contexts
            contexts = []
            time_waited = 0
            while (len(contexts) < 3 and time_waited < 60 * 12):
                res = index.query(vector=xq, 
                                top_k=3, 
                                filter={
                                        "user_id": user_id
                                    },
                                include_metadata=True)
                contexts = contexts + [
                    x['metadata']['text'] for x in res['matches']
                ]
                print(f"Retrieved {len(contexts)} contexts, sleeping for 15 seconds...")
                time.sleep(15)
                time_waited += 15

            if time_waited >= 60 * 12:
                print("Timed out waiting for contexts to be retrieved.")
                contexts = ["No contexts retrieved. Try to answer the question yourself!"]


            # build our prompt with the retrieved contexts included
            prompt_start = (
                "Answer the question based on the context below.\n\n"+
                "Context:\n"
            )
            prompt_end = (
                f"\n\nQuestion: {query}\nAnswer:"
            )
            # append contexts until hitting limit
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
            return prompt

        # Function to complete the prompt and get the answer
        def complete(prompt):
            OPENAI_API_KEY='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'
            client = OpenAI(api_key=OPENAI_API_KEY)

            # instructions
            sys_prompt = "You are a helpful assistant that always answers questions."
            # query text-davinci-003
            res = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return res.choices[0].message.content
        
        query = question
        query_with_contexts = retrieve(query)
        answer = complete(query_with_contexts)

        # Create JSON response for success case
        response = {
            "success": True,
            "error": "",
            "data": [
                {
                    "page_no": "3",
                    "paragraph_no":"5",
                    "score":"6",
                    "answer": answer
                }
            ]
        }

    except Exception as e:
        # Create JSON response for failure case
        response = {
            "success": False,
            "error": "failed to get answer from LLM Model",
            "data": []
        }

    return json.dumps(response)




# Main function
def main():
    user_id = "User1"
    file_id = "84efeced-51a4-4511-9481-a5d8f4569691"
    file_name = "GP1_2023.pdf"
    file_type = "pdf"
    url = "https://contract.s3.amazonaws.com/sample-contract/Test_Contracts/GP1_2023.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MTWJNHORTZFAMWL%2F20240327%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240327T163305Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=452dd00123fb8810a60ba08acdfbffc6243ad9f0e43032a94f73701f43b717da"
    question = "What is formation of contract"

    #Call GetInsights function
    Output = GetInsights(file_id, file_name, file_type, user_id, url)
    print(Output)

    # Call GetAnswer function
    Output2 = GetAnswer(user_id, question, file_ids='0')
    print(Output2)

if __name__ == "__main__":
    main()
