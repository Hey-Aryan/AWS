from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Get the Signed URL Part from S3 and catch it here

def Create_Texts():
    files = ("https://contract.s3.amazonaws.com/sample-contract/Test_Contracts/GP1_2023.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MTWJNHORTZFAMWL%2F20240327%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240327T163305Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=452dd00123fb8810a60ba08acdfbffc6243ad9f0e43032a94f73701f43b717da")
    #loader = TextLoader(file_path="../data/PaulGrahamEssays/vb.txt")

    ## Other options for loaders
    #loader = PyPDFLoader(files)
    loader = UnstructuredPDFLoader(files)
    #loader = OnlinePDFLoader(files)

    data = loader.load()

    # Note: If you're using PyPDFLoader then it will split by page for you already
    print (f'You have {len(data)} document(s) in your data')
    print (f'There are {len(data[0].page_content)} characters in your sample document')
    print (f'Here is a sample: {data[0].page_content[:200]}')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)

    text = texts[0]

    text = texts[0]

    metadata = {
        'char_length': len(text.page_content),
        'text': text.page_content[:1000],
        'page_no': str(text.metadata["page"]),
        'source': text.metadata["source"]
    }
    print(metadata)

    print (f'Now you have {len(texts)} documents')

    return texts
     
    #Response POST Method in SQL database 








def main():
    # Step 1 Create Texts i.e. Chuncks 
    Texts = Create_Texts()
    print(Texts[0])

if __name__ == "__main__":
    main()
