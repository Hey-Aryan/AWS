from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import tempfile
import os

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
    loader = UnstructuredFileLoader(temp_pdf_path)
    data = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)

    # Clean up temporary file
    os.unlink(temp_pdf_path)

    return texts 




#########################################################
#
#   A function to connect to the API and Access the link ---|
#                                                           |
#########################################################   |
#                                                           |
#                                                           |
#                                                           |
def main():#                                               \ /           
    # URL of the PDF file
    url = "https://s3.us-east-1.amazonaws.com/lambda.bucket1/Eshan_Resume.docx?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDkaCXVzLWVhc3QtMSJHMEUCIQDR%2BrsAlld1PWEGmzyrXR8REsuvKT490ywJbCOlEA57nAIgcvXBL96XXjaItKVGoCxJPfr182JX3BZ0oR1cfmtgrJIq0QMIcRAAGgw4NTE3MjUzNDcyOTMiDB%2FSWmNqMAbbq%2BeddiquAyTF34F4rJuv0Iz17DW%2BvqMhjElKi5gpvk78GMKHZO1JLI64eUDIjK9jPIxNeOT%2BFvZOMtqSoY23w8CNgV9mvea%2FjEfreHTMbgFOGnH89yhXvo2tv3n7%2FyjP0Xq2DvkxDPGXcbHVALsIv245reDvJ9Cp6971n4lT2Ynl3KxVEffPWwHLgiZBzULvvK1gBs5ZO1UjuyTzB9v8o5R%2B%2FB96YQevqLSvzFGJ4KaydHjk29Iet3iKozqN%2BSd7xgwBwjJaqeMXGYZo5LFTYt6cfu6HGaP37Qaz4G6JZQOzmT%2BFxpzthvsEgx%2Bl0ili7Pzm5iJ6kZqQ0M0QHCFMItSw9lIFR9IwTxDrvDdUN8BNuTlxioYVWCIfVmmp1Zord3arhAEkoSDBc%2F2xyHRimaAFOB6JMnfeybTsVZUyNad%2B%2FLsYGIkdA21dkxD3cGPzqnrBPBSA%2BW8Lul29Hdg2c3QcQ6kuzlUVNjmj9d1uVj%2Bn9td6pWkAHbNtLDgqibANB2MvVYvbr7aVDeSbSSpSsguIrYWzdZx4yYJECL7Wr9Hv0T33k6rlHK5vX09pojHZk4JQfEswyfvosAY6lAJvZyNb0pzt6f3ZD%2F82PtKSgSsMO6%2Fzykt3pVaL%2Fx4WKvUHz7Jj7CKutjTLcmwuEQHo%2BE%2BWRUNQt%2FUqFyyj9HF8gB%2B07YetglLs84W9uRfWbV65OIGRFaR%2BXEF%2Fx%2FbPSVJefCD5XtQMD9VMSRJR%2BwcKfG6bXpFuE3tx92%2B536eV962ZcNgqkTdvjI8rIvHztbMX4NrgxSF15lfsGgJukxhGWBe9wvildVRIbu87QQXzi2%2BM5WmaygV5qOE%2Fb6g2bHTp1xxHloQyRfYPU3hmsqXa8hT2KthYuMoLQ7ZRSxMDHzlf%2F2%2FOiw28UugUb1%2FgYo2jn7gOjIcXFvpEKtM7anGccntJl38U0xSfoXBiILPgBYsfxSI%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240413T082812Z&X-Amz-SignedHeaders=host&X-Amz-Expires=7200&X-Amz-Credential=ASIA4MTWJNHO42ARFJWX%2F20240413%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=68c2e484267f5942a41e7b8e0e0ea7089b4162e89e34f40a0dd81887b0ec2800"

    # Step 1: Create Texts (Chunks)
    texts = Create_Texts(url)

    # Display the first chunk
    if texts:
        print(texts[0])
    else:
        print("No text chunks created.")

if __name__ == "__main__":
    main()
