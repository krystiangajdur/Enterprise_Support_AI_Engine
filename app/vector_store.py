import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import CHROMA_DIR, PDF_PATH, EMBEDDING_MODEL

def ingest_policy_pdf():
    pdf_path = str(PDF_PATH)
    persist_directory = str(CHROMA_DIR)

    if not os.path.exists(pdf_path):
        print(f'Error: File {pdf_path} not found in the project directory.')
        return

    print('Loading the PDF file...')
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f'Wczytano stron: {len(documents)}.')

    print('Splitting the document into chunks...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    chunks = text_splitter.split_documents(documents)
    print(f'The document has been divided into {len(chunks)} fragments.')

    print('Initializing the local embedding model...')
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

    print('Creating and saving a ChromaDB vector database...')
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_directory
    )

    print(f'Success! Vector base successfuly launched to: {persist_directory}')

if __name__ == '__main__':
    ingest_policy_pdf()
    