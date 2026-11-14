import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_policy_pdf():
    pdf_path = "ModaLoop_Regulamin_i_Polityka_Logistyczna.pdf"
    persist_directory = "./chroma_db"

    if not os.path.exists(pdf_path):
        print(f'Błąd: Nie znaleziono pliku {pdf_path} w katalogu projektu.')
        return

    print('Wczytywanie pliku pdf...')
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f'Wczytano stron: {len(documents)}.')
    print(f'Zeby zobaczyc jak to wyglada: {documents}')

    print('Dzielenie dokumentu na chunki...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    chunks = text_splitter.split_documents(documents)
    print(f'Podzielono dokument na {len(chunks)} fragmentow.')

    print('Inicjalizacja lokalnego modelu embeddingow...')
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    print('Tworzenie i zapis bazy wektorowej ChromaDB...')
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_directory
    )

    print(f'Sukces! Baza wektorowa została pomyślnie zapisana w katalogu: {persist_directory}')

if __name__ == '__main__':
    ingest_policy_pdf()
    