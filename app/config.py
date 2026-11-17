import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

PDF_PATH = DATA_DIR / "ModaLoop_Regulamin_i_Polityka_Logistyczna.pdf"
ORDERS_CSV = DATA_DIR / "customers_orders.csv"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gemini-3.5-flash-lite"

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DB_READER_USER = os.getenv("DB_READER_USER", "support_reader")
DB_READER_PASSWORD = os.getenv("DB_READER_PASSWORD")