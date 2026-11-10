import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print("Connecting to PostgreSQL...")
engine = create_engine(connection_string)

print("Reading customers_orders.csv...")
df = pd.read_csv("customers_orders.csv")

print("Transforming data...")
df['customer_name'] = df['customer_name'].str.strip()

print(f'Loaded {len(df)} rows from CSV. Preview:')
print(df.head())

table_name = "orders"
print(f'Loading data into PostgreSQL table {table_name}...')

df.to_sql(
    name = table_name,
    con = engine,
    if_exists = "replace",
    index = False
)

print("ETL proccess completed succesfully! Data is now in PostgreSQL.")


