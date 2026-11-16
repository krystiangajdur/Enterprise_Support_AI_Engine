import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, ORDERS_CSV

username = DB_USER
password = DB_PASSWORD
host = DB_HOST
port = int(DB_PORT)
database = DB_NAME

connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

print("Connecting to PostgreSQL...")
engine = create_engine(connection_string)

print("Reading customers_orders.csv...")
df = pd.read_csv(ORDERS_CSV)

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


