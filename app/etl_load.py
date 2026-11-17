import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, ORDERS_CSV, DB_READER_USER, DB_READER_PASSWORD

def setup_reader_role(engine):
    if not DB_READER_PASSWORD:
        raise ValueError("DB_READER_PASSWORD is not set")

    quote = engine.dialect.identifier_preparer.quote
    role = quote(DB_READER_USER)
    db = quote(engine.url.database)
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg+roles WHERE rolname = :n"),
            {"n": DB_READER_USER},
        ).scalar()
        pw = conn.execute(
            text("SELECT quote_literal(:pw)"), {"pw": DB_READER_PASSWORD}
        ).scalar()
        verb = "ALTER" if exists else "CREATE"
        conn.execute(text(f"{verb} ROLE {role} LOGIN PASSWORD {pw}"))
        conn.execute(text(f"GRANT CONNECT ON DATABASE {db} TO {role}"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        conn.execute(text(f"GRANT SELECT ON orders TO {role}"))
        conn.execute(
            text(f"ALTER ROLE {role} SET default_transaction_read_only = on")
        )

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

print("Setting up read-only role for the agent...")
setup_reader_role(engine)

print("ETL proccess completed succesfully! Data is now in PostgreSQL.")


