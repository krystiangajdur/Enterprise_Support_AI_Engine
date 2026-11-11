import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import Tool

print('Ladowanie bazy wektorowej RAG...')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory = "./chroma.db",
    embedding_function = embeddings
)

retriever = vectorstore.as_retriever(search_kwargs = {"k":2})

def search_internal_policy(query: str):
    docs = retriever.invoke(query)
    if not docs:
        return 'Nie znaleziono odpowiednich informacji w regulaminie'
    return '\n---\n'.join([doc.page_content for doc in docs])

rag_tool = Tool(
    name = "PolicySearch",
    func = search_internal_policy,
    description = "Przydatne, gdy klient pyta o zasady firmy, regulamin, polityke zwrotow, logistyke lub reklamacje."
)

def check_order_status(order_id: str):
    clean_id = order_id.strip()
    if not clean_id:
        return f'Zamówienie o numerze {clean_id} ma status: Wyslane / W drodze do klienta.'

sql_tool = Tool(
    name = 'CheckOrderStatus',
    func = check_order_status,
    description = "Przydatne, gdy klient pyta o status swojego konkretnego zamowienia."
)

tools = [rag_tool, sql_tool]

registered_tools = [t.name for t in tools]
registered_tools_str = ''.join([i + ', ' for i in registered_tools])

def run_agent_test():
    print('System agenta hybrydowego gotowy do inicjalizacji!')
    print(f'Zarejestrowane narzedzia: {registered_tools_str}')

if __name__ == "__main__":
    run_agent_test()