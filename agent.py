import faulthandler
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
faulthandler.dump_traceback_later(30, exit=True)

load_dotenv()

print('Ladowanie bazy wektorowej RAG...')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory = "./chroma_db",
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
        return "Nie podano numeru zamówienia."
    return f'Zamówienie o numerze {clean_id} ma status: Wyslane / W drodze do klienta.'

sql_tool = Tool(
    name = 'CheckOrderStatus',
    func = check_order_status,
    description = "Przydatne, gdy klient pyta o status swojego konkretnego zamowienia."
)

print("Inicjalizowanie modelu Gemini...")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout = 30, max_retries = 1)

tools = [rag_tool, sql_tool]


agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt = (
        "Jesteś asystentem obsługi klienta sklepu ModaLoop. Twoim JEDYNYM zadaniem jest pomoc "
        "w sprawach zamówień, logistyki i regulaminu sklepu. "
        "Jeśli użytkownik zapyta o cokolwiek niezwiązanego z działaniem sklepu (np. o aktualną datę, "
        "pogodę, porady życiowe czy wiedzę ogólną), grzecznie odmów i poinformuj, że zajmujesz się "
        "wyłącznie obsługą sklepu ModaLoop. "
        "Odpowiadaj po polsku. Na pytania o regulamin odpowiadaj WYŁĄCZNIE na podstawie "
        "wyników narzędzia PolicySearch. "
        "Zawsze na końcu odpowiedzi dodaj: "
        "Jeśli odpowiedź cię nie satysfakcjonuje, spróbuj sformułować pytanie w inny sposób. "
        "Jak masz jeszcze jakieś pytania - zadaj je - postaram się odpowiedzieć najlepiej jak potrafię. :) "
    ),
)

def extract_text(content):
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )

def ask(question):
    result = agent.invoke({"messages": [{"role":"user", "content": question}]})

    for m in result["messages"]:
        for call in getattr(m, "tool_calls", []) or []:
            print(f'[narzedzie] {call['name']}({call['args']})')
    return extract_text(result['messages'][-1].content)

if __name__ == "__main__":
    print("\nSystem agenta hybrydowego gotowy do pracy!\n")

    print("--- TEST 1: Pytanie do RAG ---")
    print(f"\nOdpowiedź agenta:\n{ask('Napisz wszystkie możliwości dostawy wraz z kosztami.')}\n")

    print("--- TEST 2: Pytanie o zamówienie ---")
    print(f"\nOdpowiedź agenta:\n{ask('Co powinienem zjeść jutro na śniadanie?')}\n")