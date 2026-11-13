from fastapi import FastAPI
from pydantic import BaseModel

from agent import ask

app = FastAPI(title = "ModaLoop  AI", description = "API do obsługi klienta.")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_with_agent(request: ChatRequest):
    agents_answer = ask(request.message)

    return {"odpowiedz:": agents_answer}

