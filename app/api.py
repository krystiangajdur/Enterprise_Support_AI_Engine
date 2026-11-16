from fastapi import FastAPI,Header, HTTPException
from pydantic import BaseModel

from app.agent import ask

app = FastAPI(title = "Enterprise Support AI Engine")

class Question(BaseModel):
    question: str
    session_id: str

class Answer(BaseModel):
    answer: str

@app.post("/ask", response_model = Answer)
def ask_endpoint(body: Question, x_customer_id: str | None = Header(default=None)):
    if not x_customer_id or not x_customer_id.strip():
        raise HTTPException(status_code = 401, detail = "Missing X-Customer_Id header")
    try:
        return Answer(answer = ask(body.question, customer_id = x_customer_id.strip(), session_id=body.session_id ))
    except Exception:
        raise HTTPException(status_code = 401, detail = "The assistant is temporarily unavailable")