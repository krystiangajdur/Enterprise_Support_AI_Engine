import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agent import ask
from app.config import INDEX_FILE

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Enterprise Support AI Engine")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: str | None = None

class Answer(BaseModel):
    answer: str

@app.post("/ask", response_model = Answer)
@limiter.limit("10/minute")
def ask_endpoint(request: Request, body: Question, x_customer_id: str | None = Header(default=None)):
    customer_id = (x_customer_id or "").strip()
    question = body.question.strip()

    if not customer_id:
        raise HTTPException(status_code=401, detail="Missing X-Customer-Id header")
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")

    try:
        result = ask(
            question=question,
            customer_id=customer_id,
            session_id=body.session_id,
        )
        return Answer(answer=result["answer"])
    except Exception as exc:
        logger.exception("ask() failed")
        if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
            raise HTTPException(status_code=429, detail="Too many requests, try again in a minute")
        raise HTTPException(status_code=503, detail="The assistant is temporarily unavailable")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_FILE)