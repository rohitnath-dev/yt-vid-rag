from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag import app as rag_app

app = FastAPI()

# CORS (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "AI Backend Running"}

@app.post("/ask")
def ask(q: Query):
    result = rag_app.invoke({
        "question": q.question,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
    })

    return {
        "answer": result["answer"]
}
