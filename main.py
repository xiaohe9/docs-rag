from fastapi import FastAPI, Query
from rag import ask

app = FastAPI(title="陈晓河项目文档问答", version="0.1.0")

@app.get("/ask")
def ask_api(q: str = Query(..., min_length=2)):
    return ask(q)

@app.get("/health")
def health():
    return {"status": "ok"}