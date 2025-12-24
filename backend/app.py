from fastapi import FastAPI
from router import route_question

app = FastAPI()

@app.post("/ask")
def ask(payload: dict):
    return route_question(
        payload["specialty"],
        payload["question"]
    )
