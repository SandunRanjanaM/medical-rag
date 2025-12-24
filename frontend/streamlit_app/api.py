# frontend/streamlit_app/api.py
import requests

BACKEND_URL = "http://localhost:8000/ask"

def ask_backend(specialty: str, question: str):
    payload = {
        "specialty": specialty,
        "question": question
    }

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
