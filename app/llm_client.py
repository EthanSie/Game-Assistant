# app/llm_client.py
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3:8b"   # or whatever you're using

def call_llm(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_predict": 128  # cap length so it answers quicker
        }
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]
