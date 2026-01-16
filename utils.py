import requests

def generate_llm_response(prompt, model="gemma3", ollama_url="http://localhost:11434/api/generate"):
    """Generate response from local LLM via Ollama API"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(ollama_url, json=payload)
    response.raise_for_status()
    return response.json()["response"]