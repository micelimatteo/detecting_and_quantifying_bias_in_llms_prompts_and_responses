import requests
import json 

def generate_llm_response(prompt, model="gemma3", ollama_url="http://localhost:11434/api/generate"):
    """Generate response from local LLM via Ollama API"""
    payload = {
        "model": model,
        "prompt": prompt + "\n Please answer with no more than 512 tokens.",
        "stream": False
    }
    response = requests.post(ollama_url, json=payload)
    response.raise_for_status()
    resp = response.json()["response"]
    # remove leading/trailing Markdown fences (``` or ```json)
    if resp.startswith("```"):
        # drop the first fence line (e.g. ```json)
        first_newline = resp.find("\n")
        if first_newline != -1:
            resp = resp[first_newline+1:]
        # remove trailing fence if present
        if resp.rstrip().endswith("```"):
            resp = resp.rstrip()[:-3].rstrip()
    return resp