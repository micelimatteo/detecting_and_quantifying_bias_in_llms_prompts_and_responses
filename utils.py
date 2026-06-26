import requests
import json

def generate_llm_response(prompt, model="gemma3", ollama_url="http://localhost:11434/api/generate",
                           json_mode=False, num_predict=512):
    """Generate response from local LLM via Ollama API.

    json_mode=True asks Ollama for grammar-constrained JSON output
    (Ollama's "format": "json" option) instead of relying on the model
    to follow a textual instruction — this is what actually prevents
    unparsable output, rather than just hoping the model behaves.

    num_predict caps generation length via Ollama's own `options`, which
    is enforced server-side; a prior version asked for this in the prompt
    text ("answer with no more than 512 tokens"), which models frequently
    ignored or which truncated structured JSON output mid-object.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    if json_mode:
        payload["format"] = "json"

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