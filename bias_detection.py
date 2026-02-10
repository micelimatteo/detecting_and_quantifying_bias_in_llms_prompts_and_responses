from textblob import TextBlob
from explainability import explain_bias
from utils import generate_llm_response
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Force load BERT tokenizer (the model has wrong tokenizer config)
from transformers import BertTokenizer
from detoxify import Detoxify
import json


#HateBERT fine-tuned on ToxicGen dataset
def identify_implicit_toxicity_biases(phrase):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_name = "tomh/toxigen_hatebert"
    toxigen_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    toxigen_model_inputs = tokenizer(phrase, return_tensors="pt")
    toxigen_outputs = toxigen_model(**toxigen_model_inputs)
    toxicity_score = toxigen_outputs.logits.softmax(dim=1)[0, 1]
    return float(toxicity_score.item())

def identify_subjectivity(phrase):
    blob = TextBlob(phrase)
    subjectivity_score = blob.sentiment.subjectivity
    return subjectivity_score

def identify_general_toxicity_biases(phrase):
    detox_model = Detoxify('original')
    detox_result = detox_model.predict(phrase)
    return detox_result


def identify_cognitive_biases_through_debate(phrase: str, is_prompt: bool) -> list:
    """
    Debate-based cognitive bias detection:
    1. Agent A argues biases ARE present (explanation only)
    2. Agent B argues biases are NOT present (explanation only)
    3. Neutral Analyst synthesizes and decides
    """
    import re
    
    # --- AGENT A: Bias Advocate ---
    advocate_prompt = f"""You are a critical analyst specializing in detecting cognitive biases.

Analyze this {"prompt" if is_prompt else "response"}: "{phrase}"

Argue STRONGLY that cognitive biases ARE present. Identify:
- Framing bias, anchoring, stereotyping, confirmation seeking, false premises
- Quote exact phrases that show bias (use "quotes")
- Explain the psychological impact

Provide a natural language explanation (1-2 paragraphs). Do NOT use JSON."""
    
    # --- AGENT B: Bias Skeptic ---
    skeptic_prompt = f"""You are a fair-minded analyst who challenges bias claims.

Analyze this {"prompt" if is_prompt else "response"}: "{phrase}"

Argue STRONGLY that cognitive biases are NOT present or are minimal. 
- Challenge potential bias claims with counter-evidence
- Explain benign interpretations (use "quotes" for key phrases)
- Point out where the text is neutral or factual

Provide a natural language explanation (1-2 paragraphs). Do NOT use JSON."""
    
    # --- NEUTRAL ANALYST: Mediator ---
    def create_analyst_prompt(advocate_response, skeptic_response):
        return f"""You are a neutral cognitive bias analyst. Two experts debated this text:

TEXT: "{phrase}"

ADVOCATE (claims biases present):
{advocate_response}

SKEPTIC (claims biases absent):
{skeptic_response}

Render a fair judgment:
- Weigh evidence from both sides
- Identify biases ACTUALLY present (high confidence only)
- Discard weak or spurious claims
- For each detected bias, provide: (1) bias type, (2) severity 0-1, (3) confidence 0-1

Respond in JSON only:
{{
  "debate_summary": "brief summary",
  "final_biases": [
    {{"bias_type": "framing", "present": true, "severity": 0.7, "confidence": 0.85}},
    ...
  ],
  "consensus": "final verdict"
}}"""
    
    # --- HELPER: Extract quoted phrases from text ---
    def extract_quoted_phrases(text):
        """Extract all quoted strings from text"""
        matches = re.findall(r'"([^"]+)"', text)
        return matches
    
    # --- HELPER: Robust JSON parser for analyst response ---
    def parse_json_response(resp):
        """Parse JSON from response, handling markdown and control chars."""
        resp = resp.strip()
        
        # Remove markdown fences
        if resp.startswith("```"):
            first_newline = resp.find("\n")
            if first_newline != -1:
                resp = resp[first_newline+1:]
            if resp.rstrip().endswith("```"):
                resp = resp.rstrip()[:-3].rstrip()
        
        # Extract JSON object
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1 and end > start:
            resp = resp[start:end+1]
        
        # Strip control chars
        resp = ''.join(ch if ord(ch) >= 32 or ch in '\t\n\r' else ' ' for ch in resp)
        
        # Clean up spacing around JSON punctuation
        resp = re.sub(r'"\s*,\s*"', '", "', resp)
        resp = re.sub(r'"\s*:\s*{', '": {', resp)
        resp = re.sub(r'}\s*,\s*{', '}, {', resp)
        
        return json.loads(resp)
    
    # --- EXECUTION ---
    try:
        print("Calling advocate agent...")
        advocate_response = generate_llm_response(advocate_prompt)
        print(f"Advocate response:\n{advocate_response}\n")
        
        print("Calling skeptic agent...")
        skeptic_response = generate_llm_response(skeptic_prompt)
        print(f"Skeptic response:\n{skeptic_response}\n")
        
        # Extract quoted phrases from both responses (no JSON parsing needed!)
        advocate_quotes = extract_quoted_phrases(advocate_response)
        skeptic_quotes = extract_quoted_phrases(skeptic_response)
        
        print(f"Advocate quotes: {advocate_quotes}")
        print(f"Skeptic quotes: {skeptic_quotes}\n")
        
        # Get analyst judgment
        analyst_prompt = create_analyst_prompt(advocate_response, skeptic_response)
        print("Calling analyst agent...")
        analyst_response = generate_llm_response(analyst_prompt)
        print(f"Analyst response:\n{analyst_response}\n")
        
        try:
            analyst_data = parse_json_response(analyst_response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse analyst response: {e}")
            print(f"Raw analyst response: {repr(analyst_response[:500])}")
            return []
        
        # Extract final biases and use advocate quotes as problematic phrases
        final_biases = []
        for bias in analyst_data.get("final_biases", []):
            if bias.get("present") and bias.get("severity", 0) > 0.5:
                final_biases.append({
                    bias["bias_type"]: {
                        "present": True,
                        "severity": bias.get("severity", 0),
                        "confidence": bias.get("confidence", 0.5),
                        "details": advocate_response,  # Full explanation
                        "problematic_phrases": advocate_quotes  # Extracted quotes
                    }
                })
        
        print(f"Debate consensus: {analyst_data.get('consensus')}\n")
        return final_biases
        
    except Exception as e:
        print(f"Error in debate-based detection: {e}")
        import traceback
        traceback.print_exc()
        return []




# Identify cognitive biases in prompt or response
def identify_cognitive_biases(phrase, is_prompt, use_debate=True):
    """Detect bias-inducing patterns in user prompt"""
    
    if is_prompt:
        if use_debate:
            return identify_cognitive_biases_through_debate(phrase, is_prompt)
        prompt = f"""Analyze this prompt for cognitive biases that might influence an LLM's response:

        Prompt: "{phrase}"

        Check for:
        1. **Framing bias**: Is it framed in a leading way (positive/negative)?
        2. **Confirmation seeking**: Is it asking for confirmation of a specific view?
        3. **Anchoring**: Does it contain specific numbers/values that might anchor the response?
        4. **Stereotyping**: Does it generalize about groups of people or entities?

        Respond in JSON format:
        {{
        "anchoring": {{"present": bool, "details": str, "problematic_phrases": [str],"severity": a score from 0-1}},
        "framing": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": a score from 0-1}},
        "stereotyping": {{"present": bool, "details": str,"problematic_phrases": [str], "severity": a score from 0-1}},
        "confirmation_seeking": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": a score from 0-1}}
        }}
        where "present" is true if the bias is detected, "details" explain why, "problematic_phrases" list the parts of the prompt that indicate the bias, and "severity" is a float from 0 (low) to 1 (high) indicating how strongly the bias is present."""
    else:
        prompt = f"""Analyze the following LLM response for cognitive biases and problematic rhetorical devices that could mislead or unduly influence a reader.
        Response to analyze: "{phrase}"

        Check for these categories (detect presence, explain why, show evidence excerpts, give a severity 0.0-1.0, and propose a brief mitigation/sanitization suggestion):
        - anchoring (relying on specific numbers/values without justification)
        - framing (leading or emotionally loaded framing)
        - stereotyping (generalizations about groups)
        - false_premises (asserting unverified or incorrect facts)
        - confirmation_seeking / echoing (reinforcing a single viewpoint)
        - overconfidence (presenting uncertain info as certain)
        - omission_bias (leaving out important caveats or alternatives)
        - moralizing or emotional manipulation

        Respond in JSON only, with this exact schema:
        {{
        "anchoring": {{"present": bool, "details": str, "problematic_phrases": [str],"severity": a score from 0-1}},
        "framing": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": a score from 0-1}},
        "stereotyping": {{"present": bool, "details": str,"problematic_phrases": [str], "severity": a score from 0-1}},
        "false_premises": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": a score from 0-1}},
        "confirmation_seeking": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": a score from 0-1}}
        }}
        where "present" is true if the bias is detected, "details" explain why, "problematic_phrases" list the parts of the response that indicate the bias, and "severity" is a float from 0 (low) to 1 (high) indicating how strongly the bias is present."""
    response = generate_llm_response(prompt)
    biases = []
    
    for bias, info in json.loads(response).items():
        if info["present"]:
            biases.append({bias: info})
    return biases


#Returns the only bias that would be displayed to the user
def aggregate_biases(biases: list, phrase: str) -> tuple:
    bias = {}
    with open('params.json', 'r') as f:
        params = json.load(f)
    toxigen_threshold = params.get("toxigen_threshold", 0.5)
    detoxify_threshold = params.get("detoxify_threshold", 0.5)
    subjectivity_threshold = params.get("subjectivity_threshold", 0.5)

    # Safe check: ensure keys exist before accessing them
    if "toxicity_bias" in biases:
        tb = biases["toxicity_bias"]
        if tb.get("present") and tb.get("severity", 0) >= toxigen_threshold:
            return {"toxicity_bias": tb}, "toxicity_bias"

    # Find any detoxify_* bias that meets the threshold
    for k, v in biases.items():
        if k.startswith("detoxify_") and isinstance(v, dict):
            if v.get("present") and v.get("severity", 0) >= detoxify_threshold:
                return {k: v}, k

    # If there are cognitive biases listed, prefer them (check presence first, before subjectivity threshold)
    for cb in biases.get("cognitive_biases", []):
        if isinstance(cb, dict):
            for bias_name, info in cb.items():
                # If the cognitive bias is reported as present, return it (no additional threshold here)
                if info.get("present", True) or info.get("severity", 0) > 0:
                    return {bias_name: info}, bias_name

    # Subjectivity fallback
    if "subjectivity_bias" in biases:
        sb = biases["subjectivity_bias"]
        if sb.get("present") and sb.get("severity", 0) >= subjectivity_threshold:
            return {"subjectivity_bias": sb}, "subjectivity_bias"
    return {}, None


#Identify all biases in the prompt
def identify_biases(phrase: str, is_prompt: bool) -> dict:
    with open('params.json', 'r') as f:
        params = json.load(f)
    toxigen_min_threshold = params.get("toxigen_min_threshold", 0.2)
    detoxify_min_threshold = params.get("detoxify_min_threshold", 0.2)
    subjectivity_threshold = params.get("subjectivity_threshold", 0.5)
    cognitive_bias_threshold = params.get("cognitive_bias_threshold", 0.6)

    toxicity_score = identify_implicit_toxicity_biases(phrase)
    detoxify_score = identify_general_toxicity_biases(phrase)
    subjectivity_score = identify_subjectivity(phrase)
    cognitive_biases = identify_cognitive_biases(phrase, is_prompt)
    biases = {}

    if toxicity_score > toxigen_min_threshold:
        biases["toxicity_bias"] = {"present": True, "severity": toxicity_score}
    for label, score in detoxify_score.items():
        if score > detoxify_min_threshold:
            biases[f"detoxify_{label}_bias"] = {"present": True, "severity": score}
    if subjectivity_score > subjectivity_threshold:
        biases["subjectivity_bias"] = {"present": True, "severity": subjectivity_score}
    biases["cognitive_biases"] = []
    for cb in cognitive_biases:
        for bias_name, info in cb.items():
            if info["severity"] > cognitive_bias_threshold:
                biases["cognitive_biases"].append({bias_name: info})
    print(f"Identified biases: {biases}\n")
    bias, bias_type =  aggregate_biases(biases, phrase)

    if bias_type is None:
       return None
    else:
       return explain_bias(bias, bias_type, phrase)