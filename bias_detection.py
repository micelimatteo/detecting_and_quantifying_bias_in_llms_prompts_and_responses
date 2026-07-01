from textblob import TextBlob
import torch
from explainability import explain_bias
from utils import generate_llm_response
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
# Force load BERT tokenizer (the model has wrong tokenizer config)
from transformers import BertTokenizer
from detoxify import Detoxify
import json
import re
local_path = "./toxigen_hatebert"


# NOTE: this used to default to True, which made identify_biases() always
# return (bias_type, None, None) — skipping explain_bias() entirely and
# discarding tokens/explanation for every caller (GUI.py, CLI.py). Nothing
# in the codebase ever flips this flag; identify_biases_test() already
# exists as the dedicated bias-type-only function for batch evaluation.
test = False


def parse_json_response(resp: str) -> dict:
    """Robust JSON parser for LLM output.

    Even with Ollama's json_mode forcing syntactically-valid JSON, the
    *content* can still arrive wrapped in markdown fences, with stray
    leading/trailing prose, or truncated mid-object if num_predict was too
    low. This handles all three before falling back to json.loads.
    """
    resp = resp.strip()

    if resp.startswith("```"):
        first_newline = resp.find("\n")
        if first_newline != -1:
            resp = resp[first_newline + 1:]
        if resp.rstrip().endswith("```"):
            resp = resp.rstrip()[:-3].rstrip()

    start = resp.find("{")
    end = resp.rfind("}")
    if start != -1 and end != -1 and end > start:
        resp = resp[start:end + 1]

    # Strip control characters that break json.loads
    resp = ''.join(ch if ord(ch) >= 32 or ch in '\t\n\r' else ' ' for ch in resp)

    # Python-style literals that some models emit instead of JSON's
    resp = re.sub(r'\bTrue\b', 'true', resp)
    resp = re.sub(r'\bFalse\b', 'false', resp)
    resp = re.sub(r'\bNone\b', 'null', resp)

    # Trailing commas before a closing bracket/brace (valid in Python, not JSON)
    resp = re.sub(r',\s*([}\]])', r'\1', resp)

    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        pass

    # Last resort: the response was likely truncated mid-object (hit
    # num_predict). Trim back to the last complete "key": value pair and
    # close out whatever braces/brackets are still open.
    repaired = resp
    last_comma = repaired.rfind(",")
    if last_comma != -1:
        repaired = repaired[:last_comma]
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    repaired += "]" * max(open_brackets, 0) + "}" * max(open_braces, 0)
    return json.loads(repaired)  # let this raise if still unrecoverable


def call_llm_for_json(prompt: str, num_predict: int = 768, retries: int = 1) -> dict:
    """generate_llm_response + parse_json_response, with json_mode enabled
    and one retry (with a stricter instruction) if parsing still fails."""
    last_error = None
    for attempt in range(retries + 1):
        attempt_prompt = prompt
        if attempt > 0:
            attempt_prompt += ("\n\nYour previous reply could not be parsed as JSON. "
                                "Respond with ONLY a single valid JSON object — no markdown "
                                "fences, no commentary before or after it.")
        response = generate_llm_response(attempt_prompt, json_mode=True, num_predict=num_predict)
        try:
            return parse_json_response(response)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"  [json parse attempt {attempt + 1}/{retries + 1} failed] {e}")
            print(f"  Raw response: {repr(response[:500])}")
    raise last_error

#HateBERT fine-tuned on ToxicGen dataset
def identify_implicit_toxicity_biases(phrase):
    tokenizer = AutoTokenizer.from_pretrained(local_path, 
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]")
    toxigen_model = AutoModelForSequenceClassification.from_pretrained(local_path)

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

    # --- EXECUTION ---
    try:
        print("Calling advocate agent...")
        advocate_response = generate_llm_response(advocate_prompt, num_predict=512)
        print(f"Advocate response:\n{advocate_response}\n")

        print("Calling skeptic agent...")
        skeptic_response = generate_llm_response(skeptic_prompt, num_predict=512)
        print(f"Skeptic response:\n{skeptic_response}\n")

        # Extract quoted phrases from both responses (no JSON parsing needed!)
        advocate_quotes = extract_quoted_phrases(advocate_response)
        skeptic_quotes = extract_quoted_phrases(skeptic_response)

        print(f"Advocate quotes: {advocate_quotes}")
        print(f"Skeptic quotes: {skeptic_quotes}\n")

        # Get analyst judgment — json_mode forces syntactically valid JSON
        # output instead of relying on the model to follow the textual
        # "Respond in JSON only" instruction; one retry with a stricter
        # nudge if it still fails to parse.
        analyst_prompt = create_analyst_prompt(advocate_response, skeptic_response)
        print("Calling analyst agent...")
        try:
            analyst_data = call_llm_for_json(analyst_prompt, num_predict=768)
        except json.JSONDecodeError as e:
            print(f"Failed to parse analyst response after retry: {e}")
            return []
        print(f"Analyst data: {analyst_data}\n")

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
def identify_cognitive_biases(phrase, is_prompt, use_debate=False):
    """Detect bias-inducing patterns in user prompt"""
    
    if is_prompt:
        if use_debate:
            return identify_cognitive_biases_through_debate(phrase, is_prompt)
        prompt = prompt = f"""Analyze this prompt for cognitive biases that might influence an LLM's response:

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
        prompt = f"""Analyze this LLM response for cognitive biases:

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

    try:
        # json_mode forces syntactically valid JSON server-side, with one
        # retry (stricter nudge) if it still fails to parse — see
        # call_llm_for_json / parse_json_response above.
        data = call_llm_for_json(prompt, num_predict=768)
        biases = []

        for bias_name, info in data.items():
            if isinstance(info, dict) and info.get("present"):
                biases.append({
                    bias_name: {
                        "present": True,
                        "severity": info.get("severity", 0.5),
                        "details": info.get("details", ""),
                        "problematic_phrases": info.get("problematic_phrases", [])
                    }
                })
        print(biases)
        return biases

    except json.JSONDecodeError as e:
        print(f"Error parsing cognitive bias response after retry: {e}")
        return []
    except Exception as e:
        print(f"Error in identify_cognitive_biases: {e}")
        return []



#Returns the only bias that would be displayed to the user
def aggregate_biases(biases: list, phrase: str) -> tuple:
    bias = {}
    with open('params.json', 'r') as f:
        params = json.load(f)
    toxigen_threshold = params.get("toxigen_threshold", 0.5)
    detoxify_threshold = params.get("detoxify_threshold", 0.5)
    cognitive_bias_threshold = params.get("cognitive_bias_threshold", 0.5)

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

    # If there are cognitive biases listed, prefer them
    for cb in biases.get("cognitive_biases", []):
        if cb != []:
            if isinstance(cb, dict):
                for bias_name, info in cb.items():
                    print(f"Checking cognitive bias {bias_name} with info: {info}")
                    # If the cognitive bias is reported as present, return it
                    if info.get("present", True) or info.get("severity", 0) >= cognitive_bias_threshold:
                        return {bias_name: info}, bias_name

    return {}, None


#Identify all biases in the prompt or response and return the most severe one with biased tokens and an explanation. If test is True, return only the bias type for testing purposes.
def identify_biases(phrase: str, is_prompt: bool) -> dict:
    with open('params.json', 'r') as f:
        params = json.load(f)
    toxigen_min_threshold = params.get("toxigen_min_threshold", 0.2)
    detoxify_min_threshold = params.get("detoxify_min_threshold", 0.2)
    cognitive_bias_threshold = params.get("cognitive_bias_threshold", 0.6)

    toxigen_threshold = params.get("toxigen_threshold", 0.5)
    detoxify_threshold = params.get("detoxify_threshold", 0.5)

    # Fast-Path: toxicity classifiers always run first.
    toxicity_score = identify_implicit_toxicity_biases(phrase)
    detoxify_score = identify_general_toxicity_biases(phrase)
    subjectivity_score = identify_subjectivity(phrase)
    biases = {}

    if toxicity_score > toxigen_min_threshold:
        biases["toxicity_bias"] = {"present": True, "severity": toxicity_score}
    for label, score in detoxify_score.items():
        if score > detoxify_min_threshold:
            biases[f"detoxify_{label}_bias"] = {"present": True, "severity": score}
    biases["subjectivity_score"] = subjectivity_score
    biases["cognitive_biases"] = []

    # Deep-Path: debate runs only when no toxicity threshold is exceeded.
    # Overtly toxic inputs are already flagged; the debate targets the subtler
    # class of polite-but-biased prompts that the Fast-Path misses.
    already_toxic = (
        toxicity_score >= toxigen_threshold
        or any(s >= detoxify_threshold for s in detoxify_score.values())
    )
    if not already_toxic:
        cognitive_biases = identify_cognitive_biases(phrase, is_prompt)
        for cb in cognitive_biases:
            for bias_name, info in cb.items():
                if info["severity"] > cognitive_bias_threshold:
                    biases["cognitive_biases"].append({bias_name: info})
    print(f"Identified biases: {biases}\n")


    bias, bias_type =  aggregate_biases(biases, phrase)
    if test:
            return bias_type, None, None
    if bias_type is None:
       return None, None, None
    else:
       return explain_bias(bias, bias_type, phrase)
    


#Identify all biases in the prompt or response for testing
def identify_biases_test(phrase: str, is_prompt: bool) -> dict:
    with open('params.json', 'r') as f:
        params = json.load(f)
    toxigen_min_threshold = params.get("toxigen_min_threshold", 0.2)
    detoxify_min_threshold = params.get("detoxify_min_threshold", 0.2)
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
    # Cognitive-bias debate is always run and considered; subjectivity is reported
    # as explanatory context only, not as a gate (see tests_scripts/PIVOT_NOTES.md).
    biases["subjectivity_score"] = subjectivity_score
    biases["cognitive_biases"] = []
    for cb in cognitive_biases:
        for bias_name, info in cb.items():
            if info["severity"] > cognitive_bias_threshold:
                biases["cognitive_biases"].append({bias_name: info})
    print(f"Identified biases: {biases}\n")
    return biases