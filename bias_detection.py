from textblob import TextBlob
from explainability import explain_bias
from utils import generate_llm_response
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Force load BERT tokenizer (the model has wrong tokenizer config)
from transformers import BertTokenizer
from detoxify import Detoxify
import json

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


def identify_cognitive_biases(phrase, is_prompt):
    """Detect bias-inducing patterns in user prompt"""
    
    if is_prompt:
        prompt = f"""Analyze this prompt for cognitive biases that might influence an LLM's response:

        Prompt: "{phrase}"

        Check for:
        1. **Anchoring**: Does it contain specific numbers/values that might anchor the response?
        2. **Framing bias**: Is it framed in a leading way (positive/negative)?
        3. **Stereotyping**: Does it generalize about groups of people or entities?
        4. **False premises**: Does it assume something controversial as fact?
        5. **Confirmation seeking**: Is it asking for confirmation of a specific view?

        Respond in JSON format:
        {{
        "anchoring": {{"present": bool, "details": str, "problematic_phrases": [str],"severity": a score from 0-1}},
        "framing": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": a score from 0-1}},
        "stereotyping": {{"present": bool, "details": str,"problematic_phrases": [str], "severity": a score from 0-1}},
        "false_premises": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": a score from 0-1}},
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


#returns the only bias that would be displayed to the user
def aggregate_biases(biases: list, phrase: str) -> tuple:
    bias = {}
    with open('params.json', 'r') as f:
        params = json.load(f)
    toxigen_threshold = params.get("toxigen_threshold", 0.5)
    detoxify_threshold = params.get("detoxify_threshold", 0.5)
    subjectivity_threshold = params.get("subjectivity_threshold", 0.5)

    if biases["toxicity_bias"]["severity"] >= toxigen_threshold:
        bias = {"toxicity_bias": biases["toxicity_bias"]}
        bias_type = "toxicity_bias"
        return bias, bias_type
    elif biases["detoxify_bias"]["severity"] >= detoxify_threshold:
        bias = {"detoxify_bias": biases["detoxify_bias"]}
        bias_type = "detoxify_bias"
        return bias, bias_type
    for cb in biases["cognitive_biases"]:
        for bias_name, info in cb.items():
            if bias_name not in bias and biases["toxicity_bias"] == True:
                bias = {bias_name: info}
                bias_type = bias_name
    if bias:
        return bias, bias_type
    elif "subjectivity_bias" in biases and biases["subjectivity_bias"]["severity"] >= subjectivity_threshold:
        bias = {"subjectivity_bias": biases["subjectivity_bias"]}
        bias_type = "subjectivity_bias"
        return bias, bias_type
    else:
        return {}, None
    

#Identify all biases in the prompt
def identify_biases(phrase: str, is_prompt: bool) -> dict:
    with open('params.json', 'r') as f:
        params = json.load(f)
    toxigen_min_threshold = params.get("toxigen_min_threshold", 0.2)
    detoxify_min_threshold = params.get("detoxify_min_threshold", 0.2)
    subjectivity_threshold = params.get("subjectivity_threshold", 0.5)
    cognitive_bias_threshold = params.get("cognitive_bias_threshold", 0.6)

    # Check for toxicity bias
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
    print("Detected biases:", biases)
    bias, bias_type =  aggregate_biases(biases, phrase)
    if bias_type is None:
       return None
    else:
       return explain_bias(bias, bias_type, phrase)