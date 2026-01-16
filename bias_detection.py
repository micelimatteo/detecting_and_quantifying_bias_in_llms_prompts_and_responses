from textblob import TextBlob
from torch import threshold
from utils import generate_llm_response
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Force load BERT tokenizer (the model has wrong tokenizer config)
from transformers import BertTokenizer
from detoxify import Detoxify

def identify_implicit_toxicity_biases(phrase):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_name = "tomh/toxigen_hatebert"
    toxigen_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    toxigen_model_inputs = tokenizer(phrase, return_tensors="pt")
    toxigen_outputs = toxigen_model(**toxigen_model_inputs)
    toxicity_score = toxigen_outputs.logits.softmax(dim=1)[0, 1]
    return float(toxicity_score)


def identify_subjectivity_and_polarity(phrase):
    blob = TextBlob(phrase)
    subjectivity_score = blob.sentiment.subjectivity
    polarity_score = blob.sentiment.polarity
    return subjectivity_score, polarity_score


def identify_general_toxicity_biases(phrase):
    detox_model = Detoxify('original')
    detox_result = detox_model.predict(phrase)
    return float(detox_result)


def identify_cognitive_biases(phrase):
    """Detect bias-inducing patterns in user prompt"""
    
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
    "anchoring": {{"present": bool, "details": str, "problematic_phrases": [str],"severity": 0-1}},
    "framing": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": 0-1}},
    "stereotyping": {{"present": bool, "details": str,"problematic_phrases": [str], "severity": 0-1}},
    "false_premises": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": 0-1}},
    "confirmation_seeking": {{"present": bool, "details": str, "problematic_phrases": [str], "severity": 0-1}}
    }}
    where "present" is true if the bias is detected, "details" explain why, "problematic_phrases" list the parts of the prompt that indicate the bias, and "severity" is a float from 0 (low) to 1 (high) indicating how strongly the bias is present."""
    
    return generate_llm_response(prompt)


def identify_biases(prompt: str) -> dict:
    # Check for toxicity bias
    toxicity_score = identify_implicit_toxicity_biases(prompt)
    if toxicity_score > threshold:
        biases = {"toxicity_bias": {"present": True, "severity": toxicity_score}}
    
    cognitive_biases = identify_cognitive_biases(prompt)

    return aggregate_biases(biases, cognitive_biases)