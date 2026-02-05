from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Force load BERT tokenizer (the model has wrong tokenizer config)
from transformers import BertTokenizer
from lime.lime_text import LimeTextExplainer
import shap
import torch
import numpy as np
from transformers_interpret import SequenceClassificationExplainer
from textblob import TextBlob
from detoxify import Detoxify
from functools import partial


#HateBERT (Toxigen)
def toxigen_proba(texts):
    model_name = "tomh/toxigen_hatebert"
    toxigen_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Ensure texts is a list of strings
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, (list, tuple)):
        texts = [str(t) if not isinstance(t, str) else t for t in texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    else:
        texts = [str(texts)]
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = toxigen_model(**enc).logits.cpu().numpy()
    # softmax to get probabilities
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs  # shape (n_samples, n_classes)

def explain_toxigen_with_lime(text):
    model_name = "tomh/toxigen_hatebert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # LIME explanation for the model
    lime_explainer_model = LimeTextExplainer(class_names=[f"class{i}" for i in range(model.config.num_labels)])
    lime_exp = lime_explainer_model.explain_instance(text, toxigen_proba, num_features=10)
    return lime_exp.as_list()

def explain_toxigen_with_shap(text):
    model_name = "tomh/toxigen_hatebert"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # SHAP explanation for the model
    shap_explainer_model = shap.Explainer(toxigen_proba, shap.maskers.Text(tokenizer))
    shap_values = shap_explainer_model([text])
    return shap_values[:, :, 1]  # return shap values for class 1 (toxic)

def explain_toxigen_with_transformers_interpret(text):
    model_name = "tomh/toxigen_hatebert"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Transformers Interpret explanation for the model
    seq_explainer = SequenceClassificationExplainer(model, tokenizer)
    attributions = seq_explainer(text)
    return attributions

#TextBlob Subjectivity
def tb_subjectivity_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    
    texts = [str(t) if t else "" for t in texts]
    subj = np.array([TextBlob(t).sentiment.subjectivity if t else 0.0 for t in texts])
    p_subj = np.clip(subj, 0.0, 1.0)
    p_obj = 1.0 - p_subj
    return np.vstack([p_obj, p_subj]).T


def explain_subjectivity_with_lime(text):
    lime_explainer_model =LimeTextExplainer(class_names=["objective", "subjective"])
    lime_exp = lime_explainer_model.explain_instance(text, tb_subjectivity_proba, num_features=10)
    return lime_exp.as_list()


def predict_subjectivity_with_shap(texts):
    if texts is None:
        return np.array([])
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    
    texts = [str(t) if t else "" for t in texts]
    return np.array([TextBlob(t).sentiment.subjectivity if t else 0.0 for t in texts])


#Detoxify
import numpy as np

# Detoxify label-to-index mapping
DETOXIFY_LABELS = {
    "toxicity": 0,
    "severe_toxicity": 1,
    "obscene": 2,
    "threat": 3,
    "insult": 4,
    "identity_attack": 5
}

def detoxify_predict(texts, label):
    """
    Predict detoxify scores for a list of texts.
    Returns a 1D array of scores for the specified label.
    """
    if isinstance(texts, str):
        texts = [texts]
    
    detox_model = Detoxify('original')
    results = detox_model.predict(texts)
    
    # results is a dict like {"toxicity": [0.1, 0.2, ...], "severe_toxicity": [...], ...}
    if label in results:
        scores = results[label]
        return np.array(scores) if isinstance(scores, list) else np.array([scores])
    else:
        # fallback: return zeros if label not found
        return np.zeros(len(texts))

def explain_detoxify_with_lime(text, label):
    """Explain detoxify prediction using LIME."""
    from lime.lime_text import LimeTextExplainer
    
    lime_explainer = LimeTextExplainer(class_names=[f"not_{label}", label])
    
    # Define prediction function that returns 2D array: [[prob_class_0, prob_class_1], ...]
    def predict_proba(texts):
        if isinstance(texts, str):
            texts = [texts]
        scores = detoxify_predict(texts, label)
        # Convert 1D scores to 2D probabilities: [[1-s, s], [1-s, s], ...]
        probs = np.column_stack([1 - scores, scores])
        return probs
    
    try:
        lime_exp = lime_explainer.explain_instance(
            text,
            predict_proba,
            num_features=10,
            labels=[1]  # explain class 1 (the target label)
        )
        return lime_exp.as_list()
    except Exception as e:
        print(f"LIME explanation failed for {label}: {e}")
        return []

def explain_detoxify_with_shap(text, label):
    """Explain detoxify prediction using SHAP."""
    try:
        # Wrap detoxify_predict with the label bound
        predict_fn = partial(detoxify_predict, label=label)
        
        shap_explainer_model = shap.Explainer(
            predict_fn,
            shap.maskers.Text(" ")
        )
        shap_values = shap_explainer_model([text])
        
        # Extract SHAP values for the target class
        # shap_values.values shape: (n_samples, n_tokens)
        if hasattr(shap_values, 'values'):
            return shap_values.values[0]  # return for first (and only) sample
        else:
            return np.array(shap_values[0]) if len(shap_values) > 0 else np.array([])
    except Exception as e:
        print(f"SHAP explanation failed for {label}: {e}")
        return np.array([])



import numpy as np
import json

def extract_biased_tokens(raw_data):
    """
    Extracts tokens by aggregating available XAI tools (LIME, SHAP, TI).
    Uses Min-Max scaling to ensure biased tokens are identified even 
    when absolute scores are diluted in high-density phrases.
    """
    # Load threshold from params
    with open("params.json", "r") as f:
        params = json.load(f)
    threshold = params.get("explainability_threshold", 0.6)

    token_scores = {} # Format: {token: [score1, score2, ...]}
    
    # --- 1. Extract LIME ---
    if 'lime' in raw_data and raw_data['lime']:
        for word, score in raw_data['lime']:
            token = str(word).strip().lower()
            token_scores.setdefault(token, []).append(abs(float(score)))

    # --- 2. Extract SHAP ---
    if 'shap' in raw_data and raw_data['shap'] is not None:
        try:
            shap_vals = raw_data['shap']
            # Check if it's a numpy array
            if isinstance(shap_vals, np.ndarray):
                if shap_vals.size > 0:
                    # Flatten and use as token scores
                    for i, val in enumerate(shap_vals.flatten()):
                        token = f"token_{i}"
                        token_scores.setdefault(token, []).append(abs(float(val)))
            else:
                # Try to extract from Explanation object
                if hasattr(shap_vals, 'values') and hasattr(shap_vals, 'data'):
                    shap_arr = shap_vals.values[0] if shap_vals.values.size > 0 else np.array([])
                    shap_data = shap_vals.data[0] if shap_vals.data.size > 0 else []
                    for token, val in zip(shap_data, shap_arr):
                        t_clean = str(token).strip().lower()
                        if t_clean:
                            token_scores.setdefault(t_clean, []).append(abs(float(val)))
        except (IndexError, AttributeError, TypeError, ValueError) as e:
            print(f"Error extracting SHAP: {e}")
            pass

    # --- 3. Extract Transformers Interpret ---
    if 'transformers_interpret' in raw_data and raw_data['transformers_interpret'] is not None:
        try:
            ti_data = raw_data['transformers_interpret']
            if isinstance(ti_data, (list, tuple)):
                for token, score in ti_data:
                    t_clean = str(token).strip().lower()
                    if t_clean not in ['[cls]', '[sep]', '<s>', '</s>']:
                        token_scores.setdefault(t_clean, []).append(abs(float(score)))
        except (IndexError, AttributeError, TypeError, ValueError):
            pass

    if not token_scores:
        return []

    # --- 4. Mean Aggregation ---
    mean_scores = {token: sum(scores)/len(scores) for token, scores in token_scores.items()}
    
    # --- 5. Relative Normalization ---
    max_score = max(mean_scores.values())
    min_score = min(mean_scores.values())
    
    final_targets = []
    for token, avg_score in mean_scores.items():
        if max_score > min_score:
            relative_score = (avg_score - min_score) / (max_score - min_score)
        else:
            relative_score = 1.0
        
        if relative_score >= threshold:
            final_targets.append({
                "word": token,
                "score": round(relative_score, 3),
                "abs_intensity": round(avg_score, 3),
                "tools_detected": len(token_scores[token])
            })

    return sorted(final_targets, key=lambda x: x['score'], reverse=True)

def explain_bias(bias, bias_type, phrase):
    explanations = {}
    if bias_type is None:
        return None, None, None
    if bias_type == "toxicity_bias":
        explanations[bias_type] = {}
        explanations[bias_type]["lime"] = explain_toxigen_with_lime(phrase)
        explanations[bias_type]["shap"] = explain_toxigen_with_shap(phrase)
        explanations[bias_type]["transformers_interpret"] = explain_toxigen_with_transformers_interpret(phrase)
    elif bias_type.startswith("detoxify_"):
        label = bias_type[len("detoxify_"):-len("_bias")]
        bias_type = "detoxify"+label+"_bias"
        explanations[bias_type] = {}
        explanations[bias_type]["lime"] = explain_detoxify_with_lime(phrase, label)
        explanations[bias_type]["shap"] = explain_detoxify_with_shap(phrase, label)
    elif bias_type == "subjectivity_bias":
        explanations[bias_type] = {}
        explanations[bias_type]["lime"] = explain_subjectivity_with_lime(phrase)
        explanations[bias_type]["shap"] = predict_subjectivity_with_shap(phrase)
    elif bias_type in ["anchoring", "framing", "stereotyping", "false_premises", "confirmation_seeking"]:
        explanations[bias_type] = {}
        explanations[bias_type] = bias[bias_type]["details"]
        return bias_type, bias[bias_type], bias[bias_type]["details"]

    biased_tokens = extract_biased_tokens(explanations[bias_type])

    # Construct the natural language explanation
    if len(biased_tokens) > 1:
        explanation_str = f"This term is exhibiting **{bias_type}**."
    else:
        explanation_str = f"This term is exhibiting **{bias_type}**."
    
    return bias_type, biased_tokens, explanation_str