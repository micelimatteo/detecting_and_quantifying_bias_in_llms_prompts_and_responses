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
import json


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



def normalize_tool_scores(scores_dict):
    """
    Helper to apply Min-Max scaling to a dictionary of {token: absolute_score}.
    Ensures each tool's output is in the range [0, 1].
    """
    if not scores_dict:
        return {}
    
    vals = list(scores_dict.values())
    max_v = max(vals)
    min_v = min(vals)
    
    # Avoid division by zero if all scores are identical
    if max_v == min_v:
        return {k: 1.0 for k in scores_dict.keys()}
    
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

def extract_biased_tokens(raw_data):
    # Load threshold
    with open("params.json", "r") as f:
        params = json.load(f)
    threshold = params.get("explainability_threshold", 0.6)

    # Temporary storage for tool-specific absolute scores
    lime_raw = {}
    shap_raw = {}
    ti_raw = {}

    # --- 1. Extract LIME ---
    if 'lime' in raw_data and raw_data['lime']:
        for word, score in raw_data['lime']:
            token = str(word).strip().lower()
            lime_raw[token] = abs(float(score))

    # --- 2. Extract SHAP ---
    if 'shap' in raw_data and raw_data['shap'] is not None:
        try:
            # Simplification for your logic: extract absolute values into shap_raw
            # (Assuming standard SHAP Explanation object or array)
            shap_vals = raw_data['shap']
            if hasattr(shap_vals, 'values') and hasattr(shap_vals, 'data'):
                # Handle cases where SHAP might return multi-class arrays
                s_vals = shap_vals.values[0] if len(shap_vals.values.shape) > 1 else shap_vals.values
                s_data = shap_vals.data[0] if len(shap_vals.data.shape) > 1 else shap_vals.data
                for token, val in zip(s_data, s_vals):
                    t_clean = str(token).strip().lower()
                    if t_clean: shap_raw[t_clean] = abs(float(val))
        except Exception as e:
            print(f"SHAP extraction error: {e}")

    # --- 3. Extract Transformers Interpret ---
    if 'transformers_interpret' in raw_data and raw_data['transformers_interpret']:
        for token, score in raw_data['transformers_interpret']:
            t_clean = str(token).strip().lower()
            if t_clean not in ['[cls]', '[sep]', '<s>', '</s>']:
                ti_raw[t_clean] = abs(float(score))

    # --- 4. PRE-AGGREGATION NORMALIZATION (The Crucial Step) ---
    # This turns every tool into a [0, 1] range individually.
    lime_norm = normalize_tool_scores(lime_raw)
    shap_norm = normalize_tool_scores(shap_raw)
    ti_norm = normalize_tool_scores(ti_raw)

    # --- 5. ENSEMBLE AVERAGING ---
    # Combine normalized scores
    ensemble_scores = {}
    tools_count = {}

    for d in [lime_norm, shap_norm, ti_norm]:
        for token, norm_score in d.items():
            ensemble_scores[token] = ensemble_scores.get(token, 0) + norm_score
            tools_count[token] = tools_count.get(token, 0) + 1

    if not ensemble_scores:
        return []

    mean_scores = {t: ensemble_scores[t] / tools_count[t] for t in ensemble_scores}

    # --- 6. FINAL RELATIVE SCALING (For UI consistency) ---
    max_avg = max(mean_scores.values())
    min_avg = min(mean_scores.values())

    final_targets = []
    for token, avg_score in mean_scores.items():
        # Final stretch to ensure the 'most biased' token is 1.0
        rel_score = (avg_score - min_avg) / (max_avg - min_avg) if max_avg > min_avg else 1.0
        
        if rel_score >= threshold:
            final_targets.append({
                "word": token,
                "score": round(rel_score, 3),
                "tools_detected": tools_count[token]
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
    elif bias_type.lower() in ["anchoring", "framing", "stereotyping", "false_premises", "confirmation_seeking"]:
        # For cognitive biases, we currently do not have specific explainability methods implemented.
        return bias_type, bias[bias_type]['problematic_phrases'], bias[bias_type].get("details", "No explanation available.")
    else:
        return bias_type, None, "No explainability method implemented for this bias type."
    
    biased_tokens = extract_biased_tokens(explanations[bias_type])

    explanation_str = None
    # Construct the natural language explanation
    if len(biased_tokens) > 1:
        explanation_str = f"This term is exhibiting **{bias_type}**."
    else:
        explanation_str = f"This term is exhibiting **{bias_type}**."
    
    return bias_type, biased_tokens, explanation_str