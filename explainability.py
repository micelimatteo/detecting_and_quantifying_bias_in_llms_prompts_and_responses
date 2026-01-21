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


#HateBERT (Toxigen)
def toxigen_proba(texts):
    model_name = "tomh/toxigen_hatebert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Ensure texts is a list of strings
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, (list, tuple)):
        texts = [str(t) if not isinstance(t, str) else t for t in texts]
    else:
        texts = [str(texts)]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()
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
    return shap_values

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
def detoxify_predict(texts, label):
    # Detoxify expects a list of strings and returns a dict of scores
    detox_model = Detoxify('original')
    return [detox_model.predict(t)[label] for t in texts]

def explain_detoxify_with_lime(text, label):
    detox_model = Detoxify('original')
    lime_explainer_model = LimeTextExplainer(class_names=[f"not_{label}", label])
    lime_exp = lime_explainer_model.explain_instance(text, lambda x: [[1 - s, s] for s in detoxify_predict(x, label)], num_features=10)
    return lime_exp.as_list()

def explain_detoxify_with_shap(text, label):
    detox_model = Detoxify('original')
    shap_explainer_model = shap.Explainer(detoxify_predict, shap.maskers.Text(" "))
    shap_values = shap_explainer_model([text])
    return shap_values


def explain_bias(bias, type, phrase):
    explanations = {}
    if type == "toxicity_bias":
        explanations["lime"] = explain_toxigen_with_lime(phrase)
        explanations["shap"] = explain_toxigen_with_shap(phrase)
        explanations["transformers_interpret"] = explain_toxigen_with_transformers_interpret(phrase)
    elif type.startswith("detoxify_"):
        label = type[len("detoxify_"):-len("_bias")]
        explanations["lime"] = explain_detoxify_with_lime(phrase, label)
        explanations["shap"] = explain_detoxify_with_shap(phrase, label)
    elif type == "subjectivity_bias":
        explanations["lime"] = explain_subjectivity_with_lime(phrase)
        explanations["shap"] = predict_subjectivity_with_shap(phrase)
    elif type in ["anchoring", "framing", "stereotyping", "false_premises", "confirmation_seeking"]:
        # Placeholder for cognitive bias explanations
        explanations[type] = bias[type]["details"]
    return explanations