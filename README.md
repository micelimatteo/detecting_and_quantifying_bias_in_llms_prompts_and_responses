# Master Thesis — Detecting and Quantifying Bias in LLM Prompts and Responses

Guardrail framework for detecting toxicity and cognitive bias (framing, confirmation, anchoring, stereotyping) in LLM prompts/responses, combining fast content classifiers with an LLM-based multi-agent debate, plus an explainability layer (SHAP/LIME/Transformers Interpret) and a Gradio UI.

## Layout

- `bias_detection.py`, `bias_mitigation.py`, `explainability.py`, `utils.py`, `params.json` — core detection/mitigation/explainability pipeline and its configuration.
- `GUI.py` / `CLI.py` — Gradio dashboard and command-line entry points.
- `toxigen_hatebert/` — local HateBERT (fine-tuned on ToxiGen) checkpoint used by the toxicity classifier. **Not included in this repo** downloadable from Hugging Face: https://huggingface.co/tomh/toxigen_hatebert 
- `requirements.txt` — Python dependencies.

## Datasets

Used for the formal evaluation:
- **civil_comments** (`google/civil_comments`, HF) — toxicity baseline (Test 1); also the corpus Detoxify itself was trained on.
- **RealToxicityPrompts** (`allenai/real-toxicity-prompts`, HF) — safety-leakage evaluation (Test 2).
- **MindScope** (`https://github.com/mindhunter01/MindScope`) — cognitive bias detection and the subjectivity-gate validation (Test 3).

Used earlier for tool exploration/threshold calibration (not bundled in this repo):
- **CrowS-Pairs** (`crows_pairs_anonymized.csv`) — stereotypical-bias sentence pairs.
- **WikiBias** (`WIKIBIAS_DATA/`) — binary/fine-grained Wikipedia framing-bias classification.
- **BOLD** (`bold-main/`) — open-ended generation prompts by demographic category (gender, race, profession, political/religious ideology).
- **BiasDPO** (`BiasDPO/dataset.jsonl`) — chosen/rejected preference pairs for biased vs. neutral completions.

## Notes

- The LLM backend (Gemma 3) is served locally via [Ollama](https://ollama.com); no external API key is required.
