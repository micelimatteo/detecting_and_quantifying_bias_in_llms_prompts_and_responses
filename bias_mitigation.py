from utils import generate_llm_response 

def sanitize_prompt(prompt: str, biases: Dict) -> str:
    """
    Sanitize the prompt by removing or replacing any problematic phrases
    identified in the bias detection step.
    """
    prompt = f"""This prompt {prompt} contains this list of biases: {biases}. Please rewrite the prompt to eliminate these biases while preserving its original intent"""
    sanitized_prompt = generate_llm_response(prompt)
    
    return sanitized_prompt