from utils import generate_llm_response 

def sanitize_prompt(prompt: str, biases: str, tokens) -> str:
    """
    Sanitize the prompt by removing or replacing any problematic phrases
    identified in the bias detection step.
    """
    prompt = f"""This prompt {prompt} contains this list of biases: {biases}. Please rewrite the prompt to eliminate these biases while preserving its original intent. Respond only in the format of 'Rewritten Prompt: <your rewritten prompt here>'. Do not write any additional commentary or explanation."""
    
    response = generate_llm_response(prompt)
    rewritten_prefix = "Rewritten Prompt:"
    if rewritten_prefix in response:
        sanitized_prompt = response.split(rewritten_prefix)[-1].strip()    
    return sanitized_prompt

def mitigate_bias(response: str, biases: dict, details) -> str:
    print(f"Mitigating biases: {biases} ")
    """
    Mitigate biases in the LLM response by rephrasing or adjusting
    the content to ensure fairness and neutrality.
    """
    response = f"""You responded: '{response}' that contains this list of biases: {biases}. The problematic part of the phrase is {details} Please rewrite the response to eliminate these biases while preserving its original intent"""
    mitigated_response = generate_llm_response(response)
    
    return mitigated_response