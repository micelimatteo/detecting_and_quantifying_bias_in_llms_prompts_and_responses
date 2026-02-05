import gradio as gr
import time
import re
from bias_detection import identify_biases
from bias_mitigation import sanitize_prompt
from utils import generate_llm_response
cognitive_biases = ["anchoring", "framing", "stereotyping", "false_premises", "confirmation_seeking"]
# ==========================================
# 2. HELPER: TEXT PROCESSING FOR UI
# ==========================================

def format_highlight_data(text, biased_tokens, bias_type):
    print(biased_tokens)
    print(f"Formatting highlight data for bias type: {bias_type} with tokens: {biased_tokens['problematic_phrases'] if bias_type in cognitive_biases else biased_tokens}")

    """
    Converts text and token list into Gradio HighlightedText format:
    [(substring, label), (substring, None), ...]
    """
    if not biased_tokens:
        return [(text, None)]
    
    

    # Sort tokens by their position in text to handle them in order
    # Note: A real implementation needs start/end indices. 
    # Here we simply split by space for the demo.
    words = text.split(" ")
    highlighted_output = []
    if bias_type in cognitive_biases:
        biased_words_map = {t.lower(): t for p in biased_tokens['problematic_phrases'] for t in p.split(" ")}
    else:   
        biased_words_map = {t['word'].lower(): t for t in biased_tokens}
    for word in words:
        clean_word = word.strip(".,!?").lower()
        if clean_word in biased_words_map:
            if bias_type in cognitive_biases:
                score_label = biased_tokens['severity']
                highlighted_output.append((word + " ", f"Score: {score_label}"))
            else:
                # It's a biased word, mark it with the bias score
                score_label = str(biased_words_map[clean_word]['abs_intensity'])
                highlighted_output.append((word + " ", f"Score: {score_label}"))
        else:
            # Normal word
            highlighted_output.append((word + " ", None))   
    return highlighted_output

# ==========================================
# 3. GRADIO UI LOGIC
# ==========================================

# Custom theme for Material Design look
theme = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Roboto"), "ui-sans-serif", "system-ui"]
)

with gr.Blocks(title="Bias Guardrail") as demo:
    
    # State variables to hold context
    state_biased_tokens = gr.State([])
    state_current_prompt = gr.State("")
    state_mitigated_prompt = gr.State("")
    state_bias_type = gr.State("")  # NEW
    state_global_expl = gr.State("")  # NEW

    with gr.Row():
        gr.Markdown("## 🛡️ LLM Guardrail & Bias Mitigation Dashboard")

    with gr.Row(equal_height=True):
        
        # --- LEFT COLUMN: Chat Interface ---
        with gr.Column(scale=2, min_width=400):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                avatar_images=(None, "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg")
            )
            msg = gr.Textbox(
                label="Your Prompt",
                placeholder="Type something...",
                lines=2
            )
            with gr.Row():
                clear = gr.Button("Clear Chat", variant="secondary")
                submit = gr.Button("Send", variant="primary")

        # --- RIGHT COLUMN: Bias Analysis Panel (Hidden by default) ---
        with gr.Column(scale=1, visible=False) as guardrail_panel:
            gr.Markdown("### ⚠️ Bias Detected")
            
            # This component shows the prompt with red highlights
            bias_visualizer = gr.HighlightedText(
                label="Analyzed Prompt",
                combine_adjacent=True,
                show_legend=False,
                color_map={"Score": "red"}
            )
            
            with gr.Group():
                gr.Markdown("**Bias Detail** (Click a highlighted word)")
                # Details populated when user clicks a word
                selected_word_info = gr.JSON(label="Token Metrics", value={})
                explanation_text = gr.Markdown("Select a token to see the explanation.")
            
            gr.HTML("<hr>")
            
            # Mitigation Section
            with gr.Group():
                gr.Markdown("### 🛠️ Mitigation Strategy")
                suggestion_box = gr.Textbox(label="Suggested Rewrite", interactive=False)
                
                with gr.Row():
                    cancel_btn = gr.Button("Ignore", variant="secondary")
                    accept_btn = gr.Button("✅ Accept Correction", variant="primary")

    # ==========================================
    # 4. EVENT HANDLERS
    # ==========================================

    def on_submit(user_input, history):
        """
        1. User submits -> Run Identity Bias
        2. If Bias -> Open Panel, don't generate LLM response yet (or generate with warning)
        3. If No Bias -> Generate LLM response normally
        """
        if not user_input:
            return history, gr.update(visible=False), [], "", "", {}, "", ""

        # 1. Check for bias
        bias_type, tokens, global_expl = identify_biases(user_input, is_prompt=True)
        print(bias_type, tokens, global_expl)
        # Append user message to history
        new_history = history + [{"role": "user", "content": user_input}]

        if bias_type:
            # BIAS FOUND: Trigger the guardrail UI
            highlighted_data = format_highlight_data(user_input, tokens, bias_type)
            suggested_text = sanitize_prompt(user_input, tokens, bias_type)
            llm_response = generate_llm_response(user_input)
            new_history.append({"role": "assistant", "content": llm_response})
            
            return (
                new_history,
                gr.update(visible=True),
                tokens,
                user_input,
                suggested_text,
                highlighted_data,
                {},
                "⚠️ **Bias Detected:** Click highlighted words for details.",
                suggested_text,
                bias_type,          # NEW: save bias_type
                global_expl         # NEW: save global explanation
            )
        else:
            # NO BIAS: Normal Flow
            llm_response = generate_llm_response(user_input)
            new_history.append({"role": "assistant", "content": llm_response})
            
            return (
                new_history, 
                gr.update(visible=False), 
                [], "", "", [], {}, "", "",
                "",                 # NEW: clear bias_type
                ""                  # NEW: clear global_expl
            )

    def on_select_token(evt: gr.SelectData, current_tokens, bias_type, global_expl):
        """Updates details when clicking highlighted text"""
        # evt.value is a list: [text, label]
        raw_value = evt.value[0] if isinstance(evt.value, list) else evt.value
        selected_word = raw_value.strip(".,!?;: ").lower()
        
        # Handle cognitive biases (plain string list in 'problematic_phrases')
        if bias_type in cognitive_biases:
            # current_tokens is a dict with 'problematic_phrases' key
            phrases = current_tokens.get('problematic_phrases', [])
            found = next((p for p in phrases if p.lower() == selected_word), None)
            if found:
                # Return (json_dict, markdown_str)
                return (
                    {"term": found, "type": "Cognitive Bias", "severity": current_tokens.get('severity', 'N/A')},
                    f"**Term:** {found}\n\n**Type:** Cognitive Bias\n\n**Severity:** {current_tokens.get('severity', 'N/A')}\n\n**Details:** {global_expl}"
                )
            return ({}, "No details found.")
        else:
            # Toxicity/detoxify: current_tokens is a list of dicts with 'word' and 'score'
            found = next((t for t in current_tokens if t['word'].lower() == selected_word), None)
            if found:
                return (
                    {"word": found['word'], "score": found.get('score', 'N/A')},
                    f"**Word:** {found['word']}\n\n**Score:** {found.get('score', 'N/A')}\n\n**Reason:** {global_expl}"
                )
            return ({}, "No details found.")

    def on_select_token_wrapper(evt: gr.SelectData, current_tokens, bias_type, global_expl):
        """Wrapper to handle select event + state inputs"""
        if evt is None or not hasattr(evt, 'value'):
            return ({}, "No token selected.")
        
        raw_value = evt.value[0] if isinstance(evt.value, list) else evt.value
        selected_word = raw_value.strip(".,!?;: ").lower()
        
        # Handle cognitive biases
        if bias_type in cognitive_biases:
            phrases = current_tokens.get('problematic_phrases', [])
            found = next((p for p in phrases if p.lower() == selected_word), None)
            if found:
                return (
                    {"term": found, "type": "Cognitive Bias", "severity": current_tokens.get('severity', 'N/A')},
                    f"**Term:** {found}\n\n**Type:** Cognitive Bias\n\n**Severity:** {current_tokens.get('severity', 'N/A')}\n\n**Details:** {global_expl}"
                )
            return ({}, "No details found.")
        else:
            # Toxicity/detoxify
            found = next((t for t in current_tokens if t['word'].lower() == selected_word), None)
            if found:
                return (
                    {"word": found['word'], "score": found.get('score', 'N/A')},
                    f"**Word:** {found['word']}\n\n**Score:** {found.get('score', 'N/A')}\n\n**Reason:** {global_expl}"
                )
            return ({}, "No details found.")

    def on_accept_mitigation(suggested_prompt, history):
        """
        User accepts the fix.
        1. Replace last user message in history with new prompt.
        2. Regenerate LLM response.
        3. Hide panel.
        """
        # Remove the last exchange (old biased prompt and old response)
        # Note: In a real app, you might want to keep a record, but here we "Fix" it.
        history_pruned = history[:-2] 
        
        # Add corrected prompt
        history_pruned.append({"role": "user", "content": suggested_prompt})
        
        # Generate new response
        new_response = generate_llm_response(suggested_prompt)
        history_pruned.append({"role": "assistant", "content": new_response})
        
        return history_pruned, gr.update(visible=False)

    def on_ignore():
        return gr.update(visible=False)

    # Wire up the inputs/outputs
    submit.click(
        on_submit,
        inputs=[msg, chatbot],
        outputs=[
            chatbot, 
            guardrail_panel, 
            state_biased_tokens, 
            state_current_prompt, 
            state_mitigated_prompt,
            bias_visualizer,
            selected_word_info,
            explanation_text,
            suggestion_box,
            state_bias_type,        # NEW
            state_global_expl       # NEW
        ]
    ).then(lambda: "", None, msg)
    
    # Enter key submission
    msg.submit(
        on_submit,
        inputs=[msg, chatbot],
        outputs=[chatbot, guardrail_panel, state_biased_tokens, state_current_prompt, state_mitigated_prompt, bias_visualizer, selected_word_info, explanation_text, suggestion_box, state_bias_type, state_global_expl]
    ).then(lambda: "", None, msg)

    # Click on Highlighted Text
    bias_visualizer.select(
        on_select_token_wrapper,
        inputs=[state_biased_tokens, state_bias_type, state_global_expl],
        outputs=[selected_word_info, explanation_text]
    )

    # Accept Correction Button
    accept_btn.click(
        on_accept_mitigation,
        inputs=[state_mitigated_prompt, chatbot],
        outputs=[chatbot, guardrail_panel]
    )
    
    # Ignore Button
    cancel_btn.click(on_ignore, inputs=None, outputs=guardrail_panel)

# Launch
if __name__ == "__main__":
    demo.launch(theme=theme)