import gradio as gr
import time
import re
from bias_detection import identify_biases
from bias_mitigation import sanitize_prompt
from utils import generate_llm_response

cognitive_biases = ["anchoring", "framing", "stereotyping", "confirmation_seeking"]

# ==========================================
# 2. HELPER: TEXT PROCESSING FOR UI
# ==========================================

def format_highlight_data(text, biased_tokens, bias_type):

    print(f"Formatting highlight data for bias type: {bias_type} with tokens: {biased_tokens}\n")

    """
    Converts text and token list into Gradio HighlightedText format:
    [(substring, "Score"), (substring, None), ...]

    A constant "Score" label is used for every flagged word — matching
    color_map={"Score": "red"} on the HighlightedText component — rather
    than the per-token numeric score. Gradio's color_map keys on the exact
    label string, so emitting a different label per token (e.g. "0.823")
    never matched "Score" and rendered with no highlight color at all.
    The actual score is still available on click via state_biased_tokens
    in on_select_token_wrapper, so nothing is lost by not embedding it here.
    """

    if not biased_tokens:
        return [(text, None)]

    words = text.split(" ")
    highlighted_output = []

    def normalize(w):
        return w.strip(".,!?;:\"'").lower()

    if bias_type in cognitive_biases:
        # explain_bias() returns the problematic phrases directly as a
        # list[str] for cognitive biases (explainability.py:293) — not a
        # dict with a 'problematic_phrases' key. The previous isinstance(...,
        # dict) check was always False, so biased_words_map was always {}
        # and cognitive-bias highlighting never matched anything.
        phrases = biased_tokens if isinstance(biased_tokens, list) else []
        biased_words_map = {
            normalize(w): True
            for phrase in phrases for w in str(phrase).split(" ") if normalize(w)
        }
    else:
        # Toxicity/detoxify: biased_tokens is a list of dicts with a 'word' key
        biased_words_map = {
            normalize(t['word']): t for t in biased_tokens if isinstance(t, dict) and 'word' in t
        }

    for word in words:
        if normalize(word) in biased_words_map:
            highlighted_output.append((word + " ", "Score"))
        else:
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
        gr.Markdown("## 🛡️ Bias Identification and Mitigation Dashboard")

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
                show_label = False,
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
                gr.Markdown("### 🛠️ Suggested Correction")
                # 1. CHANGE: Set interactive=True so the user can type in it
                suggestion_box = gr.Textbox(
                    label="Suggested Rewrite (Editable)", 
                    interactive=True,
                    placeholder="Modify the suggested rewrite here..."
                )
    
                with gr.Row():
                    cancel_btn = gr.Button("Ignore", variant="secondary")
                    # 2. CHANGE: Rename for clarity
                    apply_btn = gr.Button("✅ Send Prompt", variant="primary")

    # ==========================================
    # 4. EVENT HANDLERS
    # ==========================================

    def on_submit(user_input, history):
        """
        1. User submits -> Run Identify Bias
        2. If Bias -> Open Panel, don't generate LLM response yet (or generate with warning)
        3. If No Bias -> Generate LLM response normally
        """
        if not user_input:
            return history, gr.update(visible=False), [], "", "", [], {}, "", "", "", ""

        # 1. Check for bias
        result = identify_biases(user_input, is_prompt=True)
        
        # Handle None or all-None tuple returns
        if result is None or (isinstance(result, tuple) and all(v is None for v in result)):
            bias_type, tokens, global_expl = None, [], ""
        else:
            bias_type, tokens, global_expl = result
        
        print(bias_type, tokens, global_expl)
        
        # Append user message to history
        new_history = history + [{"role": "user", "content": user_input}]

        if bias_type:
            # BIAS FOUND: Trigger the guardrail UI
            highlighted_data = format_highlight_data(user_input, tokens, bias_type, )
            suggested_text = sanitize_prompt(user_input, bias_type , tokens)
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
        

    def send_edited_correction(edited_text, history):
        """
        Takes the manually edited text from the suggestion box,
        removes the 'bad' turn, and submits the new version.
        """
        # Remove the last user prompt and the last assistant warning
        # history looks like: [... {user}, {assistant_with_alert}]
        new_history = history[:-2] if len(history) >= 2 else []
        
        # We call your process_prompt logic again with the NEW text
        # This ensures the new prompt is also audited for safety
        return process_prompt(edited_text, new_history)
    
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
            # For cognitive biases, just show the global explanation
            # No need to match individual words
            return (
                {
                    "type": "Cognitive Bias", 
                    "bias_type": bias_type,
                    "severity": current_tokens.get('severity', 'N/A') if isinstance(current_tokens, dict) else 'N/A'
                },
                f"**Bias Type:** {bias_type.replace('_', ' ').title()}\n\n**Details:** {global_expl}"
            )
        else:
            # Toxicity/detoxify: current_tokens is a list of dicts with 'word' and 'score'
            if isinstance(current_tokens, list):
                found = next((t for t in current_tokens if isinstance(t, dict) and t.get('word', '').lower() == selected_word), None)
                if found:
                    return (
                        {"word": found.get('word', ''), "score": found.get('score', found.get('abs_intensity', 'N/A'))},
                        f"**Word:** {found.get('word', '')}\n\n**Score:** {found.get('score', found.get('abs_intensity', 'N/A'))}\n\n**Reason:** {global_expl}"
                    )
            return ({}, "No details found.")

    def on_accept_mitigation(suggested_prompt, history):
        """
        User accepts the fix.
        1. Keep the original biased exchange in history (for context)
        2. Add the corrected prompt as a new turn
        3. Generate new response
        4. Hide panel.
        """
        # Keep the full history — don't remove anything
        
        # Add corrected prompt with a marker
        history.append({"role": "user", "content": f"✅ **[Corrected]** {suggested_prompt}"})
        
        # Generate new response
        new_response = generate_llm_response(suggested_prompt)
        history.append({"role": "assistant", "content": new_response})
        
        return history, gr.update(visible=False)

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
            state_bias_type,    
            state_global_expl      
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

    # Accept Correction Button or modified prompt submission
    apply_btn.click(
        on_accept_mitigation,
        inputs=[suggestion_box, chatbot],
        outputs=[chatbot, guardrail_panel]        
    )
    # Ignore Button
    cancel_btn.click(on_ignore, inputs=None, outputs=guardrail_panel)

# Launch
if __name__ == "__main__":
    demo.launch(theme=theme, share = True)