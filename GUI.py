import tkinter as tk
from tkinter import ttk, scrolledtext, font as tkfont
import json
import re
from typing import Dict, List, Tuple
import threading

class ChatMessage(ttk.Frame):
    """A single chat message bubble"""
    def __init__(self, parent, text, is_user=True, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.columnconfigure(0, weight=1)
        
        # Message bubble
        bg_color = '#1018eb' if is_user else '#360180'
        self.text_widget = tk.Text(
            self, wrap=tk.WORD, height=1, bg=bg_color, 
            relief=tk.FLAT, padx=10, pady=10, font=('Segoe UI', 10)
        )
        self.text_widget.insert('1.0', text)
        self.text_widget.config(state='disabled')
        self.text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=20 if is_user else 0)
        
        # Auto-adjust height
        self.text_widget.update_idletasks()
        lines = int(self.text_widget.index('end-1c').split('.')[0])
        self.text_widget.config(height=min(lines, 20))


class EditableHighlightedPrompt(ttk.Frame):
    """Editable prompt with highlighted problematic sections"""
    def __init__(self, parent, original_text, biases, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.original_text = original_text
        self.biases = biases
        
        self.columnconfigure(0, weight=1)
        
        # Header
        header = ttk.Label(self, text="⚠️ Issues detected in your prompt. Edit the highlighted parts:",
                          font=('Segoe UI', 10, 'bold'))
        header.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Editable text with highlights
        self.text_widget = tk.Text(
            self, wrap=tk.WORD, height=6, bg='white', 
            relief=tk.SOLID, borderwidth=1, padx=10, pady=10, 
            font=('Segoe UI', 10)
        )
        self.text_widget.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Configure tags for highlighting
        self.text_widget.tag_config('highlight', background='#ffebee', foreground='#c62828')
        self.text_widget.tag_config('tooltip', background='#fff9c4')
        
        # Insert text with highlights
        self._insert_highlighted_text()
        
        # Issues list
        issues_frame = ttk.Frame(self)
        issues_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        for bias_name, info in biases.items():
            issue_text = f"• {bias_name.upper()}: {info['details']}"
            if 'problematic_phrases' in info and info['problematic_phrases']:
                issue_text += f" [{', '.join(info['problematic_phrases'])}]"
            
            issue_label = ttk.Label(issues_frame, text=issue_text, 
                                   foreground='#d32f2f', font=('Segoe UI', 9))
            issue_label.pack(anchor=tk.W, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        
        ttk.Button(button_frame, text="✓ Accept Changes", 
                  command=self.accept_changes).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="↻ Revert", 
                  command=self.revert).pack(side=tk.LEFT, padx=(0, 5))
        
        self.accept_callback = None
    
    def _insert_highlighted_text(self):
        """Insert text and highlight problematic phrases"""
        self.text_widget.delete('1.0', tk.END)
        
        # Collect all problematic phrases
        problematic_phrases = []
        for bias_name, info in self.biases.items():
            if 'problematic_phrases' in info:
                for phrase in info['problematic_phrases']:
                    problematic_phrases.append((phrase, bias_name))
        
        # Insert text with highlights
        current_pos = 0
        text_lower = self.original_text.lower()
        
        for phrase, bias_name in problematic_phrases:
            phrase_lower = phrase.lower()
            
            # Find all occurrences
            start = 0
            while True:
                idx = text_lower.find(phrase_lower, start)
                if idx == -1:
                    break
                
                # Insert text before phrase
                if idx > current_pos:
                    self.text_widget.insert(tk.END, self.original_text[current_pos:idx])
                
                # Insert and highlight phrase
                phrase_start = self.text_widget.index(tk.END)
                self.text_widget.insert(tk.END, self.original_text[idx:idx+len(phrase)])
                phrase_end = self.text_widget.index(tk.END)
                self.text_widget.tag_add('highlight', phrase_start, phrase_end)
                
                current_pos = idx + len(phrase)
                start = idx + 1
        
        # Insert remaining text
        if current_pos < len(self.original_text):
            self.text_widget.insert(tk.END, self.original_text[current_pos:])
    
    def get_edited_text(self):
        return self.text_widget.get('1.0', 'end-1c')
    
    def accept_changes(self):
        if self.accept_callback:
            self.accept_callback(self.get_edited_text())
    
    def revert(self):
        self.text_widget.delete('1.0', tk.END)
        self._insert_highlighted_text()
    
    def set_accept_callback(self, callback):
        self.accept_callback = callback


class BiResponseDisplay(ttk.Frame):
    """Display both original and rewritten responses side by side"""
    def __init__(self, parent, original, rewritten, biases, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        
        # Header
        header = ttk.Label(self, text="⚠️ Response contains biases. Showing original and corrected versions:",
                          font=('Segoe UI', 10, 'bold'))
        header.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Original response
        original_label = ttk.Label(self, text="Original Response (with biases)", 
                                  font=('Segoe UI', 9, 'bold'))
        original_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        
        self.original_text = tk.Text(
            self, wrap=tk.WORD, height=10, bg='#ffebee', 
            relief=tk.SOLID, borderwidth=1, padx=10, pady=10, 
            font=('Segoe UI', 10), state='disabled'
        )
        self.original_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Rewritten response
        rewritten_label = ttk.Label(self, text="✓ Corrected Response", 
                                   font=('Segoe UI', 9, 'bold'), foreground='#2e7d32')
        rewritten_label.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        self.rewritten_text = tk.Text(
            self, wrap=tk.WORD, height=10, bg='#e8f5e9', 
            relief=tk.SOLID, borderwidth=1, padx=10, pady=10, 
            font=('Segoe UI', 10), state='disabled'
        )
        self.rewritten_text.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Insert text
        self.original_text.config(state='normal')
        self.original_text.insert('1.0', original)
        self.original_text.config(state='disabled')
        
        self.rewritten_text.config(state='normal')
        self.rewritten_text.insert('1.0', rewritten)
        self.rewritten_text.config(state='disabled')
        
        # Detected biases info
        biases_frame = ttk.Frame(self)
        biases_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        biases_label = ttk.Label(biases_frame, text="Detected biases:", 
                                font=('Segoe UI', 9, 'bold'))
        biases_label.pack(anchor=tk.W)
        
        for bias_name, info in biases.items():
            bias_text = f"• {bias_name.upper()} (severity: {info['severity']:.1f}): {info['details']}"
            ttk.Label(biases_frame, text=bias_text, foreground='#d32f2f', 
                     font=('Segoe UI', 9)).pack(anchor=tk.W, pady=2)


class CrossAttributionDisplay(ttk.Frame):
    """Show bias attribution with red highlighting on problematic prompt parts"""
    def __init__(self, parent, prompt, prompt_biases, response_biases, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.columnconfigure(0, weight=1)
        
        # Header
        header = ttk.Label(self, text="🔍 Bias Cross-Attribution Analysis",
                          font=('Segoe UI', 11, 'bold'))
        header.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Show prompt with red highlighting for biases that propagated
        prompt_label = ttk.Label(self, text="Your prompt (problematic parts in red):", 
                                font=('Segoe UI', 9, 'bold'))
        prompt_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        self.prompt_text = tk.Text(
            self, wrap=tk.WORD, height=4, bg='white', 
            relief=tk.SOLID, borderwidth=1, padx=10, pady=10, 
            font=('Segoe UI', 10), state='disabled'
        )
        self.prompt_text.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Configure red highlight tag
        self.prompt_text.tag_config('propagated', background='#ffcdd2', foreground='#c62828', 
                                   font=('Segoe UI', 10, 'bold'))
        
        # Insert prompt with highlighting
        self._highlight_propagated_biases(prompt, prompt_biases, response_biases)
        
        # Attribution explanation
        explanation_frame = ttk.Frame(self)
        explanation_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(explanation_frame, text="Attribution:", 
                 font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
        
        # Match biases between prompt and response
        for bias_name, response_info in response_biases.items():
            if bias_name in prompt_biases:
                # Bias propagated from prompt
                attr_text = f"🔴 {bias_name.upper()}: Present in BOTH prompt and response (propagated)"
                color = '#c62828'
            else:
                # Bias only in response
                attr_text = f"🟡 {bias_name.upper()}: Only in response (generated by LLM)"
                color = '#f57c00'
            
            label = ttk.Label(explanation_frame, text=attr_text, 
                            foreground=color, font=('Segoe UI', 9))
            label.pack(anchor=tk.W, pady=3)
    
    def _highlight_propagated_biases(self, prompt, prompt_biases, response_biases):
        """Highlight parts of prompt that caused biases in response"""
        self.prompt_text.config(state='normal')
        self.prompt_text.delete('1.0', tk.END)
        
        # Find biases that propagated
        propagated = set(prompt_biases.keys()) & set(response_biases.keys())
        
        if not propagated:
            self.prompt_text.insert('1.0', prompt)
            self.prompt_text.config(state='disabled')
            return
        
        # Collect phrases from propagated biases
        problematic_phrases = []
        for bias_name in propagated:
            if 'problematic_phrases' in prompt_biases[bias_name]:
                for phrase in prompt_biases[bias_name]['problematic_phrases']:
                    problematic_phrases.append(phrase)
        
        # Insert with highlighting
        current_pos = 0
        text_lower = prompt.lower()
        
        for phrase in problematic_phrases:
            phrase_lower = phrase.lower()
            idx = text_lower.find(phrase_lower, current_pos)
            
            if idx != -1:
                # Text before phrase
                if idx > current_pos:
                    self.prompt_text.insert(tk.END, prompt[current_pos:idx])
                
                # Highlighted phrase
                phrase_start = self.prompt_text.index(tk.END)
                self.prompt_text.insert(tk.END, prompt[idx:idx+len(phrase)])
                phrase_end = self.prompt_text.index(tk.END)
                self.prompt_text.tag_add('propagated', phrase_start, phrase_end)
                
                current_pos = idx + len(phrase)
        
        # Remaining text
        if current_pos < len(prompt):
            self.prompt_text.insert(tk.END, prompt[current_pos:])
        
        self.prompt_text.config(state='disabled')


class GuardrailGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Guardrail System")
        self.root.geometry("900x700")
        
        # State
        self.conversation_history = []
        self.current_prompt = ""
        self.prompt_biases = {}
        self.response_biases = {}
        self.auto_correct = tk.BooleanVar(value=False)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chat area (scrollable)
        chat_container = ttk.Frame(main_frame)
        chat_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(chat_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas for scrolling
        self.chat_canvas = tk.Canvas(chat_container, yscrollcommand=scrollbar.set, 
                                     bg='white', highlightthickness=0)
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.chat_canvas.yview)
        
        # Frame inside canvas
        self.chat_frame = ttk.Frame(self.chat_canvas)
        self.chat_window = self.chat_canvas.create_window((0, 0), window=self.chat_frame, 
                                                          anchor=tk.NW)
        
        # Configure scrolling
        self.chat_frame.bind('<Configure>', self._on_frame_configure)
        self.chat_canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Input area
        input_frame = ttk.Frame(main_frame, padding="10")
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Auto-correct checkbox
        ttk.Checkbutton(input_frame, text="Auto-correct biases", 
                       variable=self.auto_correct).pack(side=tk.LEFT, padx=(0, 10))
        
        # Input box
        self.input_text = tk.Text(input_frame, height=3, wrap=tk.WORD, 
                                 font=('Segoe UI', 10), relief=tk.SOLID, borderwidth=1)
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Send button
        send_btn = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_btn.pack(side=tk.LEFT)
        
        # Bind Enter key
        self.input_text.bind('<Return>', lambda e: self.send_message() if not e.state & 1 else None)
    
    def _on_frame_configure(self, event=None):
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox('all'))
    
    def _on_canvas_configure(self, event):
        canvas_width = event.width
        self.chat_canvas.itemconfig(self.chat_window, width=canvas_width)
    
    def send_message(self):
        prompt = self.input_text.get('1.0', 'end-1c').strip()
        if not prompt:
            return
        
        self.current_prompt = prompt
        self.input_text.delete('1.0', tk.END)
        
        # Add user message
        self.add_message(prompt, is_user=True)
        
        # Analyze prompt in background
        threading.Thread(target=self._analyze_and_respond, daemon=True).start()
    
    def _analyze_and_respond(self):
        # Detect biases in prompt
        self.prompt_biases = self._detect_biases(self.current_prompt, is_prompt=True)
        
        if self.prompt_biases:
            # Has issues
            if self.auto_correct.get():
                # Auto-correct mode
                self.root.after(0, self._show_auto_corrected)
            else:
                # Manual edit mode
                self.root.after(0, self._show_editable_prompt)
        else:
            # No issues, proceed directly
            self._generate_response(self.current_prompt)
    
    def _show_editable_prompt(self):
        """Show editable highlighted prompt"""
        edit_widget = EditableHighlightedPrompt(
            self.chat_frame, self.current_prompt, self.prompt_biases
        )
        edit_widget.pack(fill=tk.X, pady=5)
        
        def on_accept(edited_text):
            edit_widget.destroy()
            self.current_prompt = edited_text
            threading.Thread(target=self._generate_response, 
                           args=(edited_text,), daemon=True).start()
        
        edit_widget.set_accept_callback(on_accept)
        self._scroll_to_bottom()
    
    def _show_auto_corrected(self):
        """Show auto-corrected version"""
        corrected = self._rewrite_prompt(self.current_prompt, self.prompt_biases)
        
        # Show correction message
        correction_text = f"✓ Auto-corrected prompt:\n{corrected}"
        self.add_message(correction_text, is_user=False)
        
        self.current_prompt = corrected
        threading.Thread(target=self._generate_response, 
                        args=(corrected,), daemon=True).start()
    
    
    def _show_dual_response(self, original, rewritten):
        """Show both original and rewritten responses"""
        dual_widget = BiResponseDisplay(
            self.chat_frame, original, rewritten, self.response_biases
        )
        dual_widget.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Show cross-attribution
        if self.prompt_biases or self.response_biases:
            attr_widget = CrossAttributionDisplay(
                self.chat_frame, self.current_prompt, 
                self.prompt_biases, self.response_biases
            )
            attr_widget.pack(fill=tk.X, pady=10)
        
        self._scroll_to_bottom()
    
    def add_message(self, text, is_user):
        msg = ChatMessage(self.chat_frame, text, is_user)
        msg.pack(fill=tk.X, pady=5, anchor=tk.E if is_user else tk.W)
        self._scroll_to_bottom()
    
    def _scroll_to_bottom(self):
        self.root.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)
    
    
    # === LLM FUNCTIONS ===
    def _detect_biases(self, text: str, is_prompt: bool) -> Dict:
        biases = {}

        return biases
    
    def _rewrite_prompt(self, prompt: str, biases: Dict) -> str:
        rewritten = prompt
        return rewritten
    
    def _call_llm(self, prompt: str) -> str:
        response = "This is a mock response from the LLM."
        return response
    
    def _sanitize_response(self, response: str, biases: Dict) -> str:
        sanitized_response = response
        return sanitized_response


def main():
    root = tk.Tk()
    app = GuardrailGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()