"""
Guardrail Analyzer Module
Detects toxicity, cognitive bias, and subjectivity in prompts
"""

import re
from typing import List, Dict, Tuple
from detoxify import Detoxify
import torch


class GuardrailAnalyzer:
    def __init__(self):
        # Initialize the toxicity detection model
        self.toxicity_model = Detoxify('original')

        # Bias patterns with explanations
        self.bias_patterns = {
            'confirmation_bias': {
                'patterns': [
                    r'\b(obviously|clearly|everyone knows|it\'s obvious that)\b',
                    r'\b(prove(s)? (me|us) right|as (I|we) (always )?said)\b',
                ],
                'description': 'Confirmation Bias: Seeking information that confirms existing beliefs',
                'suggestion': 'Consider rephrasing to remain open to different perspectives'
            },
            'anchoring_bias': {
                'patterns': [
                    r'\b(first|initial(ly)?).{0,50}(better|best|correct|right)\b',
                    r'\b(starting (from|with)|based on (my )?first)\b',
                ],
                'description': 'Anchoring Bias: Over-relying on initial information',
                'suggestion': 'Consider multiple perspectives beyond the first information encountered'
            },
            'groupthink': {
                'patterns': [
                    r'\b(everyone (thinks|believes|agrees)|we all know)\b',
                    r'\b(nobody (thinks|believes)|no one disagrees)\b',
                ],
                'description': 'Groupthink: Assuming consensus without evidence',
                'suggestion': 'Acknowledge potential diversity of opinions'
            },
            'stereotyping': {
                'patterns': [
                    r'\b(all|every) (men|women|people|they|them) (are|always)\b',
                    r'\b(typical|usually) (men|women|people|they) (are|do)\b',
                ],
                'description': 'Stereotyping: Making generalizations about groups',
                'suggestion': 'Avoid broad generalizations; recognize individual differences'
            },
            'hasty_generalization': {
                'patterns': [
                    r'\b(always|never|all|every|none) (of )?(them|these|those)\b',
                    r'\b(in all cases|without exception|every single (time|one))\b',
                ],
                'description': 'Hasty Generalization: Drawing broad conclusions from limited evidence',
                'suggestion': 'Use qualifiers like "some", "often", or "in many cases"'
            },
        }

        # Subjectivity indicators
        self.subjectivity_patterns = {
            'opinion_markers': {
                'patterns': [
                    r'\b(I think|I believe|I feel|in my opinion|personally)\b',
                    r'\b(seems (to me )?like|appears to be|looks like)\b',
                ],
                'description': 'Opinion Marker: Subjective personal view',
                'suggestion': 'Consider distinguishing between opinion and fact'
            },
            'absolute_language': {
                'patterns': [
                    r'\b(best|worst|perfect|terrible|amazing|awful|horrible)\b',
                    r'\b(most (important|significant|crucial|essential))\b',
                ],
                'description': 'Absolute Language: Strong subjective judgment',
                'suggestion': 'Consider using more measured language'
            },
            'emotional_language': {
                'patterns': [
                    r'\b(love|hate|disgusting|beautiful|ugly|fantastic)\b',
                    r'\b(outrageous|ridiculous|absurd|brilliant|stupid)\b',
                ],
                'description': 'Emotional Language: Emotionally charged terms',
                'suggestion': 'Consider using more neutral, objective language'
            },
        }

    def analyze_prompt(self, prompt: str) -> Dict:
        """
        Analyze a prompt for toxicity, bias, and subjectivity

        Returns:
            Dict containing:
                - has_issues: bool
                - toxicity_score: float
                - issues: List[Dict] with detected problems
        """
        results = {
            'has_issues': False,
            'toxicity_score': 0.0,
            'issues': []
        }

        # Check toxicity
        toxicity_results = self.toxicity_model.predict(prompt)
        max_toxicity = max(toxicity_results.values())
        results['toxicity_score'] = float(max_toxicity)

        if max_toxicity > 0.5:
            # Find which type of toxicity
            toxic_types = [k for k, v in toxicity_results.items() if v > 0.5]
            results['has_issues'] = True
            results['issues'].append({
                'type': 'toxicity',
                'severity': 'high' if max_toxicity > 0.7 else 'medium',
                'description': f'Toxicity Detected: {", ".join(toxic_types)}',
                'suggestion': 'Consider rephrasing to use more respectful language',
                'start': 0,
                'end': len(prompt),
                'text': prompt,
                'score': float(max_toxicity)
            })

        # Check for cognitive biases
        for bias_type, bias_info in self.bias_patterns.items():
            for pattern in bias_info['patterns']:
                matches = re.finditer(pattern, prompt, re.IGNORECASE)
                for match in matches:
                    results['has_issues'] = True
                    results['issues'].append({
                        'type': 'cognitive_bias',
                        'bias_type': bias_type,
                        'severity': 'medium',
                        'description': bias_info['description'],
                        'suggestion': bias_info['suggestion'],
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group(),
                        'score': 0.7
                    })

        # Check for subjectivity
        for subj_type, subj_info in self.subjectivity_patterns.items():
            for pattern in subj_info['patterns']:
                matches = re.finditer(pattern, prompt, re.IGNORECASE)
                for match in matches:
                    results['has_issues'] = True
                    results['issues'].append({
                        'type': 'subjectivity',
                        'subj_type': subj_type,
                        'severity': 'low',
                        'description': subj_info['description'],
                        'suggestion': subj_info['suggestion'],
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group(),
                        'score': 0.5
                    })

        # Sort issues by position in text
        results['issues'].sort(key=lambda x: x['start'])

        return results

    def generate_correction(self, prompt: str, issue: Dict) -> str:
        """
        Generate a suggested correction for a detected issue
        """
        corrections = {
            'confirmation_bias': {
                r'\bobviously\b': 'it appears that',
                r'\bclearly\b': 'it seems that',
                r'\beveryone knows\b': 'it is commonly believed that',
            },
            'groupthink': {
                r'\beveryone thinks\b': 'many people think',
                r'\bwe all know\b': 'it is widely believed',
                r'\bnobody thinks\b': 'some may disagree, but',
            },
            'hasty_generalization': {
                r'\balways\b': 'often',
                r'\bnever\b': 'rarely',
                r'\ball of them\b': 'many of them',
                r'\bevery single\b': 'most',
            },
            'stereotyping': {
                r'\ball (men|women|people)\b': r'some \1',
                r'\bevery (men|women|people)\b': r'many \1',
            },
            'opinion_markers': {
                r'\bI think\b': 'The evidence suggests',
                r'\bI believe\b': 'Research indicates',
                r'\bin my opinion\b': 'Analysis shows',
            },
            'absolute_language': {
                r'\bbest\b': 'highly effective',
                r'\bworst\b': 'less effective',
                r'\bperfect\b': 'well-suited',
            },
            'emotional_language': {
                r'\blove\b': 'prefer',
                r'\bhate\b': 'have concerns about',
                r'\bbrilliant\b': 'effective',
                r'\bstupid\b': 'ineffective',
            },
        }

        # Get the relevant correction map
        issue_category = issue.get('bias_type') or issue.get('subj_type')
        if issue_category in corrections:
            correction_map = corrections[issue_category]
            text = issue['text']

            for pattern, replacement in correction_map.items():
                if re.search(pattern, text, re.IGNORECASE):
                    corrected = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    # Return the full prompt with the correction
                    return prompt[:issue['start']] + corrected + prompt[issue['end']:]

        # If no specific correction, return original
        return prompt
