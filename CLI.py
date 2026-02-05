from utils import generate_llm_response
import argparse
from bias_detection import identify_biases


def main(argv=None) -> int:
    while True:
        prompt = input("Enter your prompt: ")
        if prompt == "":
            break
        biases, tokens, global_expl = identify_biases(prompt, True)
        print(biases, tokens, global_expl)
        for t in tokens['problematic_phrases']:
            for w in t.split(" "):
                print(w)
        #response = generate_llm_response(prompt)
    return 0

if __name__ == "__main__":
    main()