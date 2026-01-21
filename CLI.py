from utils import generate_llm_response
import argparse
from bias_detection import identify_biases



def main(argv=None) -> int:
    while True:
        prompt = input("Enter your prompt: ")
        if prompt == "":
            break
        biases = identify_biases(prompt, True)
        print(biases)
        #response = generate_llm_response(prompt)
        #print(response)
        #print(identify_biases(response, False))
    return 0


if __name__ == "__main__":
    main()