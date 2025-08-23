"""
Author: Joshua Ashkinaze

Description: Basic litellm script for prompting.

Input:
- None

Output:
- litellm_output.json: Contains the model responses to the prompt.

Date: 2025-07-15 11:58:38
"""

import litellm
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("../src/.env")
OVERWRITE = False
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
OUTPUT_FN = "../data/clean/personal_words.json"
MODELS = ["gemini/gemini-2.5-flash"]

PROMPT = f"""

CONTEXT
Social domain theorists have defined three domains of social knowledge. 

<start>
The moral domain is “Prescriptive, generalizable understandings of
how individuals ought to behave toward others”—what is right and wrong. The societal-conventional 
domain is “Empirical, descriptive knowledge based upon a recognition of the uniformities in the social environment” 
including “societal arrangements, social organization, and social norms and customs”. 
The personal-psychological domain is knowledge that “define[s] the private aspects of one’s life” 
and “inferences about others’ thoughts, feelings, intentions, and knowledge of personality, self, and identity” 
<end>

RETURN
A list of 20 unigrams that are high-precision for the personal domain as a Python list and nothing else. It's important these unigrams will not be confused with other domains.
""".strip()


def prompt_model(prompt, model, **kwargs):
    """
    Sends a text prompt to the specified language model using litellm.
    """
    messages = [{"content": prompt, "role": "user"}]
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            **kwargs
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred during model prompting: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    if Path(OUTPUT_FN).exists() and not OVERWRITE:
        print(f"Output file {OUTPUT_FN} already exists. Set OVERWRITE=True to overwrite.")
        exit(0)

    else:
        pass

    data = []
    for model in MODELS:
        resp = prompt_model(
            prompt=PROMPT,
            model=model,
        )
        resp = resp.replace("```python", "")
        resp = resp.replace("```", "")

        output_data = {
            "model": model,
            "prompt": PROMPT,
            "response": eval(resp.strip())
        }
        data.append(output_data)
        print(f"Model: {model} - Response: {resp}")

    with open(OUTPUT_FN, "w") as f:
        json.dump(data, f, indent=2)
