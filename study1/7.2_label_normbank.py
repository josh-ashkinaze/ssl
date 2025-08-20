"""
Author: Joshua Ashkinaze

Description: Labels NormBank examples using schema of Moral, Societal-Conventional, Personal-Psychological, Other.

Input:
- data/raw/normbank.csv

Output:
- data/clean/normbank_predictions_{N}.csv: NormBank examples with model predictions.
-


Date: 2025-08-20 12:59:45
# gpt-4o-mini-2024-07-18
"""

import os
import math
import pandas as pd
from litellm import batch_completion
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm

load_dotenv("../src/.env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


N = 3000 # n to label

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)
output_fn = f"../data/clean/normbank_predictions_{N}.csv"



def log_and_print(msg):
    print(msg)
    logging.info(msg)



PROMPT_BASE = """
INSTRUCTIONS
Given a behavior, setting, and constraints,

Return whether this behavior describes

- Moral: what is right and wrong
- Societal-Conventional: customs, norms, roles, etiquette, social expectations
- Personal-Psychological: self-concept, relationships, emotions, personal preferences, identity
- Other: anything that does not fit into the above categories

RETURN

A value in ['Moral', 'Societal-Conventional', 'Personal-Psychological', 'Other'] and absolutely nothing else.
""".strip()

ALLOWED = {"moral": "Moral",
           "societal-conventional": "Societal-Conventional",
           "personal-psychological": "Personal-Psychological",
           "other": "Other"}

def turn_into_row(x):
    return f"{x['behavior']} at a {x['setting']} where {x['constraints']}"

def make_prompt(norm: str) -> str:
    return f"{PROMPT_BASE}\n\nNorm:{norm}"

def normalize_label(s: str) -> str:
    """Best-effort cleanup to one of the 4 allowed labels."""
    if not isinstance(s, str):
        return "Other"
    t = s.strip().lower()
    t = t.replace(".", "").replace("'", "")
    if t in ("societal", "conventional", "societal conventional"):
        t = "societal-conventional"
    if t in ("personal", "psychological", "personal psychological"):
        t = "personal-psychological"
    return ALLOWED.get(t, ALLOWED.get(t.split()[0], "Other"))

def batch_predict(prompts, model="gpt-4o-mini", max_tokens=10, temperature=0, num_retries=3, timeout=None):
    """
    Run a batch of prompts through LiteLLM's batch_completion.
    Returns a list of raw strings (model outputs).
    """
    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    # You can also pass provider-specific kwargs; LiteLLM will drop unsupported ones.
    # See docs: https://docs.litellm.ai/docs/completion/batching  (cited above)
    responses = batch_completion(
        model=model,
        seed=42,
        messages=messages_list,
        max_tokens=max_tokens,
        temperature=temperature,
        num_retries=num_retries,
        timeout=timeout
    )

    out = []
    for r in responses:
        try:
            content = r["choices"][0]["message"]["content"]
        except (TypeError, KeyError):
            content = getattr(getattr(r, "choices", [{}])[0], "message", {}).get("content", "")
        out.append(content if isinstance(content, str) else "")
    return out

def count_ands(x):
    return x.count("[AND]")


def main():
    if os.path.exists(output_fn):
        print("file exists")
        return None

    else:
        pass

    df = pd.read_csv("../data/raw/normbank.csv")
    df['num_ands'] = df['constraints'].apply(count_ands)
    df = df.query("num_ands <= 0").copy()
    df["norm"] = df.apply(turn_into_row, axis=1)
    df = df.sample(N, random_state=42)
    df["prompt"] = df["norm"].apply(make_prompt)

    BATCH_SIZE = 100
    prompts = df["prompt"].tolist()
    preds = []

    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        chunk = prompts[i:i + BATCH_SIZE]
        chunk_preds = batch_predict(
            chunk,
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=10,
        )
        preds.extend(chunk_preds)

    df["pred_raw"] = preds
    df["pred"] = df["pred_raw"].map(normalize_label)

    log_and_print(df["pred"].value_counts(dropna=False))


    df.to_csv(output_fn, index=False)
    log_and_print(f"Wrote to {output_fn}")

if __name__ == "__main__":
    main()

