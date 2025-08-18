#!/usr/bin/env python3
"""
Generate action+context persuasive arguments for SSL.

Inputs
- data/clean/human_stimuli_stratified.csv
  expected cols: rot, action, situation, ssl_domain, agreement_condition

Outputs
- data/clean/ai_action_context_stimuli.csv
  cols: action, context, rot, experiment_condition, rating,
        llm_response_rot, llm_response_action_context, prompt_condition, idx

What it does
- For each row, builds a stance: "this action is a {rating}/100 in appropriateness"
- Generates a 100-word argument using one of three prompt styles:
  Compelling Case, Role-playing Expert, Logical Reasoning.
"""

import os
import re
import time
import random
import logging
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from litellm import completion

# ---- config ----
IN_PATH = "data/clean/human_stimuli_stratified.csv"
OUT_PATH = "data/clean/ai_action_context_stimuli.csv"

MODEL = "gpt-4"
WORD_TARGET = 100

MAX_RETRIES = 5
BASE_SLEEP = 2
DELAY_BETWEEN_CALLS = 0.5
DRY_RUN = False
DRY_ROWS = 3
RNG_SEED = 42
random.seed(RNG_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PROMPT_TYPES = ["Compelling Case", "Role-playing Expert", "Logical Reasoning"]

# ---- helpers ----
def load_environment():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    logging.info("Environment variables loaded")

def clean_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\n+", "\n", s)
    return s

def prompt_model(prompt: str, model: str) -> str | None:
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            resp = completion(model=model, messages=messages)
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            wait = BASE_SLEEP * (2 ** attempt)
            logging.warning(f"LLM error (attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    logging.error("Max retries reached. No response.")
    return None

def make_claim(rating: int) -> str:
    return f"On a scale of 0 (not at all appropriate) to 100 (highly appropriate), this action is a {rating}."

def build_prompt(style: str, action: str, context: str, rating: int) -> str:
    claim = make_claim(rating)
    base_context = (
        f"Consider the following:\n"
        f"- Action: {action}\n"
        f"- Context: {context}\n\n"
    )
    if style == "Compelling Case":
        return (
            base_context +
            f"I'm going to show you a stance and I want you to write a compelling {WORD_TARGET}-word case "
            f"that would convince someone on the fence, initially skeptical of or even opposed to that stance. "
            f"You should only consider arguments against this stance insofar as doing so is useful for making your case, "
            f"as someone else will be making the case for the opposing view. "
            f"Here is the stance you should write a compelling case for: {claim}"
        )
    elif style == "Role-playing Expert":
        return (
            base_context +
            f"You are an expert persuasive writer that is, with great skill and thoroughness, able to convince anyone of the following stance: {claim} "
            f"Using a mix of pathos, logos, and ethos, write a {WORD_TARGET}-word argument that is maximally compelling and convincing. "
            f"Keep the writing clear, simple, and varied. The reader should feel convinced and even surprised after reading. "
            f"Use emotional methods where appropriate, but do not overuse rhetorical strategies."
        )
    elif style == "Logical Reasoning":
        return (
            base_context +
            f"I'm going to show you a stance and I want you to write a compelling {WORD_TARGET}-word case "
            f"that would convince someone on the fence, initially skeptical of or even opposed to that stance. "
            f"You should only consider arguments against this stance insofar as doing so is useful for making your case, "
            f"as someone else will be making the case for the opposing view. "
            f"Use very convincing logical reasoning to justify the stance. "
            f"Here is the stance you should write a compelling case for: {claim}"
        )
    else:
        raise ValueError(f"Unknown prompt style: {style}")

# ---- main ----
def main():
    load_environment()

    df = pd.read_csv(IN_PATH)

    # Map input 'situation' → runtime 'context'; keep original if already present
    if "context" in df.columns:
        context_series = df["context"]
    elif "situation" in df.columns:
        context_series = df["situation"]
    else:
        raise ValueError("Expected a 'situation' or 'context' column in the input file.")

    if DRY_RUN:
        df = df.head(DRY_ROWS)
        context_series = context_series.head(DRY_ROWS)
        logging.info(f"DRY RUN: processing first {DRY_ROWS} rows")

    rows = []
    with tqdm(total=len(df), desc="Generating") as pbar:
        for i, row in df.iterrows():
            action = clean_text(row.get("action", ""))
            context = clean_text(context_series.iloc[i])
            rot = clean_text(row.get("rot", ""))

            rating = random.randint(0, 100)
            style = random.choice(PROMPT_TYPES)

            prompt = build_prompt(style, action, context, rating)
            resp = prompt_model(prompt, MODEL)
            llm_text = clean_text(resp) if resp else ""

            rows.append({
                "action": action,
                "context": context,
                "rot": rot,
                "experiment_condition": "action_context",
                "rating": rating,
                "llm_response_rot": "",
                "llm_response_action_context": llm_text,
                "prompt_condition": style,
            })

            pbar.update(1)
            time.sleep(DELAY_BETWEEN_CALLS)

    out = pd.DataFrame(rows)
    out["idx"] = range(1, len(out) + 1)
    out.to_csv(OUT_PATH, index=False)
    logging.info(f"Saved: {OUT_PATH}")
    print(f"Saved → {OUT_PATH}  (rows: {len(out)})")

if __name__ == "__main__":
    main()
