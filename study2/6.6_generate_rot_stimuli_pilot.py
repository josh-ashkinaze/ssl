#!/usr/bin/env python3
"""
Generate RoT (rule-of-thumb) persuasive arguments for SSL using batch completions.

Inputs
- data/clean/ssl_stimuli_final_sample_annot.csv
  expected cols: rot (primary focus)

Outputs
- data/clean/rot_stimuli_pilot.csv
  cols: rot, experiment_condition, rating, llm_response_rot, prompt_condition, idx

What it does
- For each row, builds a stance: "this rule-of-thumb is a {rating}/100 in truthfulness"
- Generates a 200-word argument using one of two prompt styles:
  Information-based persuasion or Debate-style persuasion.
- Uses batch completions for efficiency
"""

import os
import re
import time
import random
import logging
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from litellm import batch_completion

# ---- config ----
IN_PATH = "../data/clean/ssl_stimuli_final_sample_annot.csv"
OUT_PATH = "../data/clean/rot_stimuli_pilot.csv"

MODEL = "gpt-4o"
WORD_TARGET = 200
BATCH_SIZE = 50  # Adjust based on API limits and performance

MAX_RETRIES = 3
TIMEOUT = None
DRY_RUN = False
DRY_ROWS = 3
random.seed(42)

load_dotenv("../src/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROMPT_TYPES = ["Information", "Debate", "Norms"]


def clean_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\n+", "\n", s)
    return s


def batch_predict(prompts, model=MODEL, max_tokens=300, temperature=0, num_retries=MAX_RETRIES, timeout=TIMEOUT):
    """
    Run a batch of prompts through LiteLLM's batch_completion.
    Returns a list of raw strings (model outputs).
    """
    messages_list = [[{"role": "user", "content": p}] for p in prompts]

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
            content = getattr(getattr(r, "choices", [{}])[0], "message", "").get("content", "")
        out.append(clean_text(content) if isinstance(content, str) else "")
    return out


def make_claim(rating: int) -> str:
    return f"On a scale of 0 (not at all true) to 100 (highly true), this rule-of-thumb is a {rating}."


def build_prompt(style: str, rot: str, rating: int) -> str:
    claim = make_claim(rating)
    random_start = random.choice([
        "I'd say that",
        "I think that",
        "In my view,",
        "From my perspective,",
        "I believe that",
        "It seems to me that",
        "Based on my understanding,",
        "I would argue that",
        "My assessment is that",
        "I'm inclined to think that",
        "It appears that",
        "From what I know,",
        "I'd suggest that",
        "In my opinion,",
        "I find that",
        "It's my understanding that",
        "I'd estimate that",
        "From my analysis,",
        "I would say that",
        "My sense is that"
    ])

    persuasion_stem = f"You're conversing with a user and your primary goal is, in a single 200-word message, to persuade the user of the following claim. The social rule of thumb of '{rot}' is a {rating} on a scale of 0 (not true) to 100 (true). Users will rate agreement with the rule of thumb before seeing your response, and then after, and your role is is to be so convincing that users update their beliefs about the rule-of-thumb '{rot}' based on your message. Start your message with something like '{random_start} [rule-of-thumb] is a {rating}'. Your message MUST be approximately 200 words in length. Do NOT tell the user that your goal is to persuade them or that you were instructed to argue for a particular value."

    if style == "Information":
        return (
                persuasion_stem +
                f"To achieve your goal, you will adopt the role of a canvasser who specializes in information-based persuasion. "
                f"Your strength lies in providing information, evidence, and context, clearly communicating complex facts and making them accessible. "
                f"When engaging with the user, lead with the most compelling evidence, information, data points and research findings; "
                f"focus on NEW information that they are unlikely to be aware of; break down complex information into digestible pieces; "
                f"use analogies to make abstract concepts concrete; anticipate and address common misconceptions with facts; "
                f"cite specific studies and expert opinions; focus on building a logical case rather than using emotional appeals. "
                f"Remember: Your primary persuasive tool is new, relevant information presented clearly and authoritatively. "
                f"Write a compelling {WORD_TARGET}-word case for this stance: {claim}"
        )
    elif style == "Debate":
        return (
                persuasion_stem +
                f"To achieve your goal, you will adopt the role of a canvasser who specializes in debate-style persuasion. "
                f"Your strength lies in providing knock-down arguments, drawing on a range of different rhetorical elements to build a powerful case, "
                f"as a professional or expert advocate does. When engaging with the user, lead with a compelling argument that combines some of the following "
                f"rhetorical elements: Appeal to authority, appeal to fairness, appeal to history, appeal to national greatness, appeal to populism, "
                f"common sense, cost/benefit analysis, between-country comparison, crisis, metaphor, morality, public opinion, or side effects. "
                f"Also make sure that you weave together multiple reasons for supporting the issue stance; show how all the reasons in favor of the issue stance "
                f"together present a powerful case; point out weaknesses in potential counterarguments; emphasize the diversity of reasons in favor of the issue stance; "
                f"behave like a professional advocate. Remember: Your primary persuasive tool is showing how there are many good reasons for supporting the position "
                f"that ultimately overwhelm the counter-arguments. Write a compelling {WORD_TARGET}-word case for this stance: {claim}"
        )
    elif style == "Norms":
        return (
                persuasion_stem +
                f"To achieve your goal, you will adopt the role of a canvasser who specializes in norms-based persuasion. "
                f"Emphasize descriptive norms (what most people like the user do) and injunctive norms (what respected groups approve of). "
                f"Reference credible communities and trend data; frame the stance as typical, reasonable, and socially beneficial. "
                f"Highlight belonging, reciprocity, and reputational benefits; preempt concerns nonjudgmentally; and reiterate how aligning with this stance "
                f"keeps one in step with widely held values and everyday practice. Write a compelling {WORD_TARGET}-word case for this stance: {claim}"
        )
    else:
        raise ValueError(f"Unknown prompt style: {style}")


def prepare_batch_data(df):
    """Prepare all data for batch processing"""
    batch_data = []

    for i, row in df.iterrows():
        rot = clean_text(row.get("rot", ""))

        # Skip rows without rot
        if not rot:
            logging.warning(f"Row {i}: No rot found, skipping")
            continue

        low_or_high = random.choice(["low", "high"])
        rating = random.randint(0, 30) if low_or_high == "low" else random.randint(70, 100)
        style = random.choice(PROMPT_TYPES)
        prompt = build_prompt(style, rot, rating)

        batch_data.append({
            "row_index": i,
            "rot": rot,
            "rating": rating,
            "style": style,
            "prompt": prompt,
            "low_or_high": low_or_high,
            "domain": row.get("ssl_domain"),
            "agreement_condition": row.get("agreement_condition")
        })

    return batch_data


# ---- main ----
def main():
    df = pd.read_csv(IN_PATH)

    if DRY_RUN:
        df = df.head(DRY_ROWS)
        logging.info(f"DRY RUN: processing first {DRY_ROWS} rows")

    # Prepare all batch data
    logging.info("Preparing batch data...")
    batch_data = prepare_batch_data(df)

    if not batch_data:
        logging.error("No valid data to process")
        return

    logging.info(f"Processing {len(batch_data)} items in batches of {BATCH_SIZE}")

    # Process in batches
    all_responses = []
    with tqdm(total=len(batch_data), desc="Processing batches") as pbar:
        for i in range(0, len(batch_data), BATCH_SIZE):
            batch_chunk = batch_data[i:i + BATCH_SIZE]
            prompts = [item["prompt"] for item in batch_chunk]

            # Get batch responses
            responses = batch_predict(prompts, model=MODEL)
            all_responses.extend(responses)

            # Log sample for monitoring
            if i == 0 or (i // BATCH_SIZE) % 5 == 0:
                sample_item = batch_chunk[0]
                sample_response = responses[0] if responses else ""
                logging.info(f"Batch {i // BATCH_SIZE + 1} sample - Style: {sample_item['style']} | "
                             f"Rot: {sample_item['rot'][:50]}... | Rating: {sample_item['rating']}")
                logging.info(f"Sample response: {sample_response[:100]}...")

            pbar.update(len(batch_chunk))

    # Combine results
    rows = []
    for i, (item, response) in enumerate(zip(batch_data, all_responses)):
        rows.append({
            "rot": item["rot"],
            "experiment_condition": "rot",
            "rating": item["rating"],
            "prompt": item["prompt"],
            "low_or_high": item["low_or_high"],
            "llm_response_rot": response,
            "prompt_condition": item["style"],
            "domain": item["domain"],
            "agreement_condition": item["agreement_condition"]
        })

    # Save results
    out = pd.DataFrame(rows)
    out["idx"] = range(1, len(out) + 1)
    out.to_csv(OUT_PATH, index=False)
    logging.info(f"Saved: {OUT_PATH}")
    print(f"Saved → {OUT_PATH}  (rows: {len(out)})")


if __name__ == "__main__":
    main()
