import pandas as pd
import random
import os
import time
from dotenv import load_dotenv
import logging
from litellm import completion
import ftfy
import re
from tqdm import tqdm

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_RETRIES = 5
BASE_SLEEP = 2
DELAY_BETWEEN_CALLS = 0.5

DRY_RUN = False  # Toggle to False when ready for full run
DRY_RUN_ROWS = 3

# Map prompt condition values to labels
PROMPT_CONDITIONS = {
    1: "Logical",
    2: "Emotional",
    3: "Expert"
}


def load_environment():
    """Load environment variables and set API keys."""
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
    logging.info("Environment variables loaded")


def reformat_text(text):
    """Fix encoding and normalize spacing."""
    fixed_text = ftfy.fix_text(text)
    reformatted_text = re.sub(r'\n+', '\n', fixed_text)
    return reformatted_text


def prompt_model(prompt, model):
    """Send prompt to LLM model with retry handling for rate limits and errors."""
    messages = [{"content": prompt, "role": "user"}]
    for attempt in range(MAX_RETRIES):
        try:
            response = completion(model=model, messages=messages)
            return response['choices'][0]['message']['content']
        except Exception as e:
            wait_time = BASE_SLEEP * (2 ** attempt)
            logging.warning(f"Error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    logging.error("Max retries reached. No response from API.")
    return None


def generate_prompt(rot, rating, style):
    """Generate different styles of prompts with humanlike intros."""
    intro_options = [
        f"I'd rate this a {rating} out of 100 in truthfulness.",
        f"This would be a {rating} out of 100 in truthfulness.",
        f"This seems like a {rating} out of 100 in truthfulness.",
        f"This probably lands around a {rating} out of 100 in truthfulness.",
        f"This rule feels about a {rating} out of 100 in truthfulness.",
        f"I think this deserves a {rating} out of 100 in truthfulness.",
        f"Most people would probably give this a {rating} out of 100.",
        f"Honestly, it feels like a {rating}/100 to me.",
        f"Realistically, this deserves about a {rating} out of 100.",
        f"I’d guess this is around a {rating} out of 100 in truthfulness."
    ]
    intro_line = random.choice(intro_options)

    base_prompt = (
        f"Given the rule of thumb '{rot}', write a short argument defending its accuracy. "
        f"{intro_line} "
    )

    if style == 1:
        return base_prompt + "Write a clear, logical, 100-word explanation that convinces a skeptical reader."
    elif style == 2:
        return base_prompt + "Write a 100-word argument using emotional and empathetic reasoning to persuade the reader."
    elif style == 3:
        return base_prompt + "Write an expert-level, 100-word persuasive rationale, blending logic and authority."

def main():
    load_environment()

    df = pd.read_csv("human_stimuli_stratified_cleaned.csv")

    if DRY_RUN:
        df = df.head(DRY_RUN_ROWS)
        logging.info(f"⚠️ DRY RUN: Processing only the first {DRY_RUN_ROWS} rows.")

    responses = []
    failed_rows = []

    with tqdm(total=len(df), desc="Generating responses") as pbar:
        for idx, row in df.iterrows():
            rot = row['rot'] if 'rot' in row else ""
            action = row.get('action', "")
            situation = row.get('situation', "")

            rating = random.randint(0, 100)
            style_num = random.randint(1, 3)
            style_label = PROMPT_CONDITIONS[style_num]

            prompt = generate_prompt(rot, rating, style_num)
            logging.info(f"Generated prompt: {prompt[:60]}...")

            llm_response = prompt_model(prompt, 'gpt-4')
            if llm_response:
                reformatted_response = reformat_text(llm_response)
            else:
                reformatted_response = ""
                failed_rows.append(idx)

            responses.append({
                'action': action,
                'situation': situation,
                'rot': rot,
                'experiment_condition': 'rot',
                'rating': rating,
                'llm_response_rot': reformatted_response,
                'llm_response_action_situation': "",  # left empty for RoT script
                'prompt_condition': style_label,
            })

            pbar.update(1)
            time.sleep(DELAY_BETWEEN_CALLS)

    final_df = pd.DataFrame(responses)
    final_df['idx'] = [i + 1 for i in range(len(final_df))]
    final_df.to_csv('ai_rot_stimuli.csv', index=False)
    logging.info("✅ CSV file created successfully: ai_rot_stimuli.csv")

    if failed_rows:
        df.iloc[failed_rows].to_csv('failed_rows_rot.csv', index=False)
        logging.warning(f"{len(failed_rows)} rows failed. Saved to failed_rows_rot.csv for retry.")


if __name__ == '__main__':
    main()
