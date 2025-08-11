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


def generate_prompt(action, situation, rating, style):
    """Generate more human-like prompt for the LLM."""
    intro_options = [
        f"I'd rate this a {rating} out of 100 for appropriateness.",
        f"This would be a {rating} out of 100 in terms of appropriateness.",
        f"This seems like a {rating} out of 100 on the appropriateness scale.",
        f"This probably lands around a {rating} out of 100 for appropriateness.",
        f"It feels like a {rating} out of 100 would be a fair score here.",
        f"Maybe around a {rating} out of 100 is right for this action.",
        f"Honestly, I'd give this about a {rating} out of 100.",
        f"This feels like a solid {rating}/100 for how appropriate it is.",
        f"I'm leaning toward a {rating} out of 100 on this one.",
        f"Seems like a {rating}/100 rating fits this situation."
    ]
    intro_line = random.choice(intro_options)

    base_prompt = (
        f"Given the action '{action}' in the situation '{situation}', "
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
            action = row['action']
            situation = row['situation']
            rot = row.get('rot', "")  # Leave empty since this script doesn't process RoT

            rating = random.randint(0, 100)
            style_num = random.randint(1, 3)
            style_label = PROMPT_CONDITIONS[style_num]

            prompt = generate_prompt(action, situation, rating, style_num)
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
                'experiment_condition': 'action_situation',
                'rating': rating,
                'llm_response_rot': "",  # left empty for action/situation script
                'llm_response_action_situation': reformatted_response,
                'prompt_condition': style_label,
            })

            pbar.update(1)
            time.sleep(DELAY_BETWEEN_CALLS)

    final_df = pd.DataFrame(responses)
    final_df['idx'] = [i + 1 for i in range(len(final_df))]
    final_df.to_csv('ai_action_situation_stimuli.csv', index=False)
    logging.info("✅ CSV file created successfully: ai_action_situation_stimuli.csv")

    if failed_rows:
        df.iloc[failed_rows].to_csv('failed_rows.csv', index=False)
        logging.warning(f"{len(failed_rows)} rows failed. Saved to failed_rows.csv for retry.")


if __name__ == '__main__':
    main()
