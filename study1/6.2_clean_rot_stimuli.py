import pandas as pd


def newlines2html(s: str) -> str:
    return s.replace("\n", "<br>")

df = pd.read_csv("../data/clean/ai_rot_stimuli_new_prompt.csv")
df['llm_response_rot'] = df['llm_response_rot'].apply(newlines2html)
df.to_csv("../data/clean/ai_rot_stimuli_new_prompt.csv", index=False)
