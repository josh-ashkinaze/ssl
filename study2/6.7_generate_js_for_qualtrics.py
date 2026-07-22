"""
Name: Joshua Ashkinaze

Date: 2026-07-22

Description: Generate the Qualtrics artifacts for SSL Study 2 randomization.
Builds a stimulus bank keyed by rule of thumb with parallel arrays of variants,
then emits JavaScript that picks N_TRIALS distinct rules of thumb and one
rationale within each, writing the result to embedded data. Also emits the
Loop & Merge table and the embedded field list.

Inputs:
- The CSV containing one row per rationale with
  columns rot, idx, llm_response_rot, rating, low_or_high, prompt_condition,
  domain, agreement_condition

Outputs:
- ssl_generator.js, paste into the setup question's JavaScript
- ssl_loop_table.tsv, paste into the Loop & Merge table
- ssl_embedded_fields.txt, declare these in the Survey Flow
"""

import json

import pandas as pd

STIMULI_CSV = "../data/clean/rot_stimuli_full_gpt-5.6-terra_kimi-k3_n135.csv"
OUT_DIR = "../data/clean"

N_TRIALS = 10

# Order defines the Loop & Merge column number: first entry is ${lm://Field/1}.
LOOP_FIELDS = [
    "rot_idx",
    "rot_text",
    "rationale",
    "rating",
    "direction",
    "strategy",
    "domain",
    "agreement",
]


####################
# Stimulus bank
####################

def build_rot_bank(stimuli):
    """Group rationales under each rule of thumb as parallel arrays."""
    bank = {}
    for rot_text, group in stimuli.groupby("rot"):
        bank[rot_text] = {
            "rot_text": rot_text,
            "rot_idxs": [int(i) for i in group["rot_idx"]],
            "rationales": list(group["llm_response_rot"]),
            "ratings": [int(r) for r in group["rating"]],
            "directions": list(group["low_or_high"]),
            "strategies": list(group["prompt_condition"]),
            "domain": group["domain"].iloc[0],
            "agreement": group["agreement_condition"].iloc[0],
        }
    return bank


####################
# JavaScript
####################

def build_generator_js(bank):
    """Write the setup-question script that assigns each participant's trials."""
    bank_literal = json.dumps(bank, ensure_ascii=False, indent=8)
    return f"""Qualtrics.SurveyEngine.addOnload(function() {{
    console.log("--- SSL Initialization Script Started ---");

    var N_TRIALS = {N_TRIALS};

    var allRots = {bank_literal};

    var allRotKeys = Object.keys(allRots);

    function shuffle(array) {{
        var currentIndex = array.length, temporaryValue, randomIndex;
        while (0 !== currentIndex) {{
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex -= 1;
            temporaryValue = array[currentIndex];
            array[currentIndex] = array[randomIndex];
            array[randomIndex] = temporaryValue;
        }}
        return array;
    }}

    var shuffled = shuffle(allRotKeys);
    var selectedRots = shuffled.slice(0, N_TRIALS);
    console.log("Selected RoTs:", selectedRots);

    for (var i = 0; i < N_TRIALS; i++) {{
        var rot = allRots[selectedRots[i]];

        var randomIndex = Math.floor(Math.random() * rot.rationales.length);
        var trial = i + 1;

        Qualtrics.SurveyEngine.setJSEmbeddedData("rot_idx_" + trial, rot.rot_idxs[randomIndex]);
        Qualtrics.SurveyEngine.setJSEmbeddedData("rot_text_" + trial, rot.rot_text);
        Qualtrics.SurveyEngine.setJSEmbeddedData("rationale_" + trial, rot.rationales[randomIndex]);
        Qualtrics.SurveyEngine.setJSEmbeddedData("rating_" + trial, rot.ratings[randomIndex]);
        Qualtrics.SurveyEngine.setJSEmbeddedData("direction_" + trial, rot.directions[randomIndex]);
        Qualtrics.SurveyEngine.setJSEmbeddedData("strategy_" + trial, rot.strategies[randomIndex]);
        Qualtrics.SurveyEngine.setJSEmbeddedData("domain_" + trial, rot.domain);
        Qualtrics.SurveyEngine.setJSEmbeddedData("agreement_" + trial, rot.agreement);

        console.log("Trial " + trial + ": rot_idx " + rot.rot_idxs[randomIndex]
                    + ", direction " + rot.directions[randomIndex]);
    }}
}});
"""


####################
# Loop table and fields
####################

def build_loop_table():
    """Write the Loop & Merge table, one row per trial, tab delimited."""
    rows = []
    for trial in range(1, N_TRIALS + 1):
        rows.append("\t".join(
            "${e://Field/__js_" + f"{name}_{trial}" + "}" for name in LOOP_FIELDS
        ))
    return "\n".join(rows)


def build_embedded_fields():
    """List every field the loop table reads, for the Survey Flow."""
    return "\n".join(
        f"__js_{name}_{trial}"
        for name in LOOP_FIELDS
        for trial in range(1, N_TRIALS + 1)
    )


def write_output(filename, content):
    """Write one artifact to the output directory."""
    with open(f"{OUT_DIR}/{filename}", "w", encoding="utf-8") as f:
        f.write(content)


def main():
    stimuli = pd.read_csv(STIMULI_CSV)
    bank = build_rot_bank(stimuli)

    write_output("ssl_generator.js", build_generator_js(bank))
    write_output("ssl_loop_table.tsv", build_loop_table())
    write_output("ssl_embedded_fields.txt", build_embedded_fields())

    variant_counts = [len(rot["rationales"]) for rot in bank.values()]
    print(f"unique RoTs in bank: {len(bank)}")
    print(f"rationales per RoT: min={min(variant_counts)} max={max(variant_counts)}")
    print(f"trials per participant: {N_TRIALS}")
    print(f"fields written per participant: {N_TRIALS * len(LOOP_FIELDS)}")
    print(f"loop table: {N_TRIALS} rows x {len(LOOP_FIELDS)} columns")

    for column_number, name in enumerate(LOOP_FIELDS, start=1):
        print(f"  ${{lm://Field/{column_number}}} = {name}")


if __name__ == "__main__":
    main()