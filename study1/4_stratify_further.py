#!/usr/bin/env python3
"""
Stratify SSL stimuli for review.

Inputs
- data/clean/ssl_stimuli_unannot.csv
  (expects: rot, action, situation, ssl_domain, agreement_condition)

Outputs
- data/clean/human_stimuli_stratified.csv
  (up to 9 rows per (ssl_domain, agreement_condition))

What it does
- Takes a balanced sample per (domain, agreement) using an oversample factor of 3 (3*3=9).
- Keeps only the columns needed for quick verification.
"""

import pandas as pd

# ---- configuration ----
IN_PATH = "data/clean/ssl_stimuli_unannot.csv"
OUT_PATH = "data/clean/human_stimuli_stratified.csv"

BASE_PER_BUCKET = 3
OVERSAMPLE_FACTOR = 3
TARGET_PER_BUCKET = BASE_PER_BUCKET * OVERSAMPLE_FACTOR  # 9

RNG_SEED = 42

# ---- load ----
data = pd.read_csv(IN_PATH)

required = {"rot", "action", "situation", "ssl_domain", "agreement_condition"}
missing = required - set(data.columns)
if missing:
    print(f"Warning: missing expected columns: {sorted(missing)}")

# ---- stratify ----
selected = []
domains = data["ssl_domain"].dropna().unique()
agreements = data["agreement_condition"].dropna().unique()

for d in domains:
    for a in agreements:
        subset = data[(data["ssl_domain"] == d) & (data["agreement_condition"] == a)]
        n = min(TARGET_PER_BUCKET, len(subset))
        if n > 0:
            selected.append(subset.sample(n=n, random_state=RNG_SEED))

final = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=list(required))

# ---- tidy columns and save ----
keep = [c for c in ["rot", "action", "situation", "ssl_domain", "agreement_condition"] if c in final.columns]
if keep:
    final = final[keep]

final.to_csv(OUT_PATH, index=False)

# ---- quick summary ----
print(f"Saved → {OUT_PATH}")
print(f"Target per (domain, agreement): {TARGET_PER_BUCKET}")
if not final.empty:
    print(final.groupby(["ssl_domain", "agreement_condition"]).size())
else:
    print("No rows selected.")
