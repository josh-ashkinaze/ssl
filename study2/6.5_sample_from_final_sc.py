#!/usr/bin/env python3
"""
Stratify SSL stimuli for review.

Inputs
- data/clean/ssl_stimuli_final_annot.csv
  (expects columns: rot, rot-clean, action, situation, ssl_domain, agreement_condition,
   rot-agree, action-moral-judgment, agreement_label, area, domain_valid, rot_valid)

Outputs
- data/clean/ssl_stimuli_final_sample_annot.csv
  (exactly the same columns as input, sampled)

What it does
- Filters to rows with domain_valid == 1 and rot_valid == 1.
- Samples up to 10 rows per (agreement_condition, ssl_domain).
- Sampling is reproducible (random_state=42) and without replacement (replace=False).
- Keeps all original columns (e.g., rot-clean) unchanged.
"""

import pandas as pd

# ---- configuration ----
IN_PATH = "data/clean/ssl_stimuli_final_annotated.csv"
OUT_PATH = "data/clean/ssl_stimuli_final_sample_annotated.csv"

PER_BUCKET = 10
RNG_SEED = 42

# ---- load ----
data = pd.read_csv(IN_PATH)

required = {
    "rot", "rot-clean", "action", "situation", "ssl_domain", "agreement_condition",
    "rot-agree", "action-moral-judgment", "agreement_label", "area", "domain_valid", "rot_valid"
}
missing = [c for c in required if c not in data.columns]
if missing:
    print(f"Warning: missing expected columns: {sorted(missing)}")

# ---- filter valid ----
valid = data[(data["domain_valid"] == 1) & (data["rot_valid"] == 1)].copy()

# ---- stratified sampling: 10 per (agreement_condition, ssl_domain) ----
selected = []
domains = valid["ssl_domain"].dropna().unique()
agreements = valid["agreement_condition"].dropna().unique()

for d in domains:
    for a in agreements:
        subset = valid[(valid["ssl_domain"] == d) & (valid["agreement_condition"] == a)]
        if len(subset) == 0:
            continue
        n = min(PER_BUCKET, len(subset))
        sampled = subset.sample(n=n, random_state=RNG_SEED, replace=False)
        selected.append(sampled)

final = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=list(data.columns))

# ---- save (keep all columns) ----
final.to_csv(OUT_PATH, index=False)

# ---- quick summary ----
print(f"Saved -> {OUT_PATH}")
print(f"Target per (agreement_condition, ssl_domain): {PER_BUCKET}")
if not final.empty:
    print(final.groupby(["agreement_condition", "ssl_domain"]).size())
else:
    print("No rows selected.")
