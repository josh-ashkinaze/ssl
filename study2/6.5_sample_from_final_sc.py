#!/usr/bin/env python3
"""
Sample 10 rows per (ssl_domain, agreement_condition) from annotated SSL data,
filtering for personal_valid == 1, without replacement (reproducible).

Input
- ../data/clean/ssl_stimuli_final_annot.csv

Output
- ../data/clean/ssl_stimuli_final_sample_annot.csv
"""

import pandas as pd
from pathlib import Path

# ---- config ----
IN_PATH = "../data/clean/ssl_stimuli_final_annot.csv"
OUT_PATH = "../data/clean/ssl_stimuli_final_sample_annot.csv"
N_PER = 10
RNG_SEED = 42

# ---- load ----
df = pd.read_csv(IN_PATH)

# column name robustness
DOMAIN_COL = "ssl_domain" if "ssl_domain" in df.columns else ("domain" if "domain" in df.columns else None)
AGREE_COL  = "agreement_condition" if "agreement_condition" in df.columns else ("agreement_level" if "agreement_level" in df.columns else None)

required = {"personal_valid"}
if DOMAIN_COL: required.add(DOMAIN_COL)
if AGREE_COL:  required.add(AGREE_COL)

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")

# ---- filter personal_valid == 1 ----
# accept 1 (int) or "1" (str)
personal_mask = pd.to_numeric(df["personal_valid"], errors="coerce").fillna(0).astype(int) == 1
df = df[personal_mask].copy()

# ---- stratified sampling: 10 per (domain, agreement) without replacement ----
selected_parts = []
for (d, a), g in df.groupby([DOMAIN_COL, AGREE_COL], dropna=False):
    n = min(N_PER, len(g))
    if n > 0:
        selected_parts.append(g.sample(n=n, replace=False, random_state=RNG_SEED))

final = pd.concat(selected_parts, ignore_index=True) if selected_parts else df.iloc[0:0].copy()

# ---- save ----
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
final.to_csv(OUT_PATH, index=False)

# ---- quick summary ----
print(f"Saved -> {OUT_PATH}")
if not final.empty:
    print(final.groupby([DOMAIN_COL, AGREE_COL]).size())
else:
    print("No rows selected (check filters and inputs).")
