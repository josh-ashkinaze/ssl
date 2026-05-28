"""
Name: Joshua Ashkinaze

Date: 2026-02-26

Description: Two analyses for AI chatbot survey data (Q38: Ever Used AI Chatbot).
             (1) Demographic breakdowns for Yes/No users vs. overall population,
                 printed to console and written as a LaTeX longtable.
             (2) Prolific income quotas for recruiting a quota-matched sample,
                 with within-bucket allocation proportional to Prolific pool size.

Inputs:
- None (data hard-coded from survey table image)

Outputs:
- Prints demographic breakdown table (Yes / No / Overall / Delta / RR)
- Writes demographic breakdown as LaTeX longtable to yougov_demographics.tex
- Prints Prolific income quota table
"""

import os
import pandas as pd

####################
# Survey Data
####################

DEMOGRAPHIC_GROUPS = {
    "Gender": {
        "Male":   {"N": 857,  "pct_yes": 0.61},
        "Female": {"N": 913,  "pct_yes": 0.56},
    },
    "Age": {
        "18--29": {"N": 361,  "pct_yes": 0.82},
        "30--44": {"N": 449,  "pct_yes": 0.68},
        "45--64": {"N": 581,  "pct_yes": 0.54},
        "65+":    {"N": 378,  "pct_yes": 0.33},
    },
    "Race": {
        "White":    {"N": 1116, "pct_yes": 0.56},
        "Black":    {"N": 221,  "pct_yes": 0.67},
        "Hispanic": {"N": 283,  "pct_yes": 0.57},
    },
    "Party ID": {
        "Democrat":    {"N": 539, "pct_yes": 0.60},
        "Republican":  {"N": 563, "pct_yes": 0.54},
        "Independent": {"N": 501, "pct_yes": 0.64},
    },
    "Income": {
        r"<\$50K":      {"N": 714, "pct_yes": 0.52},
        r"\$50--100K":  {"N": 502, "pct_yes": 0.58},
        r"\$100K+":     {"N": 369, "pct_yes": 0.71},
    },
    "2024 Vote": {
        "Harris":     {"N": 548, "pct_yes": 0.62},
        "Trump":      {"N": 569, "pct_yes": 0.53},
        "Not voting": {"N": 618, "pct_yes": 0.59},
    },
}

TARGET_SAMPLE_SIZE = 1000  # change to desired Prolific N

INCOME_TARGETS = {
    "<$50K":    0.402,
    "$50-100K": 0.315,
    "$100K+":   0.284,
}

PROLIFIC_BRACKETS = [
    {"label": "Less than $10,000",   "pool":  597, "bucket": "<$50K"},
    {"label": "$10,000--$15,999",    "pool":  543, "bucket": "<$50K"},
    {"label": "$16,000--$19,999",    "pool":  257, "bucket": "<$50K"},
    {"label": "$20,000--$29,999",    "pool":  693, "bucket": "<$50K"},
    {"label": "$30,000--$39,999",    "pool":  800, "bucket": "<$50K"},
    {"label": "$40,000--$49,999",    "pool":  835, "bucket": "<$50K"},
    {"label": "$50,000--$59,999",    "pool":  892, "bucket": "$50-100K"},
    {"label": "$60,000--$69,999",    "pool":  724, "bucket": "$50-100K"},
    {"label": "$70,000--$79,999",    "pool":  745, "bucket": "$50-100K"},
    {"label": "$80,000--$89,999",    "pool":  603, "bucket": "$50-100K"},
    {"label": "$90,000--$99,999",    "pool":  646, "bucket": "$50-100K"},
    {"label": "$100,000--$149,999",  "pool": 1765, "bucket": "$100K+"},
    {"label": "More than $150,000",  "pool": 1369, "bucket": "$100K+"},
]

LATEX_OUTPUT_PATH = "../tables/yougov_quota.tex"

####################
# Demographics
####################

def compute_cell_counts(groups):
    """Convert group N + pct_yes into yes/no cell counts."""
    rows = []
    for category, subgroups in groups.items():
        for group_name, data in subgroups.items():
            n_yes = round(data["N"] * data["pct_yes"])
            n_no  = data["N"] - n_yes
            rows.append({
                "category": category,
                "group":    group_name,
                "n_yes":    n_yes,
                "n_no":     n_no,
                "n_total":  data["N"],
            })
    return pd.DataFrame(rows)

def compute_category_breakdown(df, category):
    """For a single category, compute % of Yes, No, Overall, Delta, and RR."""
    subdf         = df[df["category"] == category].copy()
    total_yes     = subdf["n_yes"].sum()
    total_no      = subdf["n_no"].sum()
    total_overall = subdf["n_total"].sum()

    rows = []
    for _, row in subdf.iterrows():
        yes_pct     = round(row["n_yes"]   / total_yes     * 100, 1)
        overall_pct = round(row["n_total"] / total_overall * 100, 1)
        rows.append({
            "Group":   row["group"],
            "Yes":     yes_pct,
            "No":      round(row["n_no"] / total_no * 100, 1),
            "Overall": overall_pct,
            "Delta":   round(yes_pct - overall_pct, 1),
            "RR":      round(yes_pct / overall_pct, 2) if overall_pct > 0 else float("nan"),
        })
    return pd.DataFrame(rows)

def print_demographic_breakdowns(groups):
    """Print Yes / No / Overall / Delta / RR for every demographic category."""
    df    = compute_cell_counts(groups)
    col_w = 10

    print(f"\n{'='*70}")
    print(f"  Demographic Breakdowns: Yes vs. No vs. Overall")
    print(f"{'='*70}")

    for category in df["category"].unique():
        breakdown = compute_category_breakdown(df, category)
        print(f"\n  {category}")
        print(f"  {'-'*60}")
        print(f"  {'Group':<15} {'Yes':>{col_w}} {'No':>{col_w}} {'Overall':>{col_w}} {'Delta':>{col_w}} {'RR':>{col_w}}")
        for _, row in breakdown.iterrows():
            delta_str = f"{row['Delta']:+.1f}%"
            print(f"  {row['Group']:<15} {row['Yes']:>{col_w}.1f}% {row['No']:>{col_w}.1f}% {row['Overall']:>{col_w}.1f}% {delta_str:>{col_w}} {row['RR']:>{col_w}.2f}")

####################
# LaTeX Table
####################

def build_latex_longtable(groups):
    """Return a LaTeX longtable string of the full demographic breakdown."""
    df = compute_cell_counts(groups)

    lines = []
    lines.append(r"\begin{longtable}{llrrrrr}")
    lines.append(r"\caption{Demographic breakdown of U.S.\ chatbot users vs.\ non-users")
    lines.append(r"         (2025 YouGov survey, $N = 1{,}769$).}")
    lines.append(r"\label{tab:yougov_demographics}\\")
    lines.append(r"\toprule")
    lines.append(r"Category & Group & Yes (\%) & No (\%) & Overall (\%) & $\Delta$ (p.p.) & RR \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{7}{c}{\tablename\ \thetable{} -- continued}\\")
    lines.append(r"\toprule")
    lines.append(r"Category & Group & Yes (\%) & No (\%) & Overall (\%) & $\Delta$ (p.p.) & RR \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{r}{Continued on next page}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for category in df["category"].unique():
        breakdown = compute_category_breakdown(df, category)
        n_rows    = len(breakdown)

        for i, (_, row) in enumerate(breakdown.iterrows()):
            cat_cell   = f"\\multirow{{{n_rows}}}{{*}}{{{category}}}" if i == 0 else ""
            delta_sign = "+" if row["Delta"] >= 0 else ""
            lines.append(
                f"{cat_cell} & {row['Group']} & "
                f"{row['Yes']:.1f} & {row['No']:.1f} & {row['Overall']:.1f} & "
                f"{delta_sign}{row['Delta']:.1f} & {row['RR']:.2f} \\\\"
            )

        lines.append(r"\midrule")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)

def write_latex_table(groups, path):
    """Write the LaTeX longtable to a .tex file, creating dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    latex = build_latex_longtable(groups)
    with open(path, "w") as f:
        f.write(latex)
    print(f"\n  LaTeX table written to: {path}")

####################
# Prolific Quotas
####################

def normalize_targets(income_targets):
    """Ensure target percentages sum to exactly 1.0."""
    total = sum(income_targets.values())
    return {k: v / total for k, v in income_targets.items()}

def compute_prolific_quotas(brackets, income_targets, total_n):
    """
    For each coarse bucket, split target N across fine brackets proportional
    to their Prolific pool size. Uses largest-remainder rounding to hit exact
    per-bucket targets.
    """
    income_targets = normalize_targets(income_targets)
    df             = pd.DataFrame(brackets)

    rows = []
    for bucket, target_pct in income_targets.items():
        target_n  = total_n * target_pct
        bucket_df = df[df["bucket"] == bucket].copy()
        pool_total = bucket_df["pool"].sum()

        bucket_df["quota_raw"] = target_n * (bucket_df["pool"] / pool_total)
        bucket_df["quota"]     = bucket_df["quota_raw"].apply(int)
        bucket_df["remainder"] = bucket_df["quota_raw"] - bucket_df["quota"]

        shortfall = round(target_n) - bucket_df["quota"].sum()
        if shortfall > 0:
            top_idx = bucket_df["remainder"].nlargest(shortfall).index
            bucket_df.loc[top_idx, "quota"] += 1
        elif shortfall < 0:
            bottom_idx = bucket_df["remainder"].nsmallest(abs(shortfall)).index
            bucket_df.loc[bottom_idx, "quota"] -= 1

        rows.append(bucket_df)

    return pd.concat(rows).reset_index(drop=True), income_targets

def print_prolific_quotas(brackets, income_targets, total_n):
    """Print Prolific quota table grouped by coarse income bucket."""
    df, normalized_targets = compute_prolific_quotas(brackets, income_targets, total_n)

    print(f"\n{'='*65}")
    print(f"  Prolific Income Quotas  (Target N = {total_n})")
    print(f"{'='*65}")

    for bucket, target_pct in normalized_targets.items():
        bucket_df    = df[df["bucket"] == bucket]
        bucket_quota = bucket_df["quota"].sum()
        print(f"\n  {bucket}  —  target {target_pct*100:.1f}%  →  N={bucket_quota}")
        print(f"  {'-'*50}")
        print(f"  {'Bracket':<28} {'Quota':>8} {'% of Total':>12}")
        for _, row in bucket_df.iterrows():
            pct_of_total = row["quota"] / total_n * 100
            print(f"  {row['label']:<28} {row['quota']:>8} {pct_of_total:>11.1f}%")

    print(f"\n  {'─'*50}")
    print(f"  Total quota: {df['quota'].sum()}  /  Target: {total_n}")

####################
# Main
####################

def main():
    print_demographic_breakdowns(DEMOGRAPHIC_GROUPS)
    write_latex_table(DEMOGRAPHIC_GROUPS, LATEX_OUTPUT_PATH)
    print_prolific_quotas(PROLIFIC_BRACKETS, INCOME_TARGETS, TARGET_SAMPLE_SIZE)

if __name__ == "__main__":
    main()