#!/usr/bin/env python3
"""
Using social chemistry data to select stimuli for experiments.

Here's what this does.

Filters:
- Removes entries with bad words
- Removes entries with missing key attributes
- removes entries where mturk worker said it was bad data

Calculates
- Whether something is low, medium, or high agreement based on rot-agree

Classifies into SSL domains based on Social Domain Theory:
- Classified as moral if:
    - categorization contains "morality-ethics"
- Classified as societal-conventional if:
    - categorization contains "social-norms"
- Classified as personal-psychological if:
    - categorization contains "advice" and the rot or action text contains personal-psychological indicators which are
    comprised of ATUS roles and personal words

For each (domain, agreement level) we get N low, N medium, N high agreement entries but oversample by a factor of OVERSAMPLE_FACTOR
to account for non-compliant output since its very likely that some of these are messed up in some way (e.g, domain is wrong, not really
low agreement etc etc).

Input:
- data/raw/social-chem-101.v1.0.tsv: Social chemistry data
- data/clean/atus_roles.txt: ATUS roles for personal-psychological classification
- data/clean/personal_words.json: Personal words for personal-psychological classification

Output:
- data/clean/ssl_stimuli_unannot.csv: Selected stimuli for SSL experiments
"""

import pandas as pd
import numpy as np
import requests
from flashtext import KeywordProcessor
from collections import Counter
import logging
import os
import json
import random
import re

random.seed(42)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helpers import log_and_print


logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

# ========== PARAMETERS ==========
# Get this many for each domain and tag with low, med, high agreement
N_LOW = 5
N_MED = 5
N_HI = 5

# We oversample by a factor of OVERSAMPLE_FACTOR to account for non-compliant output
OVERSAMPLE_FACTOR = 5

# Data paths
SOCIAL_CHEM_PATH = "../data/raw/social-chem-101.v1.0.tsv"
ATUS_ROLES_PATH = "../data/clean/atus_roles.txt"
OUTPUT_PATH = "../data/clean/ssl_stimuli_unannot.csv"


# ========== HELPER FUNCTIONS ==========

def text2list(filepath):
    """Helper function to read list from text file"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def read_json(filepath):
    """Helper function to read JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


class FastFlashTextCounter:
    def __init__(self, word_lists_dict):
        self.processors = {}
        for name, word_list in word_lists_dict.items():
            processor = KeywordProcessor(case_sensitive=False)
            for word in word_list:
                processor.add_keyword(word.lower())
            self.processors[name] = processor
        log_and_print("FastFlashTextCounter initialized with word lists.")
        logging.info("FastFlashTextCounter initialized with word lists.")

    def count_keywords(self, text, processor_name):
        if pd.isna(text) or not text:
            return {}
        keywords_found = self.processors[processor_name].extract_keywords(str(text).lower())
        return dict(Counter(keywords_found))

    def contains_keywords(self, text, processor_name):
        """Check if text contains any keywords from the specified processor"""
        if pd.isna(text) or not text:
            return False
        keywords_found = self.processors[processor_name].extract_keywords(str(text).lower())
        return len(keywords_found) > 0


def contains_bad_words(x, counter):
    return counter.contains_keywords(x, 'profanity')


def handle_advice(row, all_personal_keywords):
    """Handle classification for the 'advice' category with word boundary matching."""
    rot_text = str(row['rot']).lower()
    action_text = str(row['action']).lower() if pd.notna(row['action']) else ''
    combined_text = f"{rot_text} {action_text}"

    # Create a regex pattern to match exact words with word boundaries
    keyword_pattern = re.compile(r'\b(' + '|'.join(re.escape(keyword) for keyword in all_personal_keywords) + r')\b')

    # Check if any keyword matches in the combined text
    if keyword_pattern.search(combined_text):
        return 'personal-psychological'
    return 'other'

def classify_ssl_domain(row, all_personal_keywords):
    """Classify into SSL domains based on Social Domain Theory."""
    categorization = str(row['rot-categorization']).lower()
    all_cats = [cat.strip() for cat in categorization.split(',') if cat.strip()]

    # Handle single category case
    if len(all_cats) == 1:
        if 'morality-ethics' in all_cats[0]:
            return 'moral'
        elif 'social-norms' in all_cats[0]:
            return 'societal-conventional'
        elif 'advice' in all_cats[0]:
            return handle_advice(row, all_personal_keywords)
        else:
            return 'other'

    # Handle multiple categories
    selected_category = random.choice(all_cats)
    if 'morality-ethics' in selected_category:
        return 'moral'
    elif 'social-norms' in selected_category:
        return 'societal-conventional'
    elif 'advice' in selected_category:
        return handle_advice(row, all_personal_keywords)
    else:
        return 'other'


def enhanced_filter(df):
    """Apply comprehensive filtering for SSL study stimuli"""

    # Basic quality filters
    df_clean = df.query("~has_bad_words & `rot-bad` != 1").copy()
    log_and_print(f"After basic quality filters: {len(df_clean)}")


    # Remove entries with missing key attributes
    key_cols = ['rot-agree', 'action-moral-judgment', 'rot-categorization']
    df_clean = df_clean.dropna(subset=key_cols)
    log_and_print(f"After removing missing key attributes: {len(df_clean)}")


    return df_clean


def analyze_norm_concordance(df):
    """Analyze agreement levels for experimental design"""

    # Convert rot-agree to interpretable labels
    agree_mapping = {0: '<1%', 1: '5-25%', 2: '50%', 3: '75-90%', 4: '>99%'}
    df['agreement_label'] = df['rot-agree'].map(agree_mapping)

    # Create low/medium/high agreement categories for experiments
    df['agreement_category'] = df['rot-agree'].apply(
        lambda x: 'low' if x <= 1 else 'medium' if x == 2 else 'high'
    )


    return df


def select_experimental_stimuli(df, n_low=N_LOW, n_med=N_MED, n_hi=N_HI, oversample_factor=OVERSAMPLE_FACTOR):
    """Select balanced stimuli for SSL experiments with oversampling"""

    stimuli = {}

    # Calculate oversampled amounts
    n_low_oversample = n_low * oversample_factor
    n_med_oversample = n_med * oversample_factor
    n_hi_oversample = n_hi * oversample_factor

    # For each SSL domain
    for domain in ['moral', 'societal-conventional', 'personal-psychological']:
        domain_df = df[df['ssl_domain'] == domain]

        if len(domain_df) == 0:
            log_and_print(f"Warning: No data found for domain {domain}")
            continue

        # Select low-agreement (controversial) stimuli
        low_agree_df = domain_df[domain_df['agreement_category'] == 'low']
        low_sample_size = min(n_low_oversample, len(low_agree_df))
        low_agree = low_agree_df.sample(low_sample_size, random_state=42) if low_sample_size > 0 else pd.DataFrame()

        # Select medium-agreement stimuli
        med_agree_df = domain_df[domain_df['agreement_category'] == 'medium']
        med_sample_size = min(n_med_oversample, len(med_agree_df))
        med_agree = med_agree_df.sample(med_sample_size, random_state=42) if med_sample_size > 0 else pd.DataFrame()

        # Select high-agreement (consensual) stimuli
        hi_agree_df = domain_df[domain_df['agreement_category'] == 'high']
        hi_sample_size = min(n_hi_oversample, len(hi_agree_df))
        hi_agree = hi_agree_df.sample(hi_sample_size, random_state=42) if hi_sample_size > 0 else pd.DataFrame()

        stimuli[domain] = {
            'low_agreement': low_agree,
            'medium_agreement': med_agree,
            'high_agreement': hi_agree
        }

        log_and_print(f"{domain}: {len(low_agree)} low, {len(med_agree)} medium, {len(hi_agree)} high agreement stimuli")

    return stimuli


def preview_ssl_stimuli(stimuli_dict, domain, agreement_level, n=3):
    """Preview selected stimuli for SSL experiments"""

    if domain not in stimuli_dict:
        log_and_print(f"No stimuli found for domain: {domain}")
        return

    if agreement_level not in stimuli_dict[domain]:
        log_and_print(f"No stimuli found for {agreement_level} agreement in {domain}")
        return

    df_subset = stimuli_dict[domain][agreement_level]

    if len(df_subset) == 0:
        log_and_print(f"No stimuli available for {domain} - {agreement_level}")
        return

    df_subset = df_subset.head(n)

    log_and_print(f"\n{'=' * 60}")
    log_and_print(f"{domain.upper()} - {agreement_level.upper()} AGREEMENT")
    log_and_print(f"{'=' * 60}")

    for idx, row in df_subset.iterrows():
        log_and_print(f"RoT: {row['rot']}")
        log_and_print(f"Agreement: {row['agreement_label']} ({row['rot-agree']})")
        log_and_print(f"Moral Judgment: {row['action-moral-judgment']}")
        log_and_print(f"Area: {row['area']}")
        log_and_print(f"Action: {row['action']}")
        log_and_print("-" * 50)


def print_summary_stats(df, stimuli_dict):
    """Print comprehensive summary for SSL study"""

    log_and_print(f"\n{'=' * 60}")
    log_and_print("SSL STUDY SUMMARY STATISTICS")
    log_and_print(f"{'=' * 60}")

    log_and_print(f"Total filtered dataset: {len(df)}")
    log_and_print(f"SSL Domain distribution:")
    log_and_print(df['ssl_domain'].value_counts())

    log_and_print(f"\nAgreement distribution:")
    log_and_print(df['agreement_category'].value_counts())

    log_and_print(f"\nExperimental stimuli selected:")
    for domain, conditions in stimuli_dict.items():
        low_count = len(conditions['low_agreement'])
        med_count = len(conditions['medium_agreement'])
        hi_count = len(conditions['high_agreement'])
        log_and_print(f"  {domain}: {low_count} low, {med_count} medium, {hi_count} high agreement")

    # Quality checks
    log_and_print(f"\nQuality indicators:")
    log_and_print(f"Average RoT length: {df['rot'].str.len().mean():.1f} characters")
    log_and_print(f"Moral judgment range: {df['action-moral-judgment'].min()} to {df['action-moral-judgment'].max()}")


def export_for_experiment(stimuli_dict, output_path=OUTPUT_PATH):
    """Export selected stimuli for experimental use"""

    all_stimuli = []

    for domain, conditions in stimuli_dict.items():
        for agreement_level, df_subset in conditions.items():
            if len(df_subset) > 0:
                subset_copy = df_subset.copy()
                subset_copy['ssl_domain'] = domain
                subset_copy['agreement_condition'] = agreement_level
                all_stimuli.append(subset_copy)

    if all_stimuli:
        combined_df = pd.concat(all_stimuli, ignore_index=True)

        # Select key columns for experiment
        experiment_cols = [
            'rot', 'action', 'situation', 'ssl_domain', 'agreement_condition',
            'rot-agree', 'action-moral-judgment', 'agreement_label', 'area'
        ]

        export_df = combined_df[experiment_cols]

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

        export_df.to_csv(output_path, index=False)
        log_and_print(f"\nExported {len(export_df)} stimuli to {output_path}")

        return export_df
    else:
        log_and_print("No stimuli to export!")
        return None


def main():
    """Main execution function"""

    log_and_print("Starting SSL Study 2 stimulus selection...")
    log_and_print(f"Parameters: N_LOW={N_LOW}, N_MED={N_MED}, N_HI={N_HI}, OVERSAMPLE_FACTOR={OVERSAMPLE_FACTOR}")

    # Read in the data
    log_and_print(f"Loading data from {SOCIAL_CHEM_PATH}")
    df = pd.read_csv(SOCIAL_CHEM_PATH, sep="\t")
    log_and_print(f"Original dataset size: {len(df)}")

    # Set up profanity filtering
    profane_words_url = "https://raw.githubusercontent.com/coffee-and-fun/google-profanity-words/604bad087123a4ed4425f05d13e119b98e270d30/data/en.txt"
    bad_words = requests.get(profane_words_url).text.split("\n")
    bad_words = [word.strip() for word in bad_words if word.strip()]

    # added custom phrases
    bad_words += [
        "sleeping with",
        "lingerie",
        "make love",
        "one night stand",
        "sex tape",
        "porn star",
        "adult film",
        "meth",
        "heroin",
        "phone sex",
        "cam girl",
        "thong",
        "strip club",
    ]

    wl = {'profanity': bad_words}
    counter = FastFlashTextCounter(wl)

    # Filter out bad words
    df['has_bad_words'] = df['rot'].apply(lambda x: contains_bad_words(x, counter))
    log_and_print(f"Rows with bad words: {df['has_bad_words'].sum()}")

    # Apply enhanced filtering
    df_filtered = enhanced_filter(df)

    # Load ATUS roles and personal words
    log_and_print(f"Loading ATUS roles from {ATUS_ROLES_PATH}")
    atus_roles = text2list(ATUS_ROLES_PATH)
    personal_words = read_json("../data/clean/personal_words.json")[0]['response']
    combined = atus_roles + personal_words

    # Classify SSL domains
    log_and_print("Classifying SSL domains...")
    df_filtered['ssl_domain'] = df_filtered.apply(lambda row: classify_ssl_domain(row, combined), axis=1)

    df_filtered = analyze_norm_concordance(df_filtered)

    log_and_print("Selecting experimental stimuli...")
    experimental_stimuli = select_experimental_stimuli(df_filtered)

    # Print summary statistics
    print_summary_stats(df_filtered, experimental_stimuli)

    # Preview examples from each condition
    log_and_print("\nPreviewing examples...")
    for domain in ['moral', 'societal-conventional', 'personal-psychological']:
        for agreement in ['low_agreement', 'medium_agreement', 'high_agreement']:
            preview_ssl_stimuli(experimental_stimuli, domain, agreement, n=2)

    # Export the selected stimuli
    export_df = export_for_experiment(experimental_stimuli)

    return export_df


if __name__ == "__main__":
    main()