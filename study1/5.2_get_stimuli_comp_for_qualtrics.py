"""
Author: Joshua Ashkinaze

Description: For Qualtrics pilot, need to see if better to use Social Chemistry or Normbank stimuli.

Date: 2025-08-23 09:23:35

Input:
- data/clean/ssl_stimuli_unannot.csv: Social Chemistry stimuli with agreement conditions and domains
- data/clean/init_normbank_predictions_3000.csv: Normbank stimuli with predicted domains

Output:
- data/qualtrics_experiments/compare_stimuli_lm_raw.csv: Paired stimuli from both datasets for comparison
- data/qualtrics_experiments/compare_stimuli_lm_clean.csv: This isn't created by this script but manually cleaned for grammar etc.

"""


import pandas as pd
import os

random_state=42
n_per_bucket = 5
os.makedirs("../data/qualtrics_experiments", exist_ok=True)
output_fn = "../data/qualtrics_experiments/compare_stimuli_lm_raw.csv"




# Social chemistry cleaning
#########################
#########################
def fix_s_domain(x):
    x = x.lower()
    if "moral" in x:
        return "moral"
    elif "societal" in x:
        return "societal"
    elif "personal" in x:
        return "personal"

def sentence_case(s):
    return s[0].upper() + s[1:] if s else s

def agreement_label_map(x):
    if x == 'low_agreement':
        return 'taboo'
    elif x == 'medium_agreement':
        return 'normal'
    elif x == 'high_agreement':
        return 'expected'
    else:
        return x

# Normbank cleaning
#########################
#########################


norm_label_map = {
    0: "taboo",
    1: "normal",
    2: "expected"
}

def turn_into_row(x, replace_person=False):
    s = f"{x['behavior']} at a {x['setting']} where {x['constraints']}"
    if replace_person:
        s = s.replace("[PERSON]'s", "your")
        s = s.replace("[OTHER]'s", "the other person's")
        s = sentence_case(s)
    else:
        pass
    return s


#######################################
# clean normbank stimuli
n = pd.read_csv("../data/clean/init_normbank_predictions_3000.csv")
n['norm_label'] = n['label'].map(norm_label_map)
n['stimulus'] = n.apply(turn_into_row, axis=1)
n = n.query("pred != 'Other'")
n['domain'] = n['pred'].apply(fix_s_domain)


# clean social chemistry stimuli
s = pd.read_csv("../data/clean/ssl_stimuli_unannot.csv")
s["domain"] = s["ssl_domain"].apply(fix_s_domain)
s['stimulus'] = s['rot'].apply(lambda x: sentence_case(x.strip()))
s['norm_label'] = s['agreement_condition'].apply(agreement_label_map)
#######################################



# assert they're aligned
###########################
assert set(s['domain'].unique()) == set(n['domain'].unique()), "Domains do not match between datasets"
assert set(s['norm_label'].unique()) == set(n['norm_label'].unique()), "Norm labels do not match between datasets"

print(s['norm_label'].value_counts())
print(n['norm_label'].value_counts())
###########################

# create one df
############################
s = s[['stimulus', 'domain', 'norm_label']]
s['dataset'] = 'social_chemistry'
n = n[['stimulus', 'domain', 'norm_label']]
n['dataset'] = 'normbank'
combined = pd.concat([s, n], ignore_index=True)
############################

# Create a dataframe of paired stimuli
############################
import pandas as pd

pairs = []
for d in combined['domain'].unique():
    for nl in combined['norm_label'].unique():
        # Filter rows for the current domain and norm label for each dataset
        social_subset = combined[(combined['dataset'] == 'social_chemistry') &
                                 (combined['domain'] == d) &
                                 (combined['norm_label'] == nl)]
        normbank_subset = combined[(combined['dataset'] == 'normbank') &
                                   (combined['domain'] == d) &
                                   (combined['norm_label'] == nl)]

        n_samples = min(n_per_bucket, len(social_subset), len(normbank_subset))
        if n_samples > 0:
            social_sample = social_subset.sample(n=n_samples, random_state=random_state)
            normbank_sample = normbank_subset.sample(n=n_samples, random_state=random_state)
            # Create one‑to‑one pairs with a simple loop
            for i in range(n_samples):
                pairs.append({
                    'social_chemistry_stimulus': social_sample.iloc[i]['stimulus'],
                    'normbank_stimulus': normbank_sample.iloc[i]['stimulus'],
                    'domain': d,
                    'norm_label': nl
                })

pairs_df = pd.DataFrame(pairs)
pairs_df['idx'] = range(1, len(pairs_df) + 1)
pairs_df.to_csv(output_fn, index=False)

# need to clean some of the texts for grammar etc
pairs_df.to_csv(output_fn.replace("raw", "clean"), index=False)

# sample some of these
for idx, row in pairs_df.sample(5, random_state=42).iterrows():
    print(f"Idx: {row['idx']}")
    print(f"Domain: {row['domain']}, Norm label: {row['norm_label']}")
    print(f" Social Chemistry Stimulus: {row['social_chemistry_stimulus']}")
    print(f" Normbank Stimulus: {row['normbank_stimulus']}")
    print("-----")

print(pairs_df.head())
print(len(pairs_df))
############################




