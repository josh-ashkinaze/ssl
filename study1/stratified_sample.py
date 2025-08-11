import pandas as pd

# Load the dataset
data = pd.read_csv("ssl_stimuli_unannot.csv")

# Prepare an empty list to hold selected rows
selected_rows = []

# Get unique domains and agreement levels
domains = data['ssl_domain'].unique()
agreements = data['agreement_condition'].unique()

# Loop through each combination and sample up to 3 rows
for domain in domains:
    for level in agreements:
        subset = data[(data['ssl_domain'] == domain) &
                      (data['agreement_condition'] == level)]
        sampled_rows = subset.sample(n=min(3, len(subset)), random_state=42)
        selected_rows.append(sampled_rows)

# Combine sampled rows
final = pd.concat(selected_rows)

# Keep needed columns for verification
final = final[['rot', 'action', 'situation', 'ssl_domain', 'agreement_condition']]

# Save stratified output
final.to_csv("human_stimuli_stratified.csv", index=False)

print("✅ Stratified 27 samples saved → human_stimuli_stratified.csv")
