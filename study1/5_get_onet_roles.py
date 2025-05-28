
"""
Date: 2025-05-28 10:22:42

Description: Fetches ONET roles involved in social learning

Input files:
- https://www.onetcenter.org/dl_files/database/db_29_3_text/Skills.txt: ONET skills
- https://www.onetcenter.org/dl_files/database/db_29_3_text/Occupation%20Data.txt: ONET titles

Output files:
- ../data/clean/onet_roles.txt: Cleaned ONET roles involved in social learning
- ../data/raw/onet_roles_raw.txt: Raw ONET roles involved in social learning
"""


import pandas as pd
import os
import logging

from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import random

np.random.seed(42)
random.seed(42)

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

# Load and clean skills data
skills_df = pd.read_csv("https://www.onetcenter.org/dl_files/database/db_29_3_text/Skills.txt", sep="\t")
skills_df.columns = [x.lower().replace(" ", "_").replace("*", "_") for x in skills_df.columns]
skills_df = skills_df[skills_df["recommend_suppress"] == "N"]
print("Skills columns:", skills_df.columns.tolist())

# Load and clean titles data
titles_df = pd.read_csv("https://www.onetcenter.org/dl_files/database/db_29_3_text/Occupation%20Data.txt", sep="\t")
titles_df.columns = [x.lower().replace(" ", "_").replace("*", "_") for x in titles_df.columns]
print("Titles columns:", titles_df.columns.tolist())

# Merge datasets
merged = skills_df.merge(
    titles_df,
    left_on="o_net-soc_code",
    right_on="o_net-soc_code",
    how="inner",
)



# social_skills = {
#    "Coordination": "Adjusting actions in relation to others' actions.",
#    "Instructing": "Teaching others how to do something.",
#    "Negotiation": "Bringing others together and trying to reconcile differences.",
#    "Persuasion": "Persuading others to change their minds or behavior.",
#    "Service Orientation": "Actively looking for ways to help people.",
#    "Social Perceptiveness": "Being aware of others' reactions and understanding why they react as they do."
# }

social_skills = {
   # "Coordination": "Adjusting actions in relation to others' actions.",
   "Instructing": "Teaching others how to do something.",
   # "Negotiation": "Bringing others together and trying to reconcile differences.",
   # "Persuasion": "Persuading others to change their minds or behavior.",
   # "Service Orientation": "Actively looking for ways to help people.",
   "Social Perceptiveness": "Being aware of others' reactions and understanding why they react as they do."
}


social_df = merged[merged["element_name"].isin(social_skills.keys())].copy()
social_df['data_value'] = social_df['data_value'].astype(float)


print(f"Scale types: {social_df['scale_id'].unique()}")
print(f"Scale type counts:\n{social_df['scale_id'].value_counts()}")

# z-scores: standardize each skill by scale type across all occupations
# So the logic is that for a given activity (e.g: foo), for a given scale type (e.g: 'Importance'),
# we get the z score for that activity across all occupations for that scale type.
social_df['z_data_value'] = social_df.groupby(['element_name', 'scale_id'])['data_value'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
)

print(f"\nSocial skills data shape: {social_df.shape}")
print(f"Unique skills found: {social_df['element_name'].unique()}")
print(f"Unique occupations: {social_df['o_net-soc_code'].nunique()}")

social_df2 = social_df.groupby(by=['title', 'o_net-soc_code']).agg(
    z_data_value=('z_data_value', 'mean'),
    data_value=('data_value', 'mean')
).reset_index()



N = 30
social_df2_top = sorted(social_df2.sort_values(by=['z_data_value'], ascending=False).head(N)['title'].to_list())
print(f"\nTop {N} occupations with highest z-scores:\n{social_df2_top}")
logging.info(f"Top {N} occupations with highest z-scores:\n{social_df2_top}")

social_df2_bottom = sorted(social_df2.sort_values(by=['z_data_value'], ascending=True).head(N)['title'].to_list())
print(f"\nTop {N} occupations with lowest z-scores:\n{social_df2_bottom}")
logging.info(f"Top {N} occupations with lowest z-scores:\n{social_df2_bottom}")


# CLUSTERING
##################################
##################################
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(social_df2_top)

best_k = 2
best_score = -1

print("Testing different numbers of clusters:")
for k in range(3, 12):
    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward', )
    labels = clustering.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    print(f"k={k}: silhouette={score:.4f}")
    if score > best_score:
        best_k = k
        best_score = score

print(f"\nUsing k={best_k} clusters")

clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
clusters = clustering.fit_predict(embeddings)

clustered_texts = {i: [] for i in range(best_k)}
for i, text in enumerate(social_df2_top):
    clustered_texts[clusters[i]].append(text)

for cluster, texts in clustered_texts.items():
    print(f"\nCluster {cluster} ({len(texts)} occupations):")
    for text in sorted(texts):
        print(f"  - {text}")




short_list = ['doctor', 'therapist', 'psychologist', 'counselor', 'clergy', 'psychiatrist', 'teacher', 'administrator', 'counselor', 'advisor', 'coach', 'scout', 'social worker', 'HR manager', 'HR specialist']
print(f"Short list of occupations: {short_list}")
print(f"Number of occupations in short list: {len(short_list)}")

logging.info(f"Short list of occupations: {short_list}")
logging.info(f"Number of occupations in short list: {len(short_list)}")
##################################
##################################


# Save stuff
##################################
##################################
with open("../data/clean/onet_roles.txt", "w") as f:
    for item in short_list:
        f.write(f"{item}\n")


with open("../data/raw/onet_roles_raw.txt", "w") as f:
    for item in social_df2_top:
        f.write(f"{item}\n")

##################################
##################################