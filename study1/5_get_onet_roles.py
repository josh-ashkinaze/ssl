import pandas as pd
import os
import logging
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    left_on="o_net-soc_code",  # Updated column name
    right_on="o_net-soc_code", # Updated column name
    how="inner",
)



social_skills = {
   "Coordination": "Adjusting actions in relation to others' actions.",
   "Instructing": "Teaching others how to do something.",
   "Negotiation": "Bringing others together and trying to reconcile differences.",
   "Persuasion": "Persuading others to change their minds or behavior.",
   "Service Orientation": "Actively looking for ways to help people.",
   "Social Perceptiveness": "Being aware of others' reactions and understanding why they react as they do."
}

social_skills = {
   # "Coordination": "Adjusting actions in relation to others' actions.",
   "Instructing": "Teaching others how to do something.",
   # "Negotiation": "Bringing others together and trying to reconcile differences.",
   # "Persuasion": "Persuading others to change their minds or behavior.",
   # "Service Orientation": "Actively looking for ways to help people.",
   "Social Perceptiveness": "Being aware of others' reactions and understanding why they react as they do."
}


# Filter for social skills
social_df = merged[merged["element_name"].isin(social_skills.keys())].copy()
social_df['data_value'] = social_df['data_value'].astype(float)


# Check what scale types we have
print(f"Scale types: {social_df['scale_id'].unique()}")
print(f"Scale type counts:\n{social_df['scale_id'].value_counts()}")

# Calculate z-scores: standardize each skill by scale type across all occupations
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



N = 40
social_df2_top = sorted(social_df2.sort_values(by=['z_data_value'], ascending=False).head(N)['title'].to_list())
print(f"\nTop {N} occupations with highest z-scores:\n{social_df2_top}")
logging.info(f"Top {N} occupations with highest z-scores:\n{social_df2_top}")

social_df2_bottom = sorted(social_df2.sort_values(by=['z_data_value'], ascending=True).head(N)['title'].to_list())
print(f"\nTop {N} occupations with lowest z-scores:\n{social_df2_bottom}")
logging.info(f"Top {N} occupations with lowest z-scores:\n{social_df2_bottom}")


from sklearn.feature_extraction.text import CountVectorizer
#
# # CLUSTERING
# ##########################
# ##########################
# # Convert texts to TF-IDF features
# vectorizer = CountVectorizer(
#     stop_words="english",
#     lowercase=True,
#     ngram_range=(1, 2),  # Only single words, not bigrams
#     max_df=0.8,
#     min_df=1
# )
# X = vectorizer.fit_transform(social_df2_top)
#
# best_k = 2
# best_score = -1
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X)
#     score = silhouette_score(X, labels)
#     if score > best_score:
#         best_k = k
#         best_score = score
#
# print(f"Best number of clusters (k): {best_k} with silhouette score: {best_score:.4f}")
#
# # Perform clustering with the best k
# kmeans = KMeans(n_clusters=best_k, random_state=42)
# kmeans.fit(X)
# clusters = kmeans.labels_
#
# # Print cluster assignments grouped by cluster
# clustered_texts = {i: [] for i in range(best_k)}
# for i, text in enumerate(social_df2_top):
#     clustered_texts[clusters[i]].append(text)
#
# for cluster, texts in clustered_texts.items():
#     print(f"\nCluster {cluster}:")
#     for text in texts:
#         print(f"  - {text}")



# CLUSTERING - spaCy word vectors
##########################
##########################
import spacy
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load spaCy model (install with: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

def get_title_vector(title):
    """Get average word vector for a job title"""
    doc = nlp(title)
    vectors = [token.vector for token in doc if token.has_vector and not token.is_stop]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(nlp.vocab.vectors_length)

# Get embeddings
embeddings = np.array([get_title_vector(title) for title in social_df2_top])
print(f"Embeddings shape: {embeddings.shape}")

# Rest of clustering code same as before...
best_k = 2
best_score = -1
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    if score > best_score:
        best_k = k
        best_score = score
        best_labels = labels

print(f"Best k: {best_k}, silhouette: {best_score:.4f}")

# Print results
clustered_texts = {i: [] for i in range(best_k)}
for i, text in enumerate(social_df2_top):
    clustered_texts[best_labels[i]].append(text)

for cluster, texts in clustered_texts.items():
    print(f"\nCluster {cluster}:")
    for text in texts:
        print(f"  - {text}")

##########################
##########################

# condense these^
short_list = ["director", "nurse", "executive", "supervisor", "coach", "scout", "counselor", "advisor", "manager", "therapist", "social worker", "administrator", "clergy", "psychiatrist"]
print(f"Short list of occupations: {short_list}")

logging.info(f"Short list of occupations: {short_list}")

# Save the short list to a file
with open("../data/clean/onet_roles.txt", "w") as f:
    for item in short_list:
        f.write(f"{item}\n")


with open("../data/raw/onet_roles_raw.txt", "w") as f:
    for item in social_df2_top:
        f.write(f"{item}\n")