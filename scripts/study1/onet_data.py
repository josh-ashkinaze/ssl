"""
Author: Joshua Ashkinaze

Description: This script read in ONET skills data and then finds those occupations with the highest importance for two relevant skills.

Input:
    - onet_skills.csv: ONET skills data

Output:
    - clean/condensed_list_occs.txt: List of condensed occupations with the highest importance for two relevant skills. It's condensed
    because the full list is more specific (eg: lists teacher a bunch of times).

    - raw/list_occs.txt: List of occupations with the highest importance for two relevant skills

Date: 2024-11-28 10:39:52
"""

import pandas as pd
import logging
import os
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd


logging.basicConfig(
    filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    filemode="w",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


class SBERTClusterer:
    """
    A class for clustering texts using SBERT embeddings and KMeans.

    Args:
        model: SBERT model for text embeddings
        embeddings: Generated embeddings for input texts
        best_k: Best number of clusters found
        labels: Cluster assignments for texts
        scores: Silhouette scores for different k values

    Example:
        clusterer = SBERTClusterer()
        texts = ["text1", "text2", "text3"]
        clusterer.find_k(texts, k_min=2, k_max=3)
        df = clusterer.apply_clustering(texts, k=2)
    """

    def __init__(self, sbert_model='all-MiniLM-L6-v2'):
        """Initialize with specified SBERT model."""
        self.model = SentenceTransformer(sbert_model)
        self.embeddings = None
        self.best_k = None
        self.labels = None
        self.scores = None

    def _embed_texts(self, texts):
        """Generate embeddings for input texts."""
        self.embeddings = self.model.encode(texts)
        return self.embeddings

    def find_k(self, texts, k_min=2, k_max=10, plot=True):
        """
        Find optimal k using silhouette scores.

        Args:
            texts: List of strings to cluster
            k_min: Minimum number of clusters to try
            k_max: Maximum number of clusters to try
            plot: Whether to plot silhouette scores

        Returns:
            Best k value found
        """
        self._embed_texts(texts)
        scores = []
        k_values = range(k_min, min(len(texts), k_max + 1))

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.embeddings)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(self.embeddings, labels)
                scores.append(score)
            else:
                scores.append(-1)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, scores, 'bo-')
            plt.xlabel('K (Number of Clusters)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Scores by Number of Clusters')
            plt.grid(True)
            plt.show()

        self.best_k = k_values[np.argmax(scores)]
        self.scores = scores
        print(f"Best k: {self.best_k} with silhouette score: {max(scores):.3f}")
        logging.info(f"Best k: {self.best_k} with silhouette score: {max(scores):.3f}")

        return self.best_k

    def apply_clustering(self, texts, k=None):
        """
        Apply KMeans clustering with specified k.

        Args:
            texts: List of strings to cluster
            k: Number of clusters (uses best_k if None)

        Returns:
            DataFrame with columns [text, cluster]
        """

        if self.embeddings is None:
            self._embed_texts(texts)

        if k is None:
            if self.best_k is None:
                raise ValueError("Must either specify k or run find_k first")
            k = self.best_k

        kmeans = KMeans(n_clusters=k, random_state=42)
        self.labels = kmeans.fit_predict(self.embeddings)

        df = pd.DataFrame({
            'text': texts,
            'cluster': self.labels
        })

        for cluster in sorted(df['cluster'].unique()):
            print(f"\nCluster {cluster}:")
            print(df[df['cluster'] == cluster]['text'].tolist())

            logging.info(f"\nCluster {cluster}:")
            logging.info(str(df[df['cluster'] == cluster]['text'].tolist()))

        return df



def topn_by_occ(df, skill, topn):
    """
    Gets top occupations for a given skill in ONNET.

    Args:
        df: Dataframe with ONET skills data
        skill: Skill to search for
        topn: Number of top occupations to return

    Returns:
        List of top occupations by the data value for the skills
    """
    tdf = df[df["element_name"] == skill]
    occs = (
        tdf.sort_values(by=["data_value"], ascending=False)
        .head(topn)["title"]
        .to_list()
    )
    return occs


def list2text_file(lst, filename):
    """
    Write a list to a text file.

    Args:
        lst: List of items to write
        filename: Output filename (adds .txt if needed)

    Returns:
        Nothing but it does write the file
    """
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    content = '\n'.join(str(item) for item in lst)
    with open(filename, 'w') as f:
        f.write(content)


if __name__ == "__main__":

    # 1. Get occupations
    #############################
    df = pd.read_csv("../../data/raw/onet_skills.csv")
    df.columns = [x.lower().replace(" ", "_") for x in df.columns]
    df = df[df["scale_name"] == "Importance"]
    skills_unique = df["element_name"].unique()
    logging.info(str(sorted(skills_unique)))

    top_occs = []
    target_skills = ["Social Perceptiveness", "Instructing"]
    topn = 5

    logging.info("taret skills")
    logging.info(str(target_skills))
    logging.info("topn :" + str(topn))

    for target_skill in target_skills:
        occs = topn_by_occ(df, target_skill, topn)
        top_occs.extend(occs)

    logging.info("Top occupations for skills")
    logging.info(str(top_occs))

    list2text_file(top_occs, "../../data/raw/list_occs.txt")

    condensed_list = [
        "teacher",
        "scout",
        "coach",
        "counselor",
        "therapist",
        "psychologist",
    ]
    list2text_file(condensed_list, "../../data/clean/condensed_list_occs.txt")


    # 2. Get activities
    #############################
    df = pd.read_excel("../../data/raw/Work Activities.xlsx")
    df['rel'] = df['Title'].apply(lambda x: 1 if x in top_occs else 0)
    df['data_value'] = df['Data Value']
    df = df.query("rel==1")
    n = 15
    top = df.groupby(by=['Element Name'])['data_value'].sum().sort_values(ascending=False).head(n).reset_index()[
        'Element Name'].to_list()
    list2text_file(top, "../../data/raw/list_activities.txt")
    clusterer = SBERTClusterer()
    best_k = clusterer.find_k(top, k_min=1, k_max=5)
    clusterer.apply_clustering(top)

