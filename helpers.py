import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import random
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import re

random.seed(42)
np.random.seed(42)

import seaborn as sns
import matplotlib.pyplot as plt


def make_aesthetic(
    hex_color_list=None,
    with_gridlines=False,
    bold_title=False,
    save_transparent=False,
    font_scale=2,
    custom_font="Arial",
):
    """
    Make Seaborn look clean and add space between title and plot

    Args:
        hex_color_list: List of hex colors for the palette
        with_gridlines: Whether to add gridlines
        bold_title: Whether to make the title bold
        save_transparent: Whether to save the plot with a transparent background
        font_scale: Scaling factor for font sizes
        custom_font: Font to use for the plot. Note:
            There is a very annoying thing where if you try this in Google Colab with, say,
            'Arial' but colab does not have Arial it will keep printing a warning about this,
            so I would recommend using 'Arial' only if you are sure it is available on the system.

    """

    # Note: To make some parts of title bold and others not bold, we have to use
    # latex rendering. This should work:
    # plt.title(r'$\mathbf{bolded\ title}$' + '\n' + 'And a non-bold subtitle')

    sns.set(style="white", context="paper", font_scale=font_scale)
    if not hex_color_list:
        # 2024-11-28: Reordered color list
        hex_color_list = [
            "#2C3531",  # Dark charcoal gray with green undertone
            "#D41876",  # Telemagenta
            "#00A896",  # Persian green
            "#826AED",  # Medium slate blue
            "#F45B69",  # Vibrant pinkish-red
            "#E3B505",  # Saffron
            "#89DAFF",  # Pale azure
            "#342E37",  # Dark grayish-purple
            "#7DCD85",  # Emerald
            "#F7B2AD",  # Melon
            "#D4B2D8",  # Pink lavender
            "#020887",  # Phthalo blue
            "#E87461",  # Medium-bright orange
            "#7E6551",  # Coyote
            "#F18805",  # Tangerine
        ]

    sns.set_palette(sns.color_palette(hex_color_list))

    plt.rcParams.update(
        {
            # font settings
            "font.family": custom_font,
            "font.weight": "regular",
            "axes.labelsize": 11 * font_scale,
            "axes.titlesize": 14 * font_scale,
            "xtick.labelsize": 10 * font_scale,
            "ytick.labelsize": 10 * font_scale,
            "legend.fontsize": 10 * font_scale,
            # spines/grids
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,  # Thinner spines
            "axes.grid": with_gridlines,
            "grid.alpha": 0.2,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            # title
            "axes.titlelocation": "left",
            "axes.titleweight": "bold" if bold_title else "regular",
            "axes.titlepad": 15 * (font_scale / 1),
            # fig
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.2,
            "figure.constrained_layout.w_pad": 0.2,
            # legend
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
            "legend.borderpad": 0.4,
            "legend.borderaxespad": 1.0,
            "legend.handlelength": 1.5,
            "legend.handleheight": 0.7,
            "legend.handletextpad": 0.5,
            # export
            "savefig.dpi": 300,
            "savefig.transparent": save_transparent,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
            "figure.autolayout": False,
            # math font
            "mathtext.fontset": "custom",
            "mathtext.rm": custom_font,
            "mathtext.it": f"{custom_font}:italic",
            "mathtext.bf": f"{custom_font}:bold",
        }
    )

    return hex_color_list


class SBERTClusterer():
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

    def __init__(self, sbert_model="all-MiniLM-L6-v2"):
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
            plt.plot(k_values, scores, "bo-")
            plt.xlabel("K (Number of Clusters)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Scores by Number of Clusters")
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

        df = pd.DataFrame({"text": texts, "cluster": self.labels})

        for cluster in sorted(df["cluster"].unique()):
            print(f"\nCluster {cluster}:")
            print(df[df["cluster"] == cluster]["text"].tolist())

            logging.info(f"\nCluster {cluster}:")
            logging.info(str(df[df["cluster"] == cluster]["text"].tolist()))

        return df


clusterer = SBERTClusterer()
texts = ["text1", "text2", "text3"]
clusterer.find_k(texts, k_min=2, k_max=3)
df = clusterer.apply_clustering(texts, k=2)