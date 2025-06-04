import nltk
import json
import random
import pandas as pd
import numpy as np
from nltk.corpus import wordnet, brown, reuters
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from scipy import stats

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Download the 4 most relevant and largest corpora
print("Downloading most relevant corpora for social learning...")
corpora_to_download = ['wordnet', 'brown', 'reuters', 'treebank', 'webtext']

for corpus in corpora_to_download:
    nltk.download(corpus, quiet=True)

# Load SBERT model
print("Loading SBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Build frequency database from 4 most relevant corpora
print("Building frequency database from 4 most relevant corpora...")
all_words = []

# 1. REUTERS - News/business (most relevant to social learning contexts)
print("Adding Reuters corpus (news/business - most relevant)...")
from nltk.corpus import reuters

reuters_words = [w.lower() for w in reuters.words() if w.isalpha() and len(w) > 2]
all_words.extend(reuters_words)
print(f"  Added {len(reuters_words):,} words from Reuters")

# 2. BROWN - Balanced general corpus (largest balanced corpus)
print("Adding Brown corpus (balanced general text)...")
from nltk.corpus import brown

brown_words = [w.lower() for w in brown.words() if w.isalpha() and len(w) > 2]
all_words.extend(brown_words)
print(f"  Added {len(brown_words):,} words from Brown")

# 3. WEBTEXT - Modern informal text (online learning contexts)
print("Adding WebText corpus (modern informal text)...")
from nltk.corpus import webtext

webtext_words = [w.lower() for w in webtext.words() if w.isalpha() and len(w) > 2]
all_words.extend(webtext_words)
print(f"  Added {len(webtext_words):,} words from WebText")

# 4. TREEBANK - Academic/formal text (relevant to educational contexts)
print("Adding Treebank corpus (academic/formal text)...")
from nltk.corpus import treebank

treebank_words = [w.lower() for w in treebank.words() if w.isalpha() and len(w) > 2]
all_words.extend(treebank_words)
print(f"  Added {len(treebank_words):,} words from Treebank")

print(f"Total words collected: {len(all_words):,}")
WORD_FREQ = Counter(all_words)
print(f"Unique words in frequency database: {len(WORD_FREQ):,}")


def get_word_frequency(word):
    """Get frequency of word."""
    return WORD_FREQ.get(word.lower(), 1)


def calculate_similarity(word, target_phrase):
    """Calculate SBERT similarity."""
    embeddings = sbert_model.encode([word, target_phrase])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)


def get_frequency_matched_nouns(my_nouns, n):
    """Get frequency-matched control nouns."""

    # Get frequencies of my nouns
    my_frequencies = [get_word_frequency(noun) for noun in my_nouns]
    lower = np.percentile(my_frequencies, 40)
    upper = np.percentile(my_frequencies, 60)

    print(f"My nouns frequency range: {lower:.0f} - {upper:.0f}")

    # Find control nouns in frequency range
    control_candidates = []
    for word, freq in WORD_FREQ.items():
        if lower <= freq <= upper:
            synsets = wordnet.synsets(word)
            if any(syn.pos() == 'n' for syn in synsets) and word not in my_nouns:
                control_candidates.append(word)

    # Sample control nouns
    control_nouns = random.sample(control_candidates, n)
    print(f"Selected {n} control nouns")

    return control_nouns


def create_dataframe(my_nouns, control_nouns, target_phrase):
    """Create dataframe with similarity metrics."""

    data = []

    # Process my nouns
    for noun in my_nouns:
        similarity = calculate_similarity(noun, target_phrase)

        data.append({
            'term': noun,
            'bucket': 'my_words',
            'similarity': similarity
        })

    # Process control nouns
    for noun in control_nouns:
        similarity = calculate_similarity(noun, target_phrase)

        data.append({
            'term': noun,
            'bucket': 'control_words',
            'similarity': similarity
        })

    return pd.DataFrame(data)


def run_t_test(df):
    """Run t-test and calculate effect size for similarity."""

    my_words = df[df['bucket'] == 'my_words']
    control_words = df[df['bucket'] == 'control_words']

    my_vals = my_words['similarity'].values
    control_vals = control_words['similarity'].values

    # T-test
    t_stat, p_val = stats.ttest_ind(my_vals, control_vals)

    # Cohen's d
    pooled_std = np.sqrt(((len(my_vals) - 1) * np.var(my_vals, ddof=1) +
                          (len(control_vals) - 1) * np.var(control_vals, ddof=1)) /
                         (len(my_vals) + len(control_vals) - 2))
    cohens_d = (np.mean(my_vals) - np.mean(control_vals)) / pooled_std

    # Degrees of freedom
    df_val = len(my_vals) + len(control_vals) - 2

    return {
        't_stat': t_stat,
        'p_val': p_val,
        'df': df_val,
        'my_mean': np.mean(my_vals),
        'my_std': np.std(my_vals, ddof=1),
        'control_mean': np.mean(control_vals),
        'control_std': np.std(control_vals, ddof=1),
        'cohens_d': cohens_d
    }


def print_results(results):
    """Print formatted t-test results."""

    print("STATISTICAL RESULTS")
    print(f"\nSIMILARITY:")
    print(f"t({results['df']:.0f}) = {results['t_stat']:.2f}, p = {results['p_val']:.2e}")
    print(f"My words: M = {results['my_mean']:.2f}, SD = {results['my_std']:.2f}")
    print(f"Control words: M = {results['control_mean']:.2f}, SD = {results['control_std']:.2f}")
    print(f"Cohen's d = {results['cohens_d']:.2f}")



def main():
    target_phrase = "social learning"

    # Load my nouns
    with open("../data/clean/common_nouns.json", 'r') as f:
        data = json.load(f)
        my_nouns = data['nouns']

    print(f"Loaded {len(my_nouns)} nouns: {my_nouns}")

    # Get frequency-matched control nouns
    control_nouns = get_frequency_matched_nouns(my_nouns, 200)
    print(f"Control nouns: {control_nouns}")

    # Create dataframe
    print(f"\nCalculating similarity for target: '{target_phrase}'...")
    df = create_dataframe(my_nouns, control_nouns, target_phrase)

    # Save dataframe
    df.to_csv("noun_similarity_results.csv", index=False)
    print(f"Results saved to: noun_similarity_results.csv")

    # Print dataframe
    print(f"\nDATAFRAME:")
    print(df.round(3))

    # Run statistical test
    results = run_t_test(df)
    print_results(results)

    return df, results


if __name__ == "__main__":
    df, results = main()