"""
Date: 2025-05-28 12:21:11

Description: Stores WCs for the data pulls

Input files:
- ../data/clean/arxiv_2018-01-01_2025-05-20_cs__.jsonl: Arxiv data with text field
- ../data/clean/nyt_pull_2018-01-01_2025-06-01.jsonl: NYT data with text field
- ../data/clean/producthunt_posts_2019-01-01_to_2025-05-01.json: Product Hunt data with text field

Output files:
- ../data/clean/arxiv_word_counts.json: Word counts for Arxiv data
- ../data/clean/nyt_word_counts.json: Word counts for NYT data
- ../data/clean/prod_hunt_word_counts.json: Word counts for Product Hunt data

Each wc file has the following columns:
- analysis_date: Date of analysis
- unique_idx: Unique identifier for the row
- text: Text field from the original data

    Dict defs
        'ai': ai_terms,
        'social': atus_roles + onet_roles + nouns,
        'ai_compound_roles': ai_compound_roles,
        'ai_compound_nouns': ai_compound_nouns

- {name}_word_counts: Dictionary of word counts for the respective category
- {name}_sum: Sum of word counts for the respective category
- {name}_binary: Binary indicator if {name_sum} > 0

"""



import pandas as pd
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helpers import text2list

from collections import Counter

from flashtext import KeywordProcessor
from collections import Counter
import pandas as pd
import swifter
import os
import logging
import json


logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)


# -1 means no sampling otherwise TO_SAMPLE will be number of rows
# this is because this part can be quite slow
TO_SAMPLE = -1


def list2lower(l):
    """
    Convert a list of strings to lowercase.
    """
    return [x.lower() for x in l]


class FastFlashTextCounter:

    def __init__(self, word_lists_dict):
        self.processors = {}

        for name, word_list in word_lists_dict.items():
            processor = KeywordProcessor(case_sensitive=False)
            for word in word_list:
                processor.add_keyword(word.lower())
            self.processors[name] = processor
        print("FastFlashTextCounter initialized with word lists.")
        logging.info("FastFlashTextCounter initialized with word lists.")

    def count_keywords(self, text, processor_name):
        if pd.isna(text) or not text:
            return {}

        keywords_found = self.processors[processor_name].extract_keywords(str(text).lower())
        return dict(Counter(keywords_found))


def main():

    sample_str = "" if TO_SAMPLE == -1 else f"_sample_{TO_SAMPLE}_"
    arxiv_wc_fn = f"../data/clean/arxiv_word_counts{sample_str}.json"
    nyt_wc_fn = f"../data/clean/nyt_word_counts{sample_str}.json"
    prod_hunt_wc_fn = f"../data/clean/prod_hunt_word_counts{sample_str}.json"
    #
    # if os.path.exists(arxiv_wc_fn) and os.path.exists(nyt_wc_fn) and os.path.exists(prod_hunt_wc_fn) and TO_SAMPLE == -1:
    #     print("Word counts files already exist. Exiting to avoid overwriting.")
    #     logging.info("Word counts files already exist. Exiting to avoid overwriting.")
    #     return None
    #
    # else:
    #     pass

    arxiv_fn = "../data/clean/arxiv_2018-01-01_2025-06-01_cs.jsonl"
    nyt_fn = "../data/clean/nyt_pull_2018-01-01_2025-06-01.jsonl"
    prod_hunt_fn = "../data/clean/producthunt_posts_2018-01-01_to_2025-06-01.jsonl"


    # Lists of terms
    ai_terms = list2lower(text2list("../data/clean/ai_terms.txt"))
    atus_roles = list2lower(text2list("../data/clean/atus_roles.txt"))
    onet_roles = list2lower(text2list("../data/clean/onet_roles.txt"))
    nouns = list2lower(pd.read_json("../data/clean/common_nouns.json")['nouns'].tolist())

    ai_compound_roles = list2lower(text2list("../data/clean/ai_compound_roles.txt"))
    ai_compound_nouns = list2lower(text2list("../data/clean/ai_compound_nouns.txt"))


    # read in data
    arxiv_df = pd.read_json(arxiv_fn, lines=True)
    arxiv_df['text'] = (arxiv_df['title'].fillna('') + " " + arxiv_df['abstract'].fillna('')).str.lower()
    arxiv_df['wc'] = arxiv_df['text'].swifter.apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    nyt_df = pd.read_json(nyt_fn, lines=True)
    nyt_df['text'] = (
            nyt_df['main_headline'].fillna('') + " " +
            nyt_df['abstract'].fillna('') + " " +
            nyt_df['snippet'].fillna('')
    ).str.lower()
    nyt_df['wc'] = nyt_df['text'].swifter.apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    prod_df = pd.read_json(prod_hunt_fn, lines=True)
    prod_df['text'] = (
                prod_df['name'].fillna('') + " " + prod_df['tagline'].fillna('') + " " + prod_df['description'].fillna(
            '')).str.lower()
    prod_df['wc'] = prod_df['text'].swifter.apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    if TO_SAMPLE != -1:
        arxiv_df = arxiv_df.sample(TO_SAMPLE, random_state=42)
        nyt_df = nyt_df.sample(TO_SAMPLE, random_state=42)
        prod_df = prod_df.sample(TO_SAMPLE, random_state=42)
        print(f"Sampled {TO_SAMPLE} rows from datasets.")
        logging.info(f"Sampled {TO_SAMPLE} rows from datasets.")

    else:
        pass

    # define word lists for the fast wc class
    word_lists = {
        'base-ai': list2lower(["chatbot", "artificial intelligence", "A.I", "ai"]),
        'ai': ai_terms,
        'social': atus_roles + onet_roles + nouns,
        'ai_compound_roles': ai_compound_roles,
        'ai_compound_nouns': ai_compound_nouns,
        'ai_compound': ai_compound_roles + ai_compound_nouns
    }
    counter = FastFlashTextCounter(word_lists)

    for name, word_list in word_lists.items():
        print(f"Counting keywords for '{name}'...")
        arxiv_df[f'{name}_word_counts'] = arxiv_df['text'].swifter.apply(lambda x: counter.count_keywords(x, name))
        arxiv_df[f'{name}_sum'] = arxiv_df[f'{name}_word_counts'].swifter.apply(lambda x: sum(x.values()))
        arxiv_df[f'{name}_binary'] = arxiv_df[f'{name}_sum'].apply(lambda x: 1 if x > 0 else 0)

        nyt_df[f'{name}_word_counts'] = nyt_df['text'].swifter.apply(lambda x: counter.count_keywords(x, name))
        nyt_df[f'{name}_sum'] = nyt_df[f'{name}_word_counts'].swifter.apply(lambda x: sum(x.values()))
        nyt_df[f'{name}_binary'] = nyt_df[f'{name}_sum'].apply(lambda x: 1 if x > 0 else 0)

        prod_df[f'{name}_word_counts'] = prod_df['text'].swifter.apply(lambda x: counter.count_keywords(x, name))
        prod_df[f'{name}_sum'] = prod_df[f'{name}_word_counts'].swifter.apply(lambda x: sum(x.values()))
        prod_df[f'{name}_binary'] = prod_df[f'{name}_sum'].apply(lambda x: 1 if x > 0 else 0)

        print(f"Keyword counts for '{name}' completed.")


    # save subsetted data
    wc_cols = [col for col in arxiv_df.columns if col.endswith('_word_counts')]
    sum_cols = [col for col in arxiv_df.columns if col.endswith('_sum')]
    binary_cols = [col for col in arxiv_df.columns if col.endswith('_binary')]

    arxiv_df[wc_cols + sum_cols + binary_cols  + ['analysis_date', 'unique_idx', 'text']].to_json(arxiv_wc_fn, orient='records', lines=True)
    nyt_df[wc_cols + sum_cols + binary_cols + ['analysis_date', 'unique_idx', 'text']].to_json(nyt_wc_fn, orient='records', lines=True)
    prod_df[wc_cols + sum_cols + binary_cols + ['analysis_date', 'unique_idx', 'text']].to_json(prod_hunt_wc_fn, orient='records', lines=True)

    print("done")
    logging.info("done")


if __name__ == "__main__":
    main()