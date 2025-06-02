"""
Date: 2025-05-29 10:09:06

Description: In case we need to use MediaCloud, we need "random" articles but they don't support random search.
We can do a search which is (w1 OR w2 OR w3) for the most common words in English language. But we'd want to know
coverage in a comparable corpus. So here we are just going to see how many NYT headlines contain at least one of the
20 most common words in English language.

Input files:
- None

Output files:
- None
"""

from statsmodels.stats.proportion import proportion_confint




import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashtext import KeywordProcessor
from collections import Counter
import swifter
import pandas as pd

from src.helpers import array_stats
import logging

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)


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
        print("FastFlashTextCounter initialized")

    def count_keywords(self, text, processor_name):
        if pd.isna(text) or not text:
            return {}
        keywords_found = self.processors[processor_name].extract_keywords(str(text).lower())
        return dict(Counter(keywords_found))


nyt = pd.read_json("../data/clean/nyt_pull_2018-01-01_2025-06-01.jsonl", lines=True, encoding_errors='ignore',
                   encoding="utf8")

# https://web.archive.org/web/20111226085859/http://oxforddictionaries.com/words/the-oec-facts-about-the-language
common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he',
                'as', 'you', 'do', 'at']

common_words = list2lower(common_words)

wl = {'common': common_words}


# Register with FlashText
counter = FastFlashTextCounter(wl)
nyt['common_wc'] = nyt['main_headline'].swifter.apply(lambda x: counter.count_keywords(x, 'common'))
nyt['common_wc_sum'] = nyt['common_wc'].swifter.apply(lambda x: sum(x.values()) if isinstance(x, dict) else 0)
nyt['common_wc_binary'] = nyt['common_wc_sum'].swifter.apply(lambda x: 1 if x > 0 else 0)

sum_cols = ['common_wc_sum', 'common_wc_binary']
print(nyt[sum_cols].describe())
logging.info(f"NYT common word counts:\n{nyt[sum_cols].describe()}")

sum_str = array_stats(nyt['common_wc_sum'].values, include_ci=False)
print(f"Common word counts summary: {sum_str}")
logging.info(f"Common word counts summary: {sum_str}")

binary_str = array_stats(nyt['common_wc_binary'].values, include_ci=False)
print(f"Common word counts binary summary: {binary_str}")
logging.info(f"Common word counts binary summary: {binary_str}")


# CI
total_articles = len(nyt)
articles_with_common_word = nyt['common_wc_binary'].sum()
point_estimate = articles_with_common_word / total_articles
conf_int = proportion_confint(articles_with_common_word, total_articles, alpha=0.05, method='wilson')
print(f"Proportion of articles with at least one common word: {point_estimate:.2f}")
print(f"95% confidence interval: {conf_int[0]:.2f} - {conf_int[1]:.2f}")

logging.info(f"Proportion of articles with at least one common word: {point_estimate:.2f}")
logging.info(f"95% confidence interval: {conf_int[0]:.2f} - {conf_int[1]:.2f}")


