"""
Author: Joshua Ashkinaze

Description: This script fetches daily level AI newspaper data from MediaCloud

Input:
    - None

Output:
    - ai_social.csv: Daily level AI newspaper data from MediaCloud

Date: 2024-11-27 17:35:05
"""


import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv("../../src/.env")
from  src.mc_wrapper import multiple_query_count_over_time, flatten_list, long2wide, dict_kw_list_over_time
import logging
logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)


MC_API_KEY = os.getenv('MEDIACLOUD_API_KEY')
print("API key loaded", MC_API_KEY)

START_DATE = "2014-11-30"
END_DATE = "2024-11-27"
COLLECTION = 34412234

logging.info("Parameters loaded")
logging.info("Start date: %s", START_DATE)
logging.info("End date: %s", END_DATE)
logging.info("Collection: %s", COLLECTION)


def assign_category(row):
    if any(query in row['query'] for query in topics['ai_social']):
        return 'ai_social'
    else:
        return 'plain'


def extract_core_term(x):
    core_terms = ['ai', 'chatbot', 'artificial intelligence']
    for term in core_terms:
        if term in x:
            return term
    else:
        return None

def extract_social_term(x):
    social_terms = ['advice', 'therapist', 'coach', 'advisor', 'friend']
    for term in social_terms:
        if term in x:
            return term
    else:
        return None



if __name__ == "__main__":

    # 1. Make topics
    ################################
    topics = {
        'ai': ['ai', 'chatbot', 'artificial intelligence'],
        'ai_social': []
    }

    # Social roles
    role_patterns = [
        lambda x: f"'{x} therapist'",
        lambda x: f"'{x} counselor'",
        lambda x: f"'{x} psychologist'",
        lambda x: f"'{x} coach'",
        lambda x: f"'{x} mentor'",
        lambda x: f"'{x} tutor'",
        lambda x: f"'{x} friend'",
        lambda x: f"'{x} companion'",
    ]

    # Advice patterns
    advice_patterns = [
        lambda x: f"'advice from {x}'",
        lambda x: f"'{x}' AND ('advice')",
    ]


    # Combine all patterns
    for pattern_list in [role_patterns, advice_patterns]:
        for pattern in pattern_list:
            topics['ai_social'].extend([pattern(i) for i in topics['ai']])
    
    logging.info(topics)
    

    
    # 2. Run queries
    ################################
    api = MC_API_KEY
    run = dict_kw_list_over_time(kw_dict=topics,
                                 collection=COLLECTION,
                                         start_date=START_DATE,
                                         end_date=END_DATE,
                                         api_key=api,
                                         n_jobs=-1)
    logging.info("got data")
    
    
    
    # 3. Basic Clean data
    ################################
    run['category'] = run.apply(assign_category, axis=1)
    run['is_social'] = run['query'].apply(lambda x:1 if x in topics['ai_social'] else 0)
    run['core_term'] = run['query'].apply(extract_core_term)
    run['social_term'] = run['query'].apply(extract_social_term)
    run.to_csv(f"../../data/raw/{START_DATE}_{END_DATE}_{COLLECTION}_ai_social.csv", index=False)
    logging.info("cleaned data")


    # 4. Long to wide
    ################################
    wide_df = long2wide(run, date_col='date', cat_col='category', count_col='count', how='sum')
    wide_df['date'] = pd.to_datetime(wide_df['date'])
    wide_df['ai_social'] = wide_df['ai_social'].astype(int)
    wide_df['plain'] = wide_df['plain'].astype(int)
    wide_df['social_prop'] = wide_df['ai_social'] / (
            wide_df['ai_social'] + wide_df['plain']
    )
    wide_df.to_csv(f"../../data/clean/wide_{START_DATE}_{END_DATE}_{COLLECTION}_ai_social.csv", index=False)
