"""
Author: Joshua Ashkinaze

Description: This script fetches daily level AI newspaper data from MediaCloud

Input:
    - None

Output:
    - Daily counts of AI stories

Date: 2024-11-27 17:35:05
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv("../../src/.env")
from src.mc_wrapper import (
    multiple_query_count_over_time,
    flatten_list,
    long2wide,
    dict_kw_list_over_time,
)
import logging

logging.basicConfig(
    filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    filemode="w",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


MC_API_KEY = os.getenv("MEDIACLOUD_API_KEY")
print("API key loaded", MC_API_KEY)

START_DATE = "2020-01-01"
END_DATE = "2024-11-29"
COLLECTION = 34412234

logging.info("Parameters loaded")
logging.info("Start date: %s", START_DATE)
logging.info("End date: %s", END_DATE)
logging.info("Collection: %s", COLLECTION)


def assign_category(row):
    if any(query in row["query"] for query in TOPICS["ai_social"]):
        return "ai_social"
    else:
        return "plain"


def extract_core_term(x):
    core_terms = TOPICS["ai"]
    core_terms = [i.lower() for i in core_terms]
    x = x.lower()
    for term in core_terms:
        if term in x:
            return term
    else:
        return None


def extract_social_term(x):
    for term in TOPICS["social_terms"]:
        if term in x:
            return term
    else:
        return None


if __name__ == "__main__":

    # 1. Make TOPICS
    ################################
    TOPICS = {
        "ai": [
            "ai",
            "a.i.",
            "A.I.",
            "chatbot",
            "Chatbot",
            "artificial intelligence",
            "Artificial Intelligence",
            "Artificial intelligence",
        ],
        "social_terms": [
            "therapist",
            "psychologist",
            "counselor",
            "coach",
            "scout",
            "teacher",
        ],
        "social_actions": ["advice",
                           "coaching",
                           "emotional support",
                           "social support",
                           ],
        "ai_social": [],
    }

    for ai in TOPICS["ai"]:
        for social in TOPICS["social_terms"]:
            TOPICS["ai_social"].append(f"'{ai} {social}'")
            TOPICS["ai_social"].append(f"'{ai}-powered {social}'")
            TOPICS["ai_social"].append(f"'{ai}-based {social}'")
            TOPICS["ai_social"].append(f"'{ai}-generated {social}'")


        for action in TOPICS["social_actions"]:
            TOPICS["ai_social"].append(f"'{action} from {ai}'")
            TOPICS["ai_social"].append(f"'{ai}-powered {action}'")
            TOPICS["ai_social"].append(f"'{ai}-based {action}'")
            TOPICS["ai_social"].append(f"'{ai}-generated {action}'")



    logging.info(str(TOPICS))

    for key, value in TOPICS.items():
        logging.info(f"Lists and lengths...{key}: {len(value)}")

    # 2. Run queries
    ################################
    api = MC_API_KEY
    run = dict_kw_list_over_time(
        kw_dict=TOPICS,
        collection=COLLECTION,
        start_date=START_DATE,
        end_date=END_DATE,
        api_key=api,
        n_jobs=-1,
    )
    logging.info("got data")

    # 3. Basic Clean data
    ################################
    run["category"] = run.apply(assign_category, axis=1)
    run["is_social"] = run["query"].apply(
        lambda x: 1 if x in TOPICS["ai_social"] else 0
    )
    run["core_term"] = run["query"].apply(extract_core_term)
    run["social_term"] = run["query"].apply(extract_social_term)
    run.to_csv(
        f"../../data/raw/{START_DATE}_{END_DATE}_{COLLECTION}_ai_social.csv",
        index=False,
    )
    logging.info("cleaned data")

    # 4. Long to wide
    ################################
    wide_df = long2wide(
        run, date_col="date", cat_col="category", count_col="count", how="sum"
    )
    wide_df["date"] = pd.to_datetime(wide_df["date"])
    wide_df["ai_social"] = wide_df["ai_social"].astype(int)
    wide_df["plain"] = wide_df["plain"].astype(int)
    wide_df["social_prop"] = wide_df["ai_social"] / wide_df["plain"]

    wide_df.to_csv(
        f"../../data/clean/wide_{START_DATE}_{END_DATE}_{COLLECTION}_ai_social.csv",
        index=False,
    )
