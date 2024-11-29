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

logging.basicConfig(
    filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    filemode="w",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def list2text_file(lst, filename):
    with open(f"{filename}.txt", "w") as f:
        for item in lst:
            f.write("%s\n" % item)


if __name__ == "__main__":
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

    def topn_by_occ(df, skill, topn):
        tdf = df[df["element_name"] == skill]
        occs = (
            tdf.sort_values(by=["data_value"], ascending=False)
            .head(topn)["title"]
            .to_list()
        )
        return occs

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
