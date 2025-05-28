"""
Date: 2025-05-26 12:06:46

Description: Scrapes ArXiv papers based on specified date range and categories.

Input files:
- DLs this: arxiv-metadata-oai-snapshot.json (from Kaggle dataset Cornell-University/arxiv/versions/234)

Output files:
- ../data/clean/arxiv_<start_date>_<end_date>_<allowed_categories_str>.jsonl

- ../data/raw/arxiv_<start_date>_<end_date>_<allowed_categories_str>_pre_transform.jsonl

Both are jsonl files after we apply the filtering criteria. But heres the difference:
- The cleaned one contains only the fields we want, and has a v1_date field.
- The pre-transform one contains ALL the original fields, but is filtered by date and categories.

"""
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.json_utils import JsonUtils
from src.helpers import path2correct_loc
from datetime import datetime
from functools import partial
import kagglehub

from pathlib import Path
import logging
import os
logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

start_date = '2018-01-01'
end_date = '2025-05-20'
allowed_categories = ["cs"]
allowed_categories_str = "-".join(f"{cat}" for cat in allowed_categories).strip()



logging.info(f"Filtering ArXiv entries from {start_date} to {end_date} for categories: {allowed_categories_str}")


def is_valid(entry, allowed_categories=None, allowed_major_categories=None,
             allowed_minor_categories=None, start_date=None, end_date=None):
    """
    Check if an ArXiv entry is valid based on categories and date range.
    """
    # Get v1_date first since we need it for date filtering
    versions = entry.get('versions', [])
    v1_entry = next((v for v in versions if v['version'] == 'v1'), None)
    if v1_entry:
        try:
            date_obj = datetime.strptime(v1_entry['created'], '%a, %d %b %Y %H:%M:%S %Z')
            v1_date = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            v1_date = None
    else:
        v1_date = None

    # Category check
    if any(x is not None for x in [allowed_categories, allowed_major_categories, allowed_minor_categories]):
        entry_categories = entry.get('categories', '')
        if not entry_categories:
            return False

        entry_cats = [cat.strip() for cat in entry_categories.split()]

        if allowed_categories is not None:
            if not any(cat in allowed_categories for cat in entry_cats):
                return False

        if allowed_major_categories is not None:
            entry_majors = [cat.split('.')[0] if '.' in cat else cat for cat in entry_cats]
            if not any(major in allowed_major_categories for major in entry_majors):
                return False

        if allowed_minor_categories is not None:
            entry_minors = [cat.split('.')[1] if '.' in cat and len(cat.split('.')) > 1 else ''
                            for cat in entry_cats]
            entry_minors = [minor for minor in entry_minors if minor]
            if not any(minor in allowed_minor_categories for minor in entry_minors):
                return False

    # Date check using v1_date
    if start_date is not None or end_date is not None:
        if not v1_date:
            return False

        try:
            entry_date = datetime.strptime(v1_date, '%Y-%m-%d')

            if start_date is not None:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                if entry_date < start_dt:
                    return False

            if end_date is not None:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                if entry_date > end_dt:
                    return False
        except ValueError:
            # Invalid date format
            return False

    return True


def transform(x):
    """Transform entry by adding v1_date and removing unwanted fields."""
    # Get the v1_date
    versions = x.get('versions', [])
    v1_entry = next((v for v in versions if v['version'] == 'v1'), None)
    if v1_entry:
        date_obj = datetime.strptime(v1_entry['created'], '%a, %d %b %Y %H:%M:%S %Z')
        v1_date = date_obj.strftime('%Y-%m-%d')
    else:
        v1_date = None

    # Add v1_date to the entry
    x['analysis_date'] = v1_date
    x['abstract'] = x['abstract'].replace('\n', ' ').replace('\r', ' ').strip() if x.get('abstract') else ''
    x['title'] = x.get('title', '').replace('\n', ' ').replace('\r', ' ').strip()

    # Remove unwanted fields
    fields_to_remove = {'license', 'authors', 'submitter', 'doi', 'report-no', 'journal-ref', 'versions', 'authors_parsed', 'comments'}
    return {k: v for k, v in x.items() if k not in fields_to_remove}


def main():

    output_fn = f"../data/clean/arxiv_{start_date}_{end_date}_{allowed_categories_str}.jsonl"

    pre_transform_output_fn = f"{output_fn}".replace(".jsonl", "_pre_transform.jsonl")
    pre_transform_output_fn = pre_transform_output_fn.replace("data/clean", "data/raw")



    if os.path.exists(output_fn):
        print(f"Output file {output_fn} already exists. Exiting to avoid overwriting.")
        logging.info(f"Output file {output_fn} already exists. Exiting to avoid overwriting.")
        return None



    validation_criteria = partial(is_valid,
                                  allowed_major_categories=allowed_categories,
                                  start_date=start_date,
                                  end_date=end_date)

    # dls to cache
    path = kagglehub.dataset_download("Cornell-University/arxiv/versions/234", force_download=True)
    new_location = path2correct_loc(path, "")
    print(f"New location: {new_location}")

    fn = "arxiv-metadata-oai-snapshot.json"

    jutils = JsonUtils()

    jsons = jutils.read_and_filter_json_file(filename=fn,
                                             validator_function=validation_criteria)

    counter = 0
    with open(output_fn, 'w') as f, open(pre_transform_output_fn, 'w') as f_pre:
        for entry in jsons:
            # Write original entry to pre-transform file
            entry.update({"unique_idx": counter})  # Add unique index
            json.dump(entry, f_pre)
            f_pre.write('\n')

            # Transform and write to main output file
            transformed_entry = transform(entry)
            transformed_entry.update({"unique_idx": counter})  # Add unique index
            json.dump(transformed_entry, f)
            f.write('\n')
            counter +=1

    print(f"Filtered entries saved to {output_fn}")
    logging.info(f"Filtered entries saved to {output_fn}")

    print(f"Pre-transform filtered entries saved to {pre_transform_output_fn}")
    logging.info(f"Pre-transform filtered entries saved to {pre_transform_output_fn}")

    # delete that big arxiv file since it contains way more than what we want
    to_delete = Path("arxiv-metadata-oai-snapshot.json")
    if to_delete.exists():
        os.remove(to_delete)
        print(f"Deleted raw file: {to_delete}")
        logging.info(f"Deleted raw file: {to_delete}")


if __name__ == "__main__":
    main()
