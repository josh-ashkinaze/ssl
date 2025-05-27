"""
Date: 2025-05-26 12:06:46

Description: Processes NYT articles based on specified date range.

Input files:
- Downloads this: nyt-metadata.csv (from Kaggle dataset aryansingh0909/nyt-articles-21m-2000-present/versions/681)

Output files:
- ../data/clean/nyt_<start_date>_<end_date>.jsonl
- ../data/raw/nyt_<start_date>_<end_date>_pre_transform.jsonl

Both are jsonl files after we apply the filtering criteria. But here's the difference:
- The cleaned one contains only the fields we want: ['abstract', 'headline', 'uri', 'analysis_date', 'lede', 'section']
- The pre-transform one contains ALL the original fields, but is filtered by date.
"""

import json
import pandas as pd
import duckdb
from tqdm import tqdm
from src.json_utils import JsonUtils
from src.helpers import path2correct_loc
from datetime import datetime
from functools import partial
import kagglehub
import os
import shutil
from pathlib import Path
import logging
import ast

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log",
                   level=logging.INFO,
                   format='%(asctime)s: %(message)s',
                   filemode='w',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   force=True)

start_date = '2018-01-01'
end_date = '2025-05-20'

logging.info(f"Filtering NYT entries from {start_date} to {end_date}")

def extract_headline_main(headline_json_str):
    """
    Extract main headline from JSON string like:
    {'main': 'Playoffs or No, Dallas Provides The Motivation', 'kicker': 'PRO FOOTBALL', ...}
    """
    try:
        if pd.isna(headline_json_str) or headline_json_str == '':
            return None

        # Handle if it's already a string (not JSON)
        if isinstance(headline_json_str, str) and not headline_json_str.startswith('{'):
            return headline_json_str

        # Try to parse as JSON/dict
        headline_dict = ast.literal_eval(str(headline_json_str))
        return headline_dict.get('main', str(headline_json_str))
    except:
        # If parsing fails, return as-is
        return str(headline_json_str) if headline_json_str is not None else None

def fast_filter_csv_duckdb(filename, start_date, end_date, date_col='pub_date'):
    """
    Filters a CSV file using DuckDB for potentially faster processing.
    Returns a pandas DataFrame.
    """
    logging.info("Initializing DuckDB solution...")

    try:
        start_ts = pd.to_datetime(start_date, utc=True).to_pydatetime()
        end_ts = pd.to_datetime(end_date, utc=True).to_pydatetime()
    except Exception as e:
        logging.error(f"Error converting input dates: {e}")
        return pd.DataFrame()

    logging.info(f"Filtering for dates between {start_ts} and {end_ts}")

    # DuckDB query
    query = f"""
    SELECT *
    FROM read_csv_auto('{filename}', 
                       ignore_errors=true, 
                       all_varchar=false, 
                       dateformat = 'auto', 
                       timestampformat = 'auto')
    WHERE 
        (try_cast({date_col} AS TIMESTAMP WITH TIME ZONE) >= $start_ts AND
         try_cast({date_col} AS TIMESTAMP WITH TIME ZONE) <= $end_ts);
    """

    try:
        logging.info("Executing DuckDB query...")
        con = duckdb.connect()

        with tqdm(total=1, desc="DuckDB Processing") as pbar:
            result_relation = con.execute(query, {'start_ts': start_ts, 'end_ts': end_ts})
            filtered_df = result_relation.df()
            pbar.update(1)

        con.close()

        if not filtered_df.empty:
            logging.info(f"DuckDB found {len(filtered_df):,} rows.")
        else:
            logging.info("No data found in the specified date range by DuckDB.")
        return filtered_df

    except Exception as e:
        logging.error(f"DuckDB error occurred: {e}")
        if 'con' in locals() and con:
            try:
                con.close()
            except:
                pass
        return pd.DataFrame()

def is_valid(entry, start_date=None, end_date=None):
    """
    Check if a NYT entry is valid based on date range.
    Entry is expected to be a dict converted from DataFrame row.
    """
    # Date check using pub_date -> analysis_date
    if start_date is not None or end_date is not None:
        analysis_date = entry.get('analysis_date')
        if not analysis_date:
            return False

        try:
            entry_date = datetime.strptime(analysis_date, '%Y-%m-%d')

            if start_date is not None:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                if entry_date < start_dt:
                    return False

            if end_date is not None:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                if entry_date > end_dt:
                    return False
        except (ValueError, TypeError):
            # Invalid date format
            return False

    return True

def transform(entry):
    """Transform entry by keeping only desired fields."""
    # Extract main headline from JSON if needed
    headline = entry.get('headline', '')
    if headline:
        headline = extract_headline_main(headline)

    # Create transformed entry with only desired fields
    transformed = {
        'abstract': entry.get('abstract', ''),
        'headline': headline,
        'uri': entry.get('uri', ''),
        'analysis_date': entry.get('analysis_date', ''),
        'lede': entry.get('lead_paragraph', ''),  # assuming lead_paragraph is "lede"
        'section': entry.get('section_name', '')   # assuming section_name is "section"
    }

    return transformed

def dataframe_to_jsonl_entries(df):
    """Convert DataFrame to generator of dict entries with analysis_date added."""
    for _, row in df.iterrows():
        entry = row.to_dict()

        # Convert all pandas/numpy types to native Python types for JSON serialization
        for key, value in entry.items():
            if pd.isna(value):
                entry[key] = None
            elif hasattr(value, 'item'):  # numpy types
                entry[key] = value.item()
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                entry[key] = str(value)
            elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                entry[key] = value.item() if pd.notna(value) else None

        # Add analysis_date from pub_date
        try:
            if 'pub_date' in entry and entry['pub_date'] is not None:
                pub_date = pd.to_datetime(entry['pub_date'], utc=True)
                entry['analysis_date'] = pub_date.strftime('%Y-%m-%d')
            else:
                entry['analysis_date'] = None
        except:
            entry['analysis_date'] = None

        yield entry

if __name__ == "__main__":

    output_fn = f"../data/clean/nyt_{start_date}_{end_date}.jsonl"
    pre_transform_output_fn = f"../data/raw/nyt_{start_date}_{end_date}_pre_transform.jsonl"

    if os.path.exists(output_fn):
        print(f"Output file {output_fn} already exists. Exiting to avoid overwriting.")
        logging.info(f"Output file {output_fn} already exists. Exiting to avoid overwriting.")
        exit(0)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    os.makedirs(os.path.dirname(pre_transform_output_fn), exist_ok=True)

    validation_criteria = partial(is_valid,
                                  start_date=start_date,
                                  end_date=end_date)

    path = kagglehub.dataset_download("aryansingh0909/nyt-articles-21m-2000-present/versions/681", force_download=True)
    new_location = path2correct_loc(path, "")
    print(f"New location: {new_location}")

    fn = "nyt-metadata.csv"

    # Filter CSV using DuckDB
    logging.info("Starting CSV filtering...")
    df_filtered = fast_filter_csv_duckdb(fn, start_date, end_date)

    if df_filtered.empty:
        print("No data found in date range. Exiting.")
        logging.info("No data found in date range. Exiting.")
        exit(0)

    print(f"Found {len(df_filtered)} rows in date range")
    logging.info(f"Found {len(df_filtered)} rows in date range")


    df_filtered.to_json(
        pre_transform_output_fn,
        orient="records",
        lines=True,
        date_format="iso",
        compression=None,
    )


    clean_cols = {
        "abstract": "abstract",
        "headline": "headline",
        "lead_paragraph": "lead_paragraph",
        "snippet": "snippet",
        "uri": "uri",
        "pub_date": "pub_date",
        "analysis_date": "analysis_date",
        "section_name": "section_name",
        "news_desk": "news_desk"
    }

    clean_df = (
        df_filtered.assign(
            analysis_date=pd.to_datetime(df_filtered["pub_date"], utc=True)
            .dt.strftime("%Y-%m-%d"),
        )
        .rename(columns=clean_cols)[clean_cols.values()]  # reorder
    )

    clean_df.to_json(
        output_fn,
        orient="records",
        lines=True,
        date_format="iso",
    )


    print(f"Filtered entries saved to {output_fn}")
    logging.info(f"Filtered entries saved to {output_fn}")

    print(f"Pre-transform filtered entries saved to {pre_transform_output_fn}")
    logging.info(f"Pre-transform filtered entries saved to {pre_transform_output_fn}")

    # Delete the large CSV file since it contains way more than what we want
    to_delete = Path("nyt-metadata.csv")
    if to_delete.exists():
        os.remove(to_delete)
        print(f"Deleted raw file: {to_delete}")
        logging.info(f"Deleted raw file: {to_delete}")