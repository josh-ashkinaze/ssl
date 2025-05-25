import logging
import os
import time
import xmltodict
import pandas as pd
from datetime import datetime
from sickle import Sickle
from urllib.error import HTTPError
from requests.exceptions import RequestException
from tqdm import tqdm
from collections import Counter

# Set up logging with the script filename
script_name = os.path.splitext(os.path.basename(__file__))[0]
logging.basicConfig(
    filename=f"{script_name}.log",
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    filemode='w',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

# Configuration
CONFIG = {
    "RAW_FILE": "../../../data/raw/arxiv_metadata_raw.xml",
    "CSV_FILE": "../../../data/raw/arxiv_papers.csv",
    "SET": "cs.AI",  # Changed from "cs" to "cs.AI"
    "FROM_DATE": "2018-01-01",  # Fixed start date
    "UNTIL_DATE": "2025-05-01"  # Fixed end date
}

# Create parent directories
for file_path in [CONFIG["RAW_FILE"], CONFIG["CSV_FILE"]]:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

logging.info("Using this config:")
for key, value in CONFIG.items():
    logging.info(f"{key}: {value}")


def get_next_record_with_retry(data_iterator, max_attempts=10):
    attempts = 0
    while attempts < max_attempts:
        try:
            return next(data_iterator)
        except StopIteration:
            raise
        except (HTTPError, RequestException) as e:
            attempts += 1
            retry_after = getattr(e.response, 'headers', {}).get('Retry-After', None)
            wait_time = int(retry_after) if retry_after else 30
            logging.info(f"Retrying in {wait_time} seconds (attempt {attempts}/{max_attempts})...")
            time.sleep(wait_time)
            if attempts >= max_attempts:
                logging.error("Maximum retries reached.")
                raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise


def download_metadata(config):
    connection = Sickle('http://export.arxiv.org/oai2')
    logging.info(f'Downloading {config["SET"]} papers from {config["FROM_DATE"]} to {config["UNTIL_DATE"]}')
    params = {
        'metadataPrefix': 'arXiv',
        'from': config["FROM_DATE"],
        'until': config["UNTIL_DATE"],
        'ignore_deleted': True,
        'set': config["SET"]
    }

    try:
        data = connection.ListRecords(**params)
    except Exception as e:
        logging.error(f'Connection failed: {e}')
        raise

    iters = 0
    errors = 0
    pbar = tqdm(desc="Downloading records", unit="records")

    with open(config["RAW_FILE"], 'w', encoding="utf-8") as f:
        while True:
            try:
                record = get_next_record_with_retry(data)
                f.write(record.raw + '\n')
                iters += 1
                pbar.update(1)
                if iters % 1000 == 0:
                    logging.info(f"{iters} records downloaded.")
            except StopIteration:
                break
            except Exception as e:
                errors += 1
                logging.error(f'Error: {e}')
                if errors > 5:
                    break

    pbar.close()
    logging.info(f"Downloaded {iters} records.")
    return iters


def convert_dict(record_xml):
    try:
        record = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
        record['id'] = str(record['id'])
        authors = record.get('authors', {}).get('author', [])
        if isinstance(authors, dict):
            authors = [authors]
        record['authors'] = [
            (a.get('forenames', '') + ' ' + a['keyname']).strip() for a in authors
        ] if authors else []
        record['url'] = f"https://arxiv.org/abs/{record['id']}"
        return record
    except Exception as e:
        logging.error(f"Error parsing record: {e}")
        return None


def parse_raw_data(config):
    with open(config["RAW_FILE"], 'r', encoding="utf-8") as f:
        raw_data = f.read()
    records = raw_data.split('</record>')
    records = [r + '</record>' for r in records if r.strip()]
    parsed = [convert_dict(r) for r in tqdm(records, desc="Parsing records") if r.strip()]
    df = pd.DataFrame([p for p in parsed if p])
    df['authors_str'] = df['authors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    df['keywords'] = df['categories'].apply(lambda x: x if isinstance(x, str) else ', '.join(x) if isinstance(x, list) else '')
    df.to_csv(config["CSV_FILE"], index=False)
    logging.info(f"Saved {len(df)} records to {config['CSV_FILE']}")
    return df


def analyze_papers(df, config):
    logging.info(f"Total papers: {len(df)}")
    if 'created' in df.columns:
        logging.info(f"Date range: {df['created'].min()} to {df['created'].max()}")
    if 'categories' in df.columns:
        all_cats = []
        for cat in df['categories']:
            if isinstance(cat, str):
                all_cats.extend(cat.split())
        cat_counts = Counter(all_cats)
        for cat, count in cat_counts.most_common(10):
            if cat.startswith(config['SET']):
                logging.info(f"{cat}: {count}")


def main(config=None):
    if config is None:
        config = CONFIG
    logging.info("Starting arXiv scrape...")
    num_records = download_metadata(config)
    if num_records > 0:
        df = parse_raw_data(config)
        analyze_papers(df, config)
    else:
        logging.warning("No records retrieved.")


if __name__ == "__main__":
    main()