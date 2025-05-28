import json
import time
import os
import logging
from datetime import datetime
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("../src/.env")

# Configuration
start_date = "2018-01-01"
end_date = "2025-06-01"
debug = False
api_key = os.getenv("NYT_API_KEY")

BASE_URL = "https://api.nytimes.com/svc/archive/v1"

# Setup logging
logging.basicConfig(
    filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log",
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    filemode='w',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)


def parse_date_range(start_date, end_date):
    """Parse date strings and return list of (year, month) tuples"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    months = []
    current = start.replace(day=1)

    while current <= end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return months


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
def fetch_archive_data(year, month, api_key):
    """Fetch archive data for a specific year/month"""
    url = f"{BASE_URL}/{year}/{month}.json"
    params = {"api-key": api_key}

    response = requests.get(url, params=params, timeout=60)

    if response.status_code == 429:
        # minor todo: handle rate limiting
        # It does not give an actual retry-after header, so we have to handle it manually
        # "fault":{"faultstring":"Rate limit quota violation. Quota limit  exceeded. Identifier : b0c90cbd-06bb-489e-8a4e-a8fa2c087196","detail":{"errorcode":"policies.ratelimit.QuotaViolation"
        retry_after = response.headers.get('Retry-After', 60) # can't find the actual retry-after so i am just doing 60s for now
        print("headers", response.headers)
        logging.warning(f"Rate limited. Waiting {retry_after} seconds...")
        time.sleep(int(retry_after))
        response = requests.get(url, params=params, timeout=60)

    response.raise_for_status()
    return response.json()


def transform_article(article):
    """Transform article by extracting analysis_date and removing fields"""
    transformed = article.copy()

    # Extract analysis_date from pub_date
    if 'pub_date' in article:
        try:
            dt = datetime.fromisoformat(article['pub_date'].replace('Z', '+00:00'))
            transformed['analysis_date'] = dt.strftime('%Y-%m-%d')
        except:
            transformed['analysis_date'] = article['pub_date'][:10]

    transformed['main_headline'] = article.get('headline', {}).get('main', '')

    # Remove specified fields
    for field in ['multimedia', 'document_type', 'byline', 'keywords']:
        transformed.pop(field, None)

    return transformed


def main():
    logging.info(f"Starting NYT archive pull: {start_date} to {end_date}")

    months = parse_date_range(start_date, end_date)

    if debug:
        months = months[:1]
        logging.info(f"Debug mode: Processing only {months[0]}")

    # Create output files
    output_file = f"../data/clean/nyt_pull_{start_date}_{end_date}.jsonl"
    pre_transform_file = f"../data/raw/nyt_pull_{start_date}_{end_date}_pre_transform.jsonl"

    # Clear existing files
    Path(output_file).unlink(missing_ok=True)
    Path(pre_transform_file).unlink(missing_ok=True)

    total_articles = 0
    counter = 0

    with tqdm(months, desc="Fetching months") as pbar:
        for year, month in pbar:
            pbar.set_description(f"Fetching {year}-{month:02d}")

            try:
                data = fetch_archive_data(year, month, api_key)
                articles = data.get('response', {}).get('docs', [])

                # Write articles with unique indices
                with open(output_file, 'a') as f, open(pre_transform_file, 'a') as f_pre:
                    for article in articles:
                        # Original with index
                        article['unique_idx'] = counter
                        f_pre.write(json.dumps(article) + '\n')

                        # Transformed with index
                        transformed = transform_article(article)
                        transformed['unique_idx'] = counter
                        f.write(json.dumps(transformed) + '\n')

                        counter += 1

                total_articles += len(articles)
                pbar.set_postfix(articles=len(articles), total=total_articles)

                if not debug:
                    time.sleep(1)

            except Exception as e:
                logging.error(f"Error processing {year}-{month}: {e}")
                continue

    logging.info(f"Complete. Total articles: {total_articles}")
    print(f"Complete! {total_articles} articles written to {output_file}")


if __name__ == "__main__":
    main()