#!/usr/bin/env python3
"""
Simplified Guardian API Content Scraper with JSONL output and rate limiting
"""

import requests
import json
import time
from datetime import datetime, timedelta
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)

from dotenv import load_dotenv

load_dotenv("../src/.env")

# ============== CONFIGURATION ==============
API_KEYS = os.getenv('GUARDIAN_API_KEYS', '').split(',')
START_DATE = "2024-01-01"  # YYYY-MM-DD
END_DATE = "2024-02-01"  # YYYY-MM-DD
QUERY = ""

# ===========================================

class RateLimitError(Exception):
    """Custom exception for rate limiting."""
    pass

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: print(f"Rate limited! Retrying in {retry_state.next_action.sleep} seconds...")
)
def make_api_request(url, params):
    """Make API request with retry logic for rate limits."""
    response = requests.get(url, params=params, timeout=30)

    if response.status_code == 429:
        print(f"Hit 429 rate limit at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        raise RateLimitError("Rate limit exceeded")

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        response.raise_for_status()

    return response

def get_stories_for_date(date, query=""):
    """Get all stories for a specific date."""
    stories = []
    page = 1
    request_count = 0

    while True:
        params = {
            'api-key': API_KEYS[request_count % len(API_KEYS)],
            'from-date': date,
            'to-date': date,
            'page': page,
            'page-size': 50,
            'order-by': 'newest',
            'show-fields': 'trailText,headline,body,lastModified,standfirst,shortUrl,thumbnail,wordcount,byline,starRating',
            'show-tags': 'contributor,keyword,tone,type,series',
            'show-elements': 'image,video,audio',
            'show-section': 'true',
            'format': 'json'
        }

        if query:
            params['q'] = query

        try:
            time.sleep(0.2)  # Slightly increased rate limiting
            response = make_api_request("https://content.guardianapis.com/search", params)

            data = response.json()
            results = data['response']['results']

            if not results:
                break

            for story in results:
                story['scrape_date'] = date

            stories.extend(results)

            if len(results) < 50:  # Last page
                break

            page += 1
            request_count += 1

        except RateLimitError:
            print(f"Failed to get data for {date}, page {page} after retries")
            break
        except Exception as e:
            print(f"Request failed for {date}, page {page}: {e}")
            break

    return stories

def generate_dates(start_date, end_date):
    """Generate all dates between start and end."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    current = start
    while current <= end:
        yield current.strftime('%Y-%m-%d')
        current += timedelta(days=1)

def save_to_jsonl(stories, filename):
    """Save stories to JSONL format."""
    with open(filename, 'w', encoding='utf-8') as f:
        for story in stories:
            json.dump(story, f, ensure_ascii=False)
            f.write('\n')

def main():
    """Main scraping function."""
    all_stories = []

    print(f"Scraping Guardian articles from {START_DATE} to {END_DATE}")
    print(f"Using {len(API_KEYS)} API key(s) in rotation")
    if QUERY:
        print(f"Query: {QUERY}")

    for date in generate_dates(START_DATE, END_DATE):
        print(f"Fetching stories for {date}...")

        stories = get_stories_for_date(date, QUERY)
        all_stories.extend(stories)

        print(f"Found {len(stories)} stories for {date}. Total: {len(all_stories)}")

    output_file = f"../data/raw/guardian_raw_stories_{START_DATE}_to_{END_DATE}.jsonl"
    save_to_jsonl(all_stories, output_file)

    print(f"\nScraping complete!")
    print(f"Total stories: {len(all_stories)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()