import os

import mediacloud.api
from datetime import datetime, timedelta
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging

from dotenv import load_dotenv

load_dotenv("../src/.env")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
YOUR_MC_API_KEY = os.getenv("MEDIACLOUD_API_KEY")
US_NATIONAL_COLLECTION = 34412234

# Query for random stories (common words with OR operators)
RANDOM_QUERY = "the OR be OR to OR of OR and OR a OR in OR that OR have OR I"

# Daily limit
DAILY_LIMIT = 5000


def fetch_stories_by_date_range(start_date, end_date, stories_per_day=DAILY_LIMIT):
    """
    Fetch stories from MediaCloud for a given date range, fetching up to stories_per_day for each day.

    Args:
        start_date (datetime.date or str): Start date (YYYY-MM-DD format if string)
        end_date (datetime.date or str): End date (YYYY-MM-DD format if string)
        stories_per_day (int): Maximum number of stories to fetch per day

    Returns:
        list[dict]: List of story dictionaries
    """
    # Convert string dates to datetime.date objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Initialize the MediaCloud API client
    mc_search = mediacloud.api.SearchApi(YOUR_MC_API_KEY)

    all_stories = []
    current_date = start_date

    print(f"Fetching stories from {start_date} to {end_date}")
    print(f"Target: {stories_per_day} stories per day")
    print("-" * 50)

    while current_date <= end_date:
        print(f"📅 Processing date: {current_date}")
        day_start_time = time.time()

        day_stories = fetch_stories_for_single_day(
            mc_search,
            current_date,
            stories_per_day
        )

        day_duration = time.time() - day_start_time
        all_stories.extend(day_stories)
        print(
            f"✅ Date: {current_date} - Fetched: {len(day_stories)} stories in {day_duration:.1f}s (Total: {len(all_stories)})")

        current_date += timedelta(days=1)

        # Add delay between days to be respectful to the API
        if current_date <= end_date:  # Don't wait after the last day
            print(f"⏳ Waiting 1.0s before next day...")
            time.sleep(1.0)

    print(f"\nCompleted! Total stories fetched: {len(all_stories)}")
    return all_stories


def log_retry_attempt(retry_state):
    """Custom function to log retry attempts with timing info"""
    exception = retry_state.outcome.exception()
    attempt_number = retry_state.attempt_number
    if attempt_number > 1:
        wait_time = retry_state.next_sleep
        print(f"🔄 Retry attempt #{attempt_number} - Exception: {exception}")
        print(f"⏰ Waiting {wait_time:.1f} seconds before next attempt...")
        logger.warning(f"Retry attempt {attempt_number} after {exception}. Waiting {wait_time:.1f}s")


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=log_retry_attempt,
    reraise=True
)
def fetch_page_with_retry(mc_search, query, collection_ids, start_date, end_date, pagination_token=None):
    """
    Fetch a single page of results with retry logic for 403 errors
    """
    return mc_search.story_list(
        query,
        collection_ids=collection_ids,
        start_date=start_date,
        end_date=end_date,
        pagination_token=pagination_token
    )


def fetch_stories_for_single_day(mc_search, date, max_stories):
    """
    Fetch stories for a single day

    Args:
        mc_search: MediaCloud search API client
        date (datetime.date): The date to fetch stories for
        max_stories (int): Maximum number of stories to fetch

    Returns:
        list[dict]: List of story dictionaries for that day
    """
    stories = []
    pagination_token = None
    more_stories = True

    while more_stories and len(stories) < max_stories:
        try:
            request_start_time = time.time()

            # Fetch a page of results with retry logic
            page, pagination_token = fetch_page_with_retry(
                mc_search,
                RANDOM_QUERY,
                [US_NATIONAL_COLLECTION],
                date,
                date,
                pagination_token
            )

            request_duration = time.time() - request_start_time

            # Add stories to our collection (don't exceed max_stories)
            stories_to_add = page[:max_stories - len(stories)]
            stories.extend(stories_to_add)

            print(f"  📖 Fetched page: {len(page)} stories in {request_duration:.1f}s (Day total: {len(stories)})")

            # Check if we should continue
            more_stories = pagination_token is not None and len(stories) < max_stories

            # Add a small delay between successful requests
            if more_stories:
                print(f"  ⏳ Waiting 0.5s before next page...")
                time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error fetching stories for {date} after all retries: {e}")
            logger.error(f"Final error for {date}: {e}")
            break

    return stories


def fetch_stories_bulk(start_date, end_date, max_total_stories=None):
    """
    Alternative function to fetch stories in bulk for the entire date range
    (useful if you want to limit total stories across all days rather than per day)

    Args:
        start_date (datetime.date or str): Start date
        end_date (datetime.date or str): End date
        max_total_stories (int): Maximum total stories across all days

    Returns:
        list[dict]: List of story dictionaries
    """
    # Convert string dates to datetime.date objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Initialize the MediaCloud API client
    mc_search = mediacloud.api.SearchApi(YOUR_MC_API_KEY)

    stories = []
    pagination_token = None
    more_stories = True

    print(f"Fetching stories from {start_date} to {end_date}")
    if max_total_stories:
        print(f"Maximum total stories: {max_total_stories}")
    print("-" * 50)

    bulk_start_time = time.time()

    while more_stories:
        if max_total_stories and len(stories) >= max_total_stories:
            break

        try:
            request_start_time = time.time()

            # Fetch a page of results with retry logic
            page, pagination_token = fetch_page_with_retry(
                mc_search,
                RANDOM_QUERY,
                [US_NATIONAL_COLLECTION],
                start_date,
                end_date,
                pagination_token
            )

            request_duration = time.time() - request_start_time

            # Add stories to our collection
            if max_total_stories:
                stories_to_add = page[:max_total_stories - len(stories)]
            else:
                stories_to_add = page

            stories.extend(stories_to_add)

            print(f"📖 Fetched page: {len(page)} stories in {request_duration:.1f}s (Total: {len(stories)})")

            # Check if we should continue
            more_stories = pagination_token is not None
            if max_total_stories and len(stories) >= max_total_stories:
                more_stories = False

            # Add a small delay between requests
            if more_stories:
                print(f"⏳ Waiting 0.5s before next page...")
                time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error fetching stories after all retries: {e}")
            logger.error(f"Final bulk fetch error: {e}")
            break

    bulk_duration = time.time() - bulk_start_time
    print(f"\n✅ Completed! Total stories fetched: {len(stories)} in {bulk_duration:.1f}s")
    return stories