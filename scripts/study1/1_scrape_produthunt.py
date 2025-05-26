import argparse
import logging
import requests
import datetime
import time
import json
import os
import logging
import csv
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv("../../src/.env")  # Load environment variables from .env file

# Set up logging
logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)


start_date_str = "2025-05-01"
end_date_str = "2025-05-01"

logging.info(f"Start date: {start_date_str}, End date: {end_date_str}")

"""
Script to fetch ProductHunt posts within a given timeframe including description and text content.
"""

# Define a custom exception for rate limiting
class RateLimitException(Exception):
    """Exception raised when API rate limits are hit"""
    pass

# Define a custom exception for generic API errors
class APIError(Exception):
    """Generic exception for API errors not related to rate limiting"""
    pass

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitException, requests.exceptions.RequestException, APIError)),
    reraise=True
)
def fetch_posts_in_timeframe(start_date_str, end_date_str, token, first=20, after=None):
    """
    Fetch posts within a timeframe using ProductHunt API with description field

    Args:
        start_date_str: Start date in<\ctrl3348>-MM-DD format
        end_date_str: End date in<\ctrl3348>-MM-DD format
        token: Developer token for API access
        first: Number of results to return per page (default: 20)
        after: Cursor for pagination

    Returns:
        Dictionary with posts, total count, and pagination info
    """
    # Parse dates and create timestamp range
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

    start_timestamp = start_date.replace(hour=0, minute=0, second=0).isoformat()
    end_timestamp = end_date.replace(hour=23, minute=59, second=59).isoformat()

    # Build GraphQL query filters
    filters = [
        f'postedAfter: "{start_timestamp}"',
        f'postedBefore: "{end_timestamp}"',
        f'first: {first}'
    ]

    if after:
        filters.append(f'after: "{after}"')

    # GraphQL query with description field included
    query = f"""{{
      posts({", ".join(filters)}) {{
        totalCount
        edges {{
          node {{
            id
            name
            slug
            tagline
            description
            url
            website
            votesCount
            commentsCount
            createdAt
            featuredAt
          }}
        }}
        pageInfo {{
          hasNextPage
          endCursor
        }}
      }}
    }}"""

    # Set up headers with developer token and User-Agent
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "curl/7.64.1"
    }

    # Make API request
    response = requests.post(
        "https://api.producthunt.com/v2/api/graphql",
        json={"query": query},
        headers=headers
    )

    # Print status for debugging
    logging.info(f"Status Code: {response.status_code}")

    # Check for rate limiting
    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After', '60')
        logging.info(f"Rate limited. Suggested retry after: {retry_after} seconds")
        raise RateLimitException(f"Rate limit exceeded. Retry after {retry_after} seconds")
    # Process response if successful
    if response.status_code == 200:
        data = response.json()

        if "errors" in data:
            error_message = data["errors"][0]["message"] if data["errors"] else "Unknown error"
            logging.info(f"Full error: {data['errors']}")

            # Check for rate limit type error in GraphQL errors
            if "rate limit" in error_message.lower():
                raise RateLimitException(f"GraphQL Rate Limit Error: {error_message}")
            else:
                raise APIError(f"GraphQL Error: {error_message}")

        if "data" in data and "posts" in data["data"]:
            posts_data = data["data"]["posts"]
            total_count = posts_data["totalCount"]

            # Extract posts from edges
            posts = []
            if "edges" in posts_data and posts_data["edges"]:
                for edge in posts_data["edges"]:
                    if "node" in edge:
                        posts.append(edge["node"])

            # Extract pagination info
            page_info = posts_data["pageInfo"]

            return {
                "posts": posts,
                "count": len(posts),
                "total_count": total_count,
                "has_next_page": page_info["hasNextPage"],
                "end_cursor": page_info["endCursor"]
            }

        raise APIError("Error: Unexpected response structure")


    # Handle other HTTP errors
    response.raise_for_status()

    # Return error information (this might not be the best for retry logic, but kept for now)
    return f"Error: {response.status_code} - {response.text}"

def fetch_all_posts_in_timeframe(start_date_str, end_date_str, token, max_pages=20):
    """
    Fetch all posts within a timeframe with pagination

    Args:
        start_date_str: Start date in<\ctrl3348>-MM-DD format
        end_date_str: End date in<\ctrl3348>-MM-DD format
        token: Developer token for API access
        max_pages: Maximum number of pages to fetch (default: 20)

    Returns:
        List of all posts and total count
    """
    all_posts = []
    page = 1
    has_next_page = True
    cursor = None
    total_count = 0

    while has_next_page and page <= max_pages:
        logging.info(f"Fetching page {page} for timeframe {start_date_str} to {end_date_str}...")

        try:
            result = fetch_posts_in_timeframe(start_date_str, end_date_str, token, after=cursor)

            if isinstance(result, str):  # Error occurred
                logging.info(result)
                break

            if "posts" in result:
                all_posts.extend(result["posts"])
                has_next_page = result["has_next_page"]
                cursor = result["end_cursor"]
                total_count = result["total_count"]

                logging.info(f"Retrieved {len(result['posts'])} posts (total: {len(all_posts)} of {total_count})")

                if not has_next_page:
                    logging.info("No more pages available.")
                    break

            page += 1

        except RateLimitException as e:
            retry_after = str(e).split("Retry after ")[-1].split(" ")[0]
            wait_seconds = int(retry_after)
            logging.info(f"Rate limit encountered. Waiting for {wait_seconds} seconds before retrying page {page}...")
            time.sleep(wait_seconds)
            logging.info(f"Resuming fetch for page {page}...")
            continue  # Continue to the next iteration of the loop (same page)

        except Exception as e:
            logging.info(f"Error fetching page {page}: {str(e)}")
            break

    return {
        "posts": all_posts,
        "count": len(all_posts),
        "total_count": total_count,
        "timeframe": {
            "start_date": start_date_str,
            "end_date": end_date_str
        }
    }

def fetch_posts_by_single_day(start_date_str, end_date_str, token):
    """
    Fetch posts by processing one day at a time to reduce complexity

    Args:
        start_date_str: Start date in<\ctrl3348>-MM-DD format
        end_date_str: End date in<\ctrl3348>-MM-DD format
        token: Developer token for API access

    Returns:
        Combined list of all posts
    """
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

    all_posts = []
    total_count = 0
    current_date = start_date

    while current_date <= end_date:
        current_date_str = current_date.strftime("%Y-%m-%d")

        logging.info(f"\n=== Processing date: {current_date_str} ===")

        try:
            # Use the same date for both start and end to get just one day
            result = fetch_all_posts_in_timeframe(current_date_str, current_date_str, token)

            if isinstance(result, str):  # Error occurred
                logging.info(result)
            else:
                all_posts.extend(result["posts"])
                total_count += result["total_count"]
                logging.info(f"Added {result['count']} posts from {current_date_str}")

        except RateLimitException as e:
            retry_after = str(e).split("Retry after ")[-1].split(" ")[0]
            wait_seconds = int(retry_after)
            logging.info(f"Rate limit encountered while processing {current_date_str}. Waiting for {wait_seconds} seconds...")
            time.sleep(wait_seconds)
            logging.info(f"Resuming fetch for {current_date_str}...")
            continue  # Continue to the next day

        except Exception as e:
            logging.info(f"Error processing date {current_date_str}: {str(e)}")

        # Move to next day
        current_date += datetime.timedelta(days=1)

    return {
        "posts": all_posts,
        "count": len(all_posts),
        "total_count": total_count,
        "timeframe": {
            "start_date": start_date_str,
            "end_date": end_date_str
        }
    }

def save_posts_to_json(posts_data, output_file):
    """
    Save posts data to a JSON file

    Args:
        posts_data: Dictionary with posts data
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(posts_data, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved {posts_data['count']} posts to {output_file}")

def save_posts_to_csv(posts_data, output_file):
    """
    Save posts data to a CSV file

    Args:
        posts_data: Dictionary with posts data
        output_file: Output file path
    """
    if not posts_data.get("posts"):
        logging.info("No posts to save")
        return

    posts = posts_data["posts"]

    # Define CSV fields (including description)
    fieldnames = [
        "id", "name", "slug", "tagline", "description", "url", "website",
        "votesCount", "commentsCount", "createdAt", "featuredAt"
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for post in posts:
            # Create row with available fields
            row = {field: post.get(field, "") for field in fieldnames}
            writer.writerow(row)

    logging.info(f"Saved {posts_data['count']} posts to {output_file}")

# Example usage
if __name__ == "__main__":
    developer_token = os.environ["PRODUCT_HUNT_API_KEY"]

    result = fetch_posts_by_single_day(start_date_str, end_date_str, developer_token)

    if isinstance(result, str):  # Error occurred
        logging.info(f"Error: {result}")
    else:
        # Print summary
        logging.info(f"\nSummary for {start_date_str} to {end_date_str}:")
        logging.info(f"Retrieved {result['count']} posts of {result['total_count']} total")


        # Save results to files
        save_posts_to_json(result, f"../../data/clean/producthunt_posts_{start_date_str}_to_{end_date_str}.json")

        # Print the first 5 posts as a sample
        logging.info("\nSample posts (with descriptions):")
        for i, post in enumerate(result["posts"][:5]):
            logging.info(f"{i+1}. {post['name']} (slug: {post['slug']})")
            logging.info(f"   URL: {post['url']}")
            logging.info(f"   Tagline: {post.get('tagline', 'N/A')}")

            # Print partial description if available
            description = post.get('description', 'No description available')
            if description:
                # Truncate long descriptions for display
                preview = description[:150] + "..." if len(description) > 150 else description
                logging.info(f"   Description preview: {preview}")

