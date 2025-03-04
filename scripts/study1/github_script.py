import requests
import csv
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# source venv/bin/activate to activate virtual environment
# GitHub personal access token for authentication
# Load environment variables from .env file
load_dotenv()

# Access the GitHub token from environment variables
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Use GITHUB_TOKEN in your script as before
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# Function to query GitHub API
def query_github_api(query, created_date):
    base_url = "https://api.github.com/search/repositories"
    params = {
        "q": f"{query} created:{created_date}",
        "per_page": 1  # We only need the count of results
    }
    response = requests.get(base_url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get("total_count", 0)
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return 0

# Function to generate daily counts for each keyword
def fetch_daily_counts(keywords, start_date, end_date):
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=1)

    results = []

    while start_date_obj <= end_date_obj:
        current_date = start_date_obj.strftime("%Y-%m-%d")
        for query, terms in keywords.items():
            core_term = terms.get("core_term")
            social_term = terms.get("social_term")
            is_social = 1 if social_term else 0
            
            # Construct search query
            search_query = f"{core_term} {social_term}" if social_term else core_term
            
            # Query GitHub API
            count = query_github_api(search_query.strip(), current_date)
            
            # Append result
            results.append({
                "date": current_date,
                "value": count,
                "query": search_query.strip(),
                "core_term": core_term,
                "social_term": social_term,
                "is_social": is_social
            })
        
        start_date_obj += delta
    
    return results

# Write results to CSV
def write_to_csv(results, output_file="output.csv"):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["date", "value", "query", "core_term", "social_term", "is_social"])
        writer.writeheader()
        writer.writerows(results)

# Example usage
if __name__ == "__main__":
    keywords_dict = {
        "ai": {"core_term": "ai", "social_term": None},
        "social_terms": {"core_term": "chatbot", "social_term": "therapist"},
        "ai_social": {"core_term": "ai", "social_term": "collaboration"}
    }
    
    start_date = "2024-01-01"
    end_date = "2024-01-03"  # Example small range
    
    # Fetch daily counts
    data = fetch_daily_counts(keywords_dict, start_date, end_date)
    
    # Write to CSV
    write_to_csv(data)
    
    print("CSV file generated successfully!")
