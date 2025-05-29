#!/bin/bash

set -e  # Exit on any error

if [[ -f "../src/.env" ]]; then
    source "../src/.env"
    echo "Loaded environment variables from ../src/.env"
else
    echo "ERROR: ../src/.env file not found"
    exit 1
fi

# CHECK ALL ENV VARS EXIST
echo "checking env variables..."

if [[ -z "${PRODUCT_HUNT_API_KEY:-}" ]]; then
    echo "ERROR: PRODUCT_HUNT_API_KEY not set in .env"
    exit 1
fi

if [[ -z "${KAGGLE_KEY:-}" ]]; then
    echo "ERROR: KAGGLE_KEY not set in .env"
    exit 1
fi

if [[ -z "${KAGGLE_USERNAME:-}" ]]; then
    echo "ERROR: KAGGLE_USERNAME not set in .env"
    exit 1
fi

if [[ -z "${NYT_API_KEY:-}" ]]; then
    echo "ERROR: NYT_API_KEY not set in .env"
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set in .env"
    exit 1
fi

echo "All environment variables OK"

# Create data directories if dont exist
mkdir -p ../data/clean
mkdir -p ../data/raw



echo "Running Study 1..."

echo "Scraping ProductHunt..."
python3 1_scrape_product_hunt.py

echo "Scraping ArXiv..."
python3 2_scrape_arxiv_papers.py

echo "Scraping NYT..."
python3 3_scrape_nyt_archives.py

echo "Getting ATUS roles..."
python3 4_get_atus_roles.py

echo "Getting ONET roles..."
python3 5_get_onet_roles.py

echo "Getting SSL actions..."
python3 6_get_ssl_actions.py

echo "Getting AI terms..."
python3 7_get_more_ai_terms.py

echo "Getting word counts"
python3 8_get_word_counts.py

echo "S1 finished"