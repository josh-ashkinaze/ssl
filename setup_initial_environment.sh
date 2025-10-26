#!/bin/bash

# Exit on any error
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required packages..."
pip install litellm requests beautifulsoup4 seaborn matplotlib scikit-learn pandas openai sentence-transformers scipy statsmodels chardet ftfy python-dotenv

echo "Generating requirements.txt..."
pip freeze > requirements.txt

echo "Setup complete! Requirements saved to requirements.txt"
echo ""
echo "To activate the virtual environment in future sessions, run:"
echo "source venv/bin/activate"
echo ""
echo "To deactivate when done, run:"
echo "deactivate"