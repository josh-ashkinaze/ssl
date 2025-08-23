"""
Author: Joshua Ashkinaze

Description: Downloads the normbank dataset

Input:
- None

Output:
- downloads normbank to data/raw/normbank.csv. Also adds a column called `idx` for indexing.


Date: 2025-08-20 12:59:45
"""
import pandas as pd


df = pd.read_csv("hf://datasets/SALT-NLP/NormBank/NormBank.csv")
df['idx'] = range(len(df))

df.to_csv("../data/raw/normbank.csv", index=False)
