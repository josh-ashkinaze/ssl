from datetime import datetime
from typing import List

import mediacloud.api
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def query_count_over_time(
        query, collection, start_date, end_date, api_key
) -> pd.DataFrame:
    """
    Get story counts from MediaCloud collection based on query and date range.

    Args:
        query (str): Search query string (use double quotes for exact phrases)
        collection (int): MediaCloud collection ID
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        api_key (str): MediaCloud API key

    Returns:
       dataframe with columns [date, total_count, count, ratio, query collection]

    Usage Example:
      US_NATIONAL_COLLECTION = 34412234
      df = query_count_over_time(
        query='"climate change"',  # Note double quotes for exact phrase
        collection=US_NATIONAL_COLLECTION,
        start_date="2024-01-01",
        end_date="2024-01-31",
        api_key='yourkey'
      )

    Useful links about MC collections:

    # General
    - This website lists all the different collections [https://search.mediacloud.org/collections]
    - This link explains how the partisan lists were constructed [https://www.mediacloud.org/blog/3-approaches-to-quantifying-us-partisanship-in-news-sources]

    # Specific collections useful to me
    - US national news = 34412234 [https://search.mediacloud.org/collections/34412234]

    Note: These are from "Berkman Klein 2019 Partisanship" on the Partisanship collections page
    - Tweeted Mostly by Followers of Liberal Politicians 2019 = 200363061 [https://search.mediacloud.org/collections/200363061]
    - Tweeted Somewhat More by Followers of Liberal Politicians 2019 = 200363048 [https://search.mediacloud.org/collections/200363048]
    - Tweeted Evenly by Followers of Conservative & Liberal Politicians 2019 = 200363050 [https://sources.mediacloud.org/#/collections/200363050
    - Tweeted Somewhat More by Followers of Conservative Politicians 2019 = 200363062 [https://sources.mediacloud.org/#/collections/200363062]
    - Tweeted Mostly by Followers of Conservative Politicians 200363049 [https://sources.mediacloud.org/#/collections/200363049]
    """
    mc_search = mediacloud.api.SearchApi(api_key)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    try:
        results = mc_search.story_count_over_time(
            query, start_dt, end_dt, collection_ids=[collection]
        )

        df = pd.DataFrame.from_dict(results)
        df["date"] = pd.to_datetime(df["date"])
        df["query"] = query
        df["collection"] = collection
        print(df)

        return df

    except Exception as e:
        print(f"Error fetching story counts for query '{query}': {e}")
        return None


def multiple_query_count_over_time(
        queries, collection, start_date, end_date, api_key, n_jobs
) -> pd.DataFrame:
    """
    Get story counts for multiple queries, optionally in parallel.

    Args:
        queries: List of query strings
        collection: MediaCloud collection ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: MediaCloud API key
        n_jobs: Number of parallel jobs. If None, runs sequentially.
               If -1, uses all available cores. Else, uses `n_jobs' cores.

    Returns:
        Combined DataFrame with all query results

    Usage Example:
      US_NATIONAL_COLLECTION = 34412234
      API_KEY = "yourkey"
      queries = [
          '"critical race theory"',
          '"affirmative action"',
          '"environmental justice"',
          "'fire safety'",
          "'weather forecasting'",
          "'nutrition science'"
      ]

      df_parallel = multiple_query_count_over_time(
          queries=queries,
          collection=US_NATIONAL_COLLECTION,
          start_date="2022-01-01",
          end_date="2024-11-20",
          api_key=API_KEY,
          n_jobs=-1
      )
    """

    def process_query(query):
        return query_count_over_time(
            query=query,
            collection=collection,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

    if n_jobs is None:
        all_dfs = []
        for query in tqdm(queries, desc="Processing queries"):
            df = process_query(query)
            if df is not None:
                all_dfs.append(df)
    else:
        all_dfs = Parallel(n_jobs=n_jobs)(
            delayed(process_query)(query)
            for query in tqdm(
                queries, desc=f"Processing queries in parallel (n_jobs={n_jobs})"
            )
        )
        all_dfs = [df for df in all_dfs if df is not None]

    if not all_dfs:
        print("No successful query results")
        return pd.DataFrame()

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    return final_df


def dict_kw_list_over_time(kw_dict, collection, start_date, end_date, api_key, n_jobs):
    """
    Get story counts for multiple queries across multiple categories.

    This is a wrapper around `multiple_query_count_over_time` that allows you to
    input a dictionary of categories and their associated queries. For example,

    d =  {
        'ai': ['ai','chatbot', 'artificial intelligence'],
        'ai_social': ['social ai]
    }

    df = dict_kw_list_over_time(.., d)
    This would return a df with cols:
    [date, total_count, count, ratio, query, collection, category]

    Where `category` is the key in the dictionary.

    Args:
        kw_dict: Dictionary of categories and their associated queries
        collection: MediaCloud collection ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: MediaCloud API key
        n_jobs: Number of parallel jobs. If None, runs sequentially.
               If -1, uses all available cores. Else, uses `n_jobs' cores.

    """
    dfs = []
    for kw, kw_list in kw_dict.items():

        df = multiple_query_count_over_time(
            queries=kw_list,
            collection=collection,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
            n_jobs=n_jobs,
        )
        df["category"] = kw
        if df is not None:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def flatten_list(list_of_lists: List[List]) -> List:
    """
    Simple func to flatten a list of lists

    Args:
        list_of_lists: List of lists to flatten

    Returns:
        Flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]


def long2wide(
        tdf: pd.DataFrame,
        date_col: str = "date",
        cat_col: str = "cat",
        count_col: str = "count",
        how="sum",
) -> pd.DataFrame:
    """Convert long format DataFrame to wide format using pivot.

    This is useful for if you have multiple terms per category, and each day you
    want sum of counts per category. Or the category can be query.

    Args:
        tdf: Input DataFrame in long format
        date_col: Name of the date column
        cat_col: Name of the category column
        count_col: Name of the count/value column
        how: How to aggregate the counts. Can be 'sum', 'mean', 'median'

    Returns:
        DataFrame in wide format with daily sums per category

    Usage:
      Let's say you have a df like and are doing sum option:
      [date, query, cat, count] --> will pivot to be
      [date, sum(count|cat1), sum(count|cat2)...]

      Example:
      Input:
        date       cat    count
        2024-01-01 cat1   10
        2024-01-01 cat1   5
        2024-01-01 cat2   7
        2024-01-02 cat1   3

      Output:
        date       cat1  cat2
        2024-01-01 15    7
        2024-01-02 3     0
    """
    assert how in ["sum", "mean", "median"], "Bad option!! see docs"

    grouped = tdf.groupby([date_col, cat_col])[count_col].agg(how).reset_index()
    result = grouped.pivot(
        index=date_col, columns=cat_col, values=count_col
    ).reset_index()
    result[date_col] = pd.to_datetime(result[date_col])
    # fill NaN values with 0 for dates where a category had no entries
    return result.fillna(0)
