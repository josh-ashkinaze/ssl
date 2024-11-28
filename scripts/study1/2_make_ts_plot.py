"""
Author: Joshua Ashkinaze

Description: Nice time series plot of AI social role as prop of AI news stories

Input:
    - wide_2014-11-30_2024-11-27_34412234_ai_social.csv: Daily level AI newspaper data from MediaCloud

Output:
    Plots with fn like
    - ai_social_{n}_{ewm}_days.pdf: the ts plot

Date: 2024-11-27 18:28:39
"""

import os
from src.helpers import make_aesthetic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)



# Parameters
###################
n_days = 30
method = 'rolling'
font_scale = 2.2

mypal = make_aesthetic(font_scale=font_scale)
ai_events = [
    {
        'name': 'InstructGPT',
        'date': "2022-1-27",
        'color': "#D41876",
        'label': 'InstructGPT Paper'
    },
    {
        'name': 'ChatGPT Release',
        'date': "2022-11-30",
        'color': "#00A896",
        'label': 'ChatGPT Release'
    }
]
logging.info("Parameters...")
logging.info("n_days: %s", n_days)
logging.info("font_scale: %s", font_scale)
logging.info("ai_events: %s", ai_events)
###################


def smooth_df(df, events, ndays, method='ewm'):
    if method == 'ewm':
        df['smooth'] = df['social_prop'].ewm(
            span=ndays,
            adjust=False
        ).mean()

        df['se'] = df['social_prop'].ewm(
            span=ndays,
            adjust=False
        ).std() / np.sqrt(ndays)
    elif method == 'rolling':
        df['smooth'] = df['social_prop'].rolling(
            window=ndays,
            center=True
        ).mean()

        df['se'] = df['social_prop'].rolling(
            window=ndays,
            center=True
        ).std() / np.sqrt(ndays)

    df['ci_upper'] = df['smooth'] + 1.96 * df['se']
    df['ci_lower'] = df['smooth'] - 1.96 * df['se']

    df['period'] = pd.cut(df['date'],
                          bins=[pd.Timestamp.min] +
                               [pd.Timestamp(e['date']) for e in events] +
                               [pd.Timestamp.max],
                          labels=[f'Pre-{ai_events[0]['name']}'] + [e['label'] for e in events])
    return df

if __name__ == '__main__':
    mypal = make_aesthetic(font_scale=font_scale)
    df = pd.read_csv("../../data/clean/wide_2014-11-30_2024-11-27_34412234_ai_social.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.query("plain + ai_social>0")

    # make sure we get same kinda graph with different settings
    n_days = [7, 14, 21, 30]
    method_names = {
        'ewm': 'EWMA',
        'rolling': 'MA'
    }

    for n_days in n_days:
        for method, method_name in method_names.items():
            df = smooth_df(df, ndays=n_days, events=ai_events, method=method)

            plt.figure(figsize=(18, 12))
            sns.scatterplot(data=df,
                            x='date',
                            y='smooth',
                            hue='period',
                            palette=['black'] + [e['color'] for e in ai_events],
                            alpha=0.9,
                            s=50)

            plt.fill_between(df['date'],
                             df['ci_lower'],
                             df['ci_upper'],
                             alpha=0.1,
                             color='gray')

            sns.lineplot(data=df,
                         x='date',
                         y='smooth',
                         color='black',
                         alpha=0.1,
                         linewidth=1)

            pre_gpt_levels = df[df['date'] < pd.Timestamp(ai_events[0]['date'])]['social_prop'].mean()
            plt.axhline(pre_gpt_levels, color='gray', linestyle='-.', label=f'Pre-{ai_events[0]['name']} Mean', linewidth=2)

            for event in ai_events:
                plt.axvline(x=pd.Timestamp(event['date']),
                            color=event['color'],
                            linestyle='dashed',
                            linewidth=2)

            plt.ylabel(f'Proportion\n({n_days}-Day {method_name})')
            plt.xlabel('Date')
            plt.title(
                "Proportion of AI News Stories Containing AI Social Role Phrases\n(e.g: 'AI therapist', 'Chatbot coach') Across 249 U.S Newspapers",
                fontweight='bold',
                ha='left')
            plt.xticks(rotation=0)
            plt.legend(frameon=True,  loc='upper left')
            plt.tight_layout()
            plt.savefig(f"../../plots/ai_social_{n_days}_{method}_days.pdf", dpi=300)
            plt.savefig(f"../../plots/ai_social_{n_days}_{method}_days.png", dpi=300)

