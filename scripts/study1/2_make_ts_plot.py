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
from statsmodels.nonparametric.smoothers_lowess import lowess

logging.basicConfig(
    filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    filemode="w",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# Parameters
###################

font_scale = 2.2
mypal = make_aesthetic(font_scale=font_scale)
ai_events = [
    {
        "name": "InstructGPT",
        "date": "2022-1-27",
        "color": "#D41876",
        "label": "InstructGPT Paper",
    },
    {
        "name": "ChatGPT Release",
        "date": "2022-11-30",
        "color": "#00A896",
        "label": "ChatGPT Release",
    },
]
logging.info("Parameters...")
logging.info("font_scale: %s", font_scale)
logging.info("ai_events: %s", ai_events)


###################


def smooth_df(df, dv, events, ndays, method="ewm", frac=0.1):
    if method == "ewm":
        df["smooth"] = (
            df[dv]
            .ewm(
                span=ndays,
            )
            .mean()
        )

    elif method == "rolling":
        df["smooth"] = df[dv].rolling(window=ndays, center=True).mean()

    elif method == "loess":
        # Convert dates to numeric values for LOESS
        x = (df["date"] - df["date"].min()).dt.total_seconds().values
        y = df[dv].values

        # Perform LOESS smoothing
        smoothed = lowess(
            y,
            x,
            frac=frac,  # Fraction of data used for local regression
            it=3,  # Number of robustifying iterations
            return_sorted=False,
        )
        df["smooth"] = smoothed

    df["period"] = pd.cut(
        df["date"],
        bins=[pd.Timestamp.min]
        + [pd.Timestamp(e["date"]) for e in events]
        + [pd.Timestamp.max],
        labels=[f"Pre-{ai_events[0]['name']}"] + [e["label"] for e in events],
    )
    return df


if __name__ == "__main__":
    mypal = make_aesthetic(font_scale=font_scale)
    df = pd.read_csv(
        "../../data/clean/wide_2014-11-30_2024-11-27_34412234_ai_social.csv"
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.query("plain + ai_social>0")

    n_days_list = [7, 14, 21, 30]
    method_names = {"ewm": "EWMA", "rolling": "MA"}
    dvs = {
        "social_prop": {
            "ylabel": "Proportion",
            "title": "Prop. of AI News Stories Containing AI Social Support or Social Role Phrases\n(e.g.: 'AI therapist', 'Chatbot coach') Across 249 U.S Newspapers",
        },
        "plain": {
            "ylabel": "Daily Count",
            "title": "Daily Count of AI News Stories Across 249 U.S Newspapers",
        },
        "ai_social": {
            "ylabel": "Daily Count",
            "title": "Daily Count of AI News Stories Containing AI Social Support or Social Role Phrases\n(e.g.: 'AI therapist', 'Chatbot coach') Across 249 U.S Newspapers",
        },
    }

    for dv, dv_data in dvs.items():
        for n_days in n_days_list:
            for method, method_name in method_names.items():
                df = smooth_df(df, ndays=n_days, events=ai_events, method=method, dv=dv)

                plt.figure(figsize=(18, 12))
                sns.scatterplot(
                    data=df,
                    x="date",
                    y="smooth",
                    hue="period",
                    palette=["black"] + [e["color"] for e in ai_events],
                    alpha=0.9,
                    s=50,
                )

                sns.lineplot(
                    data=df, x="date", y="smooth", color="black", alpha=0.1, linewidth=1
                )

                pre_gpt_levels = df[df["date"] < pd.Timestamp(ai_events[0]["date"])][
                    "social_prop"
                ].mean()
                plt.axhline(
                    pre_gpt_levels,
                    color="gray",
                    linestyle="-.",
                    label=f"Pre-{ai_events[0]['name']} Mean",
                    linewidth=2,
                )

                for event in ai_events:
                    plt.axvline(
                        x=pd.Timestamp(event["date"]),
                        color=event["color"],
                        linestyle="dashed",
                        linewidth=2,
                    )

                plt.ylabel(f"{dv_data['ylabel']}\n({n_days}-Day {method_name})")
                plt.xlabel("Date")
                plt.title(f"{dv_data['title']}", fontweight="bold", ha="left")
                plt.xticks(rotation=0)
                plt.legend(frameon=True, loc="upper left")
                plt.tight_layout()
                # plt.savefig(f"../../plots/ai_social_{n_days}_{method}_days.pdf", dpi=300)
                plt.savefig(f"../../plots/s1_{dv}_{n_days}_{method}_days.png", dpi=300)
