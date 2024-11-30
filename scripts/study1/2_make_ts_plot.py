"""
Author: Joshua Ashkinaze

Description: Nice time series plot of AI social role as prop of AI news stories

Input:
    - wide_2014-11-30_2024-11-27_34412234_ai_social.csv: Daily level AI newspaper data from MediaCloud

Output:
    Plots with fn like s1_[other params]

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
fn = "../../data/clean/wide_2020-01-01_2024-11-29_34412234_ai_social.csv"
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


def plot_multi_freq_trends():
    """
    Plot z-score trends with confidence intervals for AI-related phrases at multiple frequencies.
    """

    # Read in df
    dates = fn.split("_")[1:3]
    dates = "_".join(dates)
    df = pd.read_csv(fn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.query("plain + ai_social>0")
    for var in ["ai_social", "plain"]:
        df[f"{var}_z"] = (df[var] - df[var].mean()) / df[var].std()

    frequencies = [("Daily", "D"), ("Weekly", "W"), ("Monthly", "M"), ("Yearly", "Y")]

    fig, axes = plt.subplots(len(frequencies), 1, figsize=(24, 12))

    for idx, (title_prefix, freq) in enumerate(frequencies):
        grouped = (
            df.groupby(pd.Grouper(freq=freq))
            .agg(
                {
                    "ai_social_z": ["mean", "std", "count"],
                    "plain_z": ["mean", "std", "count"],
                }
            )
            .reset_index()
        )

        grouped.columns = [
            "date",
            "ai_social_mean",
            "ai_social_std",
            "ai_social_count",
            "plain_mean",
            "plain_std",
            "plain_count",
        ]

        # Calculate confidence intervals
        z = 1.96
        for var in ["ai_social", "plain"]:
            grouped[f"{var}_ci_upper"] = grouped[f"{var}_mean"] + z * grouped[
                f"{var}_std"
            ] / np.sqrt(grouped[f"{var}_count"])
            grouped[f"{var}_ci_lower"] = grouped[f"{var}_mean"] - z * grouped[
                f"{var}_std"
            ] / np.sqrt(grouped[f"{var}_count"])

        ax = axes[idx]

        # change to ax.scatter for scatterplot
        plot_func = ax.plot

        plot_func(
            grouped["date"],
            grouped["ai_social_mean"],
            label="Social AI Phrases",
            color="#D41876",
        )
        ax.fill_between(
            grouped["date"],
            grouped["ai_social_ci_lower"],
            grouped["ai_social_ci_upper"],
            alpha=0.2,
            color="#D41876",
        )

        plot_func(
            grouped["date"],
            grouped["plain_mean"],
            label="All AI Phrases",
            color="#00A896",
        )
        ax.fill_between(
            grouped["date"],
            grouped["plain_ci_lower"],
            grouped["plain_ci_upper"],
            alpha=0.2,
            color="#00A896",
        )

        title_str = f"{title_prefix} Trends"
        if freq != "D":
            title_str += " with 95% CIs"
        ax.set_title(title_str)
        ax.set_xlabel("")
        ax.set_ylabel("Z-Score")
        ax.legend(loc="upper left", edgecolor="white")
        ax.tick_params(axis="x", rotation=45)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"../../plots/all_trends_{dates}.png", dpi=300)
    return fig


if __name__ == "__main__":
    plot_multi_freq_trends()
    mypal = make_aesthetic(font_scale=font_scale)
    dates = fn.split("_")[1:3]
    dates = "_".join(dates)
    df = pd.read_csv(fn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.query("plain + ai_social>0")

    n_days_list = [7, 14, 21, 30]
    method_names = {"ewm": "EWMA", "rolling": "MA"}
    dvs = {
        "social_prop": {
            "ylabel": "Proportion",
            "title": "Prop. of AI News Stories Containing AI Social Support or Social Role Phrases\n(e.g.: 'AI therapist', 'Chatbot coach') Across 249 U.S. Newspapers",
        },
        "plain": {
            "ylabel": "Daily Count",
            "title": "Daily Count of AI News Stories Across 249 U.S. Newspapers",
        },
        "ai_social": {
            "ylabel": "Daily Count",
            "title": "Daily Count of AI News Stories Containing AI Social Support or Social Role Phrases\n(e.g.: 'AI therapist', 'Chatbot coach') Across 249 U.S. Newspapers",
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
                    dv
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
                plt.savefig(
                    f"../../plots/s1_{dv}_{n_days}_{method}_{dates}_days.png", dpi=300
                )
                plt.close()
