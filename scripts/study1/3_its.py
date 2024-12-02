"""
Author: Joshua Ashkinaze

Description: Runs OLS and ARIMAX models on the AI Social data for an ITS.

Date: 2024-11-29 22:06:50
"""

import numpy as np
from stargazer.stargazer import Stargazer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pmdarima as pm
from src.helpers import make_aesthetic

mypal = make_aesthetic()


HOLIDAYS = [False, True]
MODES = ["daily", "weekly"]


def add_shocks(df, shock_list, pulse_window=0, types=None):
    """Add shock variables to dataframe"""
    df = df.copy()

    if types is None:
        types = ["level", "pulse", "trend", "pre_trend"]

    df["time"] = (df.index - df.index.min()).days

    for shock in shock_list:
        name = shock["name"]
        date = pd.to_datetime(shock["date"])

        if "level" in types:
            df[f"{name}_level"] = (df.index >= date).astype(int)

        if "pulse" in types:
            if pulse_window == 0:
                df[f"{name}_pulse"] = (df.index == date).astype(int)
            else:
                end_date = date + pd.Timedelta(days=pulse_window)
                df[f"{name}_pulse"] = (
                    (df.index >= date) & (df.index < end_date)
                ).astype(int)

        if "trend" in types:
            df[f"{name}_trend"] = np.where(df.index >= date, (df.index - date).days, 0)

        if "pre_trend" in types:
            df[f"{name}_pretrend"] = np.where(
                df.index < date, (date - df.index).days, 0
            )

    return df


def handle_dates(df, mode="daily", add_holidays=False):
    """Add holiday indicators and aggregate to weekly if specified."""

    def get_thanksgiving_range(year):
        """Get Wednesday-Sunday range around Thanksgiving."""
        nov_first = pd.Timestamp(f"{year}-11-01")
        thurs = nov_first + pd.Timedelta(days=(24 - nov_first.weekday()))
        return (thurs - pd.Timedelta(days=1), thurs + pd.Timedelta(days=2))

    df = df.copy()

    # Dict of holidays with date ranges
    holidays = {
        "Christmas": lambda y: (f"{y}-12-24", f"{y}-12-26"),
        "NewYears": lambda y: (f"{y}-12-31", f"{y + 1}-01-02"),
        "Thanksgiving": lambda y: get_thanksgiving_range(y),
    }

    if mode == "daily":
        if add_holidays:
            years = df.index.year.unique()
            for holiday in holidays:
                df[f"Holiday{holiday}"] = 0
                for year in years:
                    start, end = holidays[holiday](year)
                    df.loc[
                        (df.index >= start) & (df.index <= end), f"Holiday{holiday}"
                    ] = 1
        else:
            pass

    if mode == "weekly":
        # Reset index to access date
        df = df.reset_index()
        df["week"] = df["date"].dt.isocalendar().week
        df["year"] = df["date"].dt.year

        if add_holidays:
            # When you group by week its not obvious what to do
            # with holiday indicators so I just take the max, meaning
            # if any day in the week is a holiday, the week is a holiday
            agg_dict = {
                "social_prop": "mean",
                "HolidayChristmas": "max",
                "HolidayNewYears": "max",
                "HolidayThanksgiving": "max",
            }
        else:
            agg_dict = {"social_prop": "mean"}
        df = df.groupby(["year", "week"]).agg(agg_dict)
        df = df.reset_index()
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["week"].astype(str) + "-1",
            format="%Y-%W-%w",
        )
        df = df.set_index("date")
        df["trend"] = range(1, len(df) + 1)

    return df


def summarize_ts_fit(dates, y, yhat):
    """
    Summarize time series fit with common metrics and plots

    Args:
        dates: array of dates
        y: actual values
        yhat: predicted values

    Returns:
        dict with metrics
    """

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, yhat))
    mae = mean_absolute_error(y, yhat)
    mape = np.mean(np.abs((y - yhat) / y)) * 100
    r2 = r2_score(y, yhat)

    # Plot fits
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y, "b.", alpha=0.5, label="Observed", markersize=2)
    plt.plot(dates, yhat, "r-", label="Fitted", alpha=0.2)
    plt.title("Time Series Fit")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot residuals
    residuals = y - yhat
    plt.figure(figsize=(12, 4))
    plt.plot(dates, residuals, "k.", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.title("Residuals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()

    # Return metrics
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


if __name__ == "__main__":

    for holiday in HOLIDAYS:

        for mode in MODES:

            ###################################
            print(f"\nRunning ITS for AI Social Data ({mode}) and Holiday={holiday}")
            ###################################

            # Load and prepare data
            ###################################
            ###################################
            df = pd.read_csv(
                "../../data/clean/wide_2020-01-01_2024-11-29_34412234_ai_social.csv"
            )
            df["date"] = pd.to_datetime(df["date"])
            df["trend"] = [i + 1 for i in range(len(df))]
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day

            # add holidays
            df = df.set_index("date")
            df = handle_dates(df, mode=mode, add_holidays=holiday)

            # Define interventions
            ###################################
            interventions = [
                {"name": "InstructGPT Paper", "date": "2022-01-27", "color": mypal[1]},
                {"name": "ChatGPT Release", "date": "2022-11-30", "color": mypal[2]},
            ]

            df = add_shocks(df, interventions)

            # Create matrix
            ###################################
            shock_cols = [c for c in df.columns if "_level" in c]
            holidays = [c for c in df.columns if "Holiday" in c]
            X = df[shock_cols + ["trend"] + holidays]
            y = df["social_prop"]

            # 1. Simple OLS
            ###################################
            ols_model = sm.OLS(y, sm.add_constant(X)).fit()

            # 2. Auto ARIMA model
            ###################################
            arima_model = pm.auto_arima(
                y,
                X=X,
                max_d=3,
                max_p=3,
                max_q=3,
                start_p=0,
                start_q=0,
                trace=True,
                error_action="ignore",
                random_state=42,
                suppress_warnings=True,
            )

            # Plot results
            ###################################
            ###################################
            plt.figure(figsize=(12, 6), facecolor="white")
            # Cool hack I found: Add "_nolegend_" to not plot actual dots but then plot empty arrays and make bigger
            # to control legend size w/o changing dot size for real
            plt.plot(
                df.index,
                y,
                ".",
                color="gray",
                alpha=0.7,
                markersize=1 if mode != "weekly" else 5,
                label="_nolegend_",
            )
            plt.plot(
                [],
                [],
                ".",
                color="gray",
                alpha=0.7 if mode == "weekly" else 0.5,
                markersize=10,
                label=f'Raw Data {"(Weekly Mean)" if mode == "weekly" else "(Daily)"}',
            )

            plt.plot(
                df.index,
                ols_model.fittedvalues,
                color=mypal[0],
                label=f"OLS Predictions",
                alpha=0.9,
                linewidth=2.5,
                linestyle=":",
            )
            plt.plot(
                df.index,
                arima_model.predict(n_periods=len(df), X=X),
                color=mypal[0],
                label=f"ARIMAX Predictions",
                alpha=0.9,
                linewidth=2.5,
                linestyle="--",
            )
            for shock in interventions:
                plt.axvline(
                    x=pd.to_datetime(shock["date"]),
                    color=shock["color"],
                    linestyle="--",
                    alpha=0.4,
                    linewidth=2,
                    label=shock["name"],
                )

            # mode_str = "(Weekly)" if mode == "weekly" else "(Daily)"
            plt.title(
                f"ITS Analysis: Proportion of AI News Stories Containing\nSocial Role or Social Function Phrases",
                fontweight="bold",
            )
            plt.xlabel("Date")
            plt.ylabel("Proportion")
            plt.legend(facecolor="white", edgecolor="none", loc="upper left")
            plt.tight_layout()
            plt.savefig(f"../../plots/its_{mode}_Holiday{str(holiday)}ai_social.png", dpi=300)
            ###################################
            ###################################

            # Print results
            ###################################
            ###################################
            print("\nOLS Results:")
            print(ols_model.summary())
            print("R2:", ols_model.rsquared)

            print("\nARIMA Results:")
            print("Model order:", arima_model.order)
            print("\nCoefficients:")
            print(arima_model.arima_res_.params.round(4))
            print(arima_model.summary())
            print("R2:", r2_score(y, arima_model.fittedvalues()))
            ###################################
            ###################################

            # Summarize fits
            ###################################
            summarize_ts_fit(df.index, ols_model.fittedvalues, y)
            summarize_ts_fit(df.index, arima_model.fittedvalues(), y)
            ###################################
