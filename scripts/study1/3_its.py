"""
Author: Joshua Ashkinaze

Description: Runs OLS and ARIMAX models on the AI Social data for an ITS.

Date: 2024-11-29 22:06:50
"""



import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pmdarima as pm
from src.helpers import make_aesthetic


mypal = make_aesthetic()

def add_shocks(df, shock_list, pulse_window=0, types=None):
    """Add shock variables to dataframe"""
    df = df.copy()

    if types is None:
        types = ['level', 'pulse', 'trend', 'pre_trend']

    df['time'] = (df.index - df.index.min()).days

    for shock in shock_list:
        name = shock['name']
        date = pd.to_datetime(shock['date'])

        if 'level' in types:
            df[f'{name}_level'] = (df.index >= date).astype(int)

        if 'pulse' in types:
            if pulse_window == 0:
                df[f'{name}_pulse'] = (df.index == date).astype(int)
            else:
                end_date = date + pd.Timedelta(days=pulse_window)
                df[f'{name}_pulse'] = ((df.index >= date) & (df.index < end_date)).astype(int)

        if 'trend' in types:
            df[f'{name}_trend'] = np.where(df.index >= date, (df.index - date).days, 0)

        if 'pre_trend' in types:
            df[f'{name}_pretrend'] = np.where(df.index < date, (date - df.index).days, 0)

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
    plt.plot(dates, y, 'b.', alpha=0.5, label='Observed', markersize=2)
    plt.plot(dates, yhat, 'r-', label='Fitted', alpha=0.2)
    plt.title('Time Series Fit')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot residuals
    residuals = y - yhat
    plt.figure(figsize=(12, 4))
    plt.plot(dates, residuals, 'k.', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Residuals Over Time')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.show()

    # Return metrics
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


# Load and prepare data
###################################
df = pd.read_csv("../../data/clean/wide_2020-01-01_2024-11-29_34412234_ai_social.csv")
df['date'] = pd.to_datetime(df['date'])
df['trend'] = [i+1 for i in range(len(df))]
df = df.set_index('date')

# Define interventions
###################################
interventions = [
    {"name": "InstructGPT Paper", "date": "2022-01-27", "color": mypal[1]},
    {"name": "ChatGPT Release", "date": "2022-11-30", "color": mypal[2]}
]

df = add_shocks(df, interventions)

# Create matrix
###################################
shock_cols = [c for c in df.columns if  "_level" in c]
X = df[shock_cols + ['trend']]
y = df['social_prop']

# 1. Simple OLS
###################################
ols_model = sm.OLS(y, sm.add_constant(X)).fit()

# 2. Auto ARIMA model
###################################
arima_model = pm.auto_arima(y,
                            X=X,
                            max_d=3,
                            max_p=3,
                            max_q=3,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True)

# Plot results
###################################
###################################
plt.figure(figsize=(12, 6), facecolor='white')
# Cool hack I found: Add "_nolegend_" to not plot actual dots but then plot empty arrays and make bigger
# to control legend size w/o changing dot size for real
plt.plot(df.index, y, '.', color='gray', alpha=0.7, markersize=1, label='_nolegend_')
plt.plot([], [], '.', color='gray', alpha=0.7, markersize=10, label='Raw Data (Daily)')

plt.plot(df.index, ols_model.fittedvalues, color=mypal[0], label='OLS Predictions',
         alpha=0.9, linewidth=2.5, linestyle='--')
plt.plot(df.index, arima_model.predict(n_periods=len(df), X=X), color=mypal[0],
         label='ARIMAX Predictions', alpha=0.9, linewidth=2.5, linestyle=':')
for shock in interventions:
    plt.axvline(x=pd.to_datetime(shock['date']),
                color=shock['color'],
                linestyle='--',
                alpha=0.4,
                linewidth=2,
                label=shock['name'])
plt.title('ITS Analysis: Proportion of AI News Stories Containing\nSocial Role or Social Function Phrases',
          fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Proportion')
plt.legend(facecolor='white',
           edgecolor='none', loc='upper left')
plt.tight_layout()
plt.savefig("../../plots/its_ai_social.png", dpi=300)
###################################
###################################

# Print results
###################################
###################################
print("\nOLS Results:")
print(ols_model.summary())

print("\nARIMA Results:")
print("Model order:", arima_model.order)
print("\nCoefficients:")
print(arima_model.arima_res_.params.round(4))
print(arima_model.summary())
###################################
###################################


# Summarize fits
###################################
###################################
summarize_ts_fit(df.index, ols_model.fittedvalues, y)
summarize_ts_fit(df.index, arima_model.fittedvalues(), y)
###################################
###################################

