"""
baseline.py

Naive baseline forecast: predict tomorrow's sales = average of last 7 days.
Every "real" model (SARIMA, Prophet, LightGBM) must beat this to be worth using.
"""

import pandas as pd
import numpy as np


def naive_forecast(train_series: pd.Series, horizon: int) -> np.ndarray:
    """
    Predict the next `horizon` days as the mean of the last 7 observed days.
    Simplest possible forecast - the bar everything else must clear.
    """
    last_7_mean = train_series.tail(7).mean()
    return np.full(horizon, last_7_mean)


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/ca1_foods_features.parquet")

    # Pick the best-selling item to test on
    top_item = df.groupby("item_id")["sales"].sum().idxmax()
    print(f"Using top-selling item for model demo: {top_item}")

    item_df = df[df["item_id"] == top_item].sort_values("date")
    train = item_df.iloc[:-28]  # hold out last 28 days as test
    test = item_df.iloc[-28:]

    preds = naive_forecast(train["sales"], horizon=28)
    mae = np.mean(np.abs(test["sales"].values - preds))
    print(f"Naive baseline MAE on last 28 days: {mae:.3f}")