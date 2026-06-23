"""
feature_engineering.py

Turns the cleaned time series into a supervised-learning-ready table by adding:
- Lag features        : sales N days ago (model learns from recent history)
- Rolling features     : rolling mean/std of sales (captures recent trend/volatility)
- Calendar features    : day of week, month, is_weekend, is_event, is_snap
- Price features        : price change indicator (did a promo happen?)

These features are what let a generic regression model (LightGBM) understand
sequential/time patterns, even though it has no built-in notion of "time."
"""

import pandas as pd
import numpy as np


LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sales_lag_N columns: sales from N days ago, per item/store."""
    df = df.sort_values(["item_id", "store_id", "date"]).copy()
    grouped = df.groupby(["item_id", "store_id"])["sales"]

    for lag in LAGS:
        df[f"sales_lag_{lag}"] = grouped.shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean/std of sales over past N days.
    Uses shift(1) first so the rolling window only looks at PAST days,
    never the current day (avoids leaking the answer into the features).
    """
    df = df.sort_values(["item_id", "store_id", "date"]).copy()
    shifted = df.groupby(["item_id", "store_id"])["sales"].shift(1)

    for window in ROLLING_WINDOWS:
        df[f"sales_rollmean_{window}"] = (
            shifted.groupby([df["item_id"], df["store_id"]])
            .rolling(window)
            .mean()
            .reset_index(drop=True)
        )
        df[f"sales_rollstd_{window}"] = (
            shifted.groupby([df["item_id"], df["store_id"]])
            .rolling(window)
            .std()
            .reset_index(drop=True)
        )

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple calendar-derived features."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month
    df["is_event"] = df["event_name_1"].notna().astype(int)
    # SNAP flag depends on state - CA store uses snap_CA
    df["is_snap"] = df["snap_CA"]  # adjust column if using a different state's store
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag price changes (proxy for promotions)."""
    df = df.sort_values(["item_id", "store_id", "date"]).copy()
    df["price_change"] = (
        df.groupby(["item_id", "store_id"])["sell_price"]
        .diff()
        .fillna(0)
    )
    df["is_promo"] = (df["price_change"] < 0).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in order."""
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_price_features(df)

    # Drop early rows where lag/rolling features are NaN (no history yet)
    before = len(df)
    df = df.dropna(subset=[f"sales_lag_{max(LAGS)}", f"sales_rollmean_{max(ROLLING_WINDOWS)}"])
    after = len(df)
    print(f"Dropped {before - after} rows with insufficient history for lag/rolling features.")

    return df


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/ca1_foods_clean.parquet")
    df_feat = build_features(df)
    df_feat.to_parquet("data/processed/ca1_foods_features.parquet")
    print(f"Final feature set shape: {df_feat.shape}")
    feature_cols = [c for c in df_feat.columns if "lag" in c or "roll" in c or c in
                     ["day_of_week", "is_weekend", "month", "is_event", "is_snap", "is_promo"]]
    print("Feature columns created:", feature_cols)
    print(df_feat[["item_id", "date", "sales"] + feature_cols].head(10))