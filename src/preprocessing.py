"""
preprocessing.py

Key job: distinguish real "no demand" days from "stockout" days.
A stockout is when sell_price exists (item was sellable) but sales == 0
for an unusually long stretch compared to that item's normal pattern.

Logic used here (simple, explainable version):
- If sales == 0 AND sell_price is NaN -> item not even being sold that day, drop/ignore
- If sales == 0 AND sell_price exists AND the zero-streak length is unusually long
  (longer than `stockout_streak_threshold` days) -> flag as likely stockout
- Otherwise, zero sales with a normal/short streak = genuine low demand, keep as-is
"""

import pandas as pd
import numpy as np


def add_zero_streak_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each item_id/store_id series, compute the length of consecutive
    zero-sales streaks. Needed to tell a short natural lull apart from
    a long stockout gap.
    """
    df = df.sort_values(["item_id", "store_id", "date"]).copy()

    def streak_lengths(sales_series: pd.Series) -> pd.Series:
        is_zero = (sales_series == 0).astype(int)
        # group consecutive equal values
        group_id = (is_zero != is_zero.shift()).cumsum()
        streak_len = is_zero.groupby(group_id).transform("size") * is_zero
        return streak_len

    df["zero_streak_len"] = (
        df.groupby(["item_id", "store_id"])["sales"]
        .apply(streak_lengths)
        .reset_index(drop=True)
    )
    return df


def flag_stockouts(df: pd.DataFrame, stockout_streak_threshold: int = 14) -> pd.DataFrame:
    """
    Add a boolean column `is_stockout` marking rows likely caused by
    the item being out of stock rather than genuinely no demand.
    """
    df = add_zero_streak_length(df)

    df["is_stockout"] = (
        (df["sales"] == 0)
        & (df["sell_price"].notna())
        & (df["zero_streak_len"] >= stockout_streak_threshold)
    )

    return df


def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning step before feature engineering / modeling:
    - flags stockouts
    - drops rows where item wasn't sellable at all (sell_price is NaN)
    - keeps stockout flag as a feature rather than dropping those rows
      (model should learn that demand was suppressed, not absent)
    """
    df = flag_stockouts(df)

    before = len(df)
    df = df[df["sell_price"].notna()].copy()
    after = len(df)
    print(f"Dropped {before - after} rows where item wasn't sellable (no price).")

    n_stockout_days = df["is_stockout"].sum()
    print(f"Flagged {n_stockout_days} rows as likely stockouts "
          f"({n_stockout_days / len(df) * 100:.2f}% of remaining data).")

    return df


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/ca1_foods.parquet")
    df_clean = clean_for_modeling(df)
    df_clean.to_parquet("data/processed/ca1_foods_clean.parquet")
    print("Saved cleaned data to data/processed/ca1_foods_clean.parquet")
    print(df_clean[["item_id", "date", "sales", "sell_price", "zero_streak_len", "is_stockout"]].head(20))