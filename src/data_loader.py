"""
data_loader.py

Loads and merges M5 forecasting data:
- sales_train_evaluation.csv : wide-format daily sales per item/store
- calendar.csv                : date features, events, SNAP flags
- sell_prices.csv             : weekly prices per item/store

Output: a single long-format dataframe with one row per (item, store, date).
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")


def load_raw_files():
    """Load the three raw M5 CSVs."""
    sales = pd.read_csv(RAW_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(RAW_DIR / "calendar.csv")
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv")
    return sales, calendar, prices


def melt_sales_to_long(sales: pd.DataFrame, id_cols=None) -> pd.DataFrame:
    """
    Convert wide sales (one column per day: d_1, d_2, ...) into long format:
    one row per item_id/store_id/day.
    """
    if id_cols is None:
        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    day_cols = [c for c in sales.columns if c.startswith("d_")]

    long_df = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    return long_df


def merge_all(filter_store_id: str = None, filter_cat_id: str = None) -> pd.DataFrame:
    """
    Full pipeline: load raw files, melt sales to long format, merge calendar + prices.

    Args:
        filter_store_id: optional, e.g. "CA_1" - restricts to one store to keep size manageable
        filter_cat_id: optional, e.g. "FOODS" - restricts to one category

    Returns:
        Long-format dataframe: date, item_id, store_id, sales, sell_price, calendar features
    """
    sales, calendar, prices = load_raw_files()

    if filter_store_id:
        sales = sales[sales["store_id"] == filter_store_id]
    if filter_cat_id:
        sales = sales[sales["cat_id"] == filter_cat_id]

    long_sales = melt_sales_to_long(sales)

    # Merge calendar to get actual dates + event/SNAP features
    long_sales = long_sales.merge(calendar, on="d", how="left")

    # Merge prices (keyed by item_id, store_id, wm_yr_wk)
    long_sales = long_sales.merge(
        prices, on=["item_id", "store_id", "wm_yr_wk"], how="left"
    )

    # Clean types
    long_sales["date"] = pd.to_datetime(long_sales["date"])
    long_sales = long_sales.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    return long_sales


if __name__ == "__main__":
    # Quick manual test: one store, one category, to keep it fast
    df = merge_all(filter_store_id="CA_1", filter_cat_id="FOODS")
    print(df.shape)
    print(df.head())
    df.to_parquet("data/processed/ca1_foods.parquet")
    print("Saved to data/processed/ca1_foods.parquet")