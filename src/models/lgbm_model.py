"""
lgbm_model.py

Unlike SARIMA/Prophet (one model per item), LightGBM trains ONE model
across ALL items at once, using the engineered features (lags, rolling
stats, price, calendar) as inputs. It learns general patterns like
"high lag_7 + SNAP day + weekend = higher demand" that transfer across items.

Train/test split: last 28 days (globally, across all items) held out as test,
same as SARIMA/Prophet, so comparisons are fair.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb


FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rollmean_7", "sales_rollstd_7",
    "sales_rollmean_14", "sales_rollstd_14",
    "sales_rollmean_28", "sales_rollstd_28",
    "day_of_week", "is_weekend", "month",
    "is_event", "is_snap", "is_promo", "sell_price",
    "item_id_encoded", "item_avg_sales",
]
TARGET_COL = "sales"


def add_item_features(df: pd.DataFrame, train_cutoff_date) -> pd.DataFrame:
    """
    Adds:
    - item_id_encoded: numeric code per item, so the model knows WHICH item it's predicting
    - item_avg_sales: that item's historical average (computed from train period only,
      to avoid leaking future info into the feature)
    """
    df = df.copy()
    df["item_id_encoded"] = df["item_id"].astype("category").cat.codes

    train_only = df[df["date"] <= train_cutoff_date]
    item_avg = train_only.groupby("item_id")["sales"].mean().rename("item_avg_sales")
    df = df.merge(item_avg, on="item_id", how="left")
    df["item_avg_sales"] = df["item_avg_sales"].fillna(df["sales"].mean())

    return df


def train_test_split_by_date(df: pd.DataFrame, test_days: int = 28):
    cutoff_date = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff_date]
    test = df[df["date"] > cutoff_date]
    return train, test, cutoff_date


def train_lgbm(train_df: pd.DataFrame):
    """Trains on log1p(sales) to reduce the dominance of high-volume items'
    large raw errors, then we invert with expm1 at prediction time."""
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        random_state=42,
        verbose=-1,
    )
    y_log = np.log1p(train_df[TARGET_COL])
    model.fit(train_df[FEATURE_COLS], y_log)
    return model


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/ca1_foods_features.parquet")

    train, test, cutoff_date = train_test_split_by_date(df, test_days=28)
    df = add_item_features(df, train_cutoff_date=cutoff_date)
    train, test, _ = train_test_split_by_date(df, test_days=28)
    print(f"Train rows: {len(train)}, Test rows: {len(test)}")

    model = train_lgbm(train)

    test = test.copy()
    pred_log = model.predict(test[FEATURE_COLS])
    test["pred"] = np.expm1(pred_log)  # invert log1p back to real sales scale
    test["pred"] = test["pred"].clip(lower=0)  # sales can't be negative

    overall_mae = np.mean(np.abs(test[TARGET_COL] - test["pred"]))
    print(f"LightGBM overall MAE (all items, last 28 days): {overall_mae:.3f}")

    top_item = "FOODS_3_090"
    item_test = test[test["item_id"] == top_item]
    item_mae = np.mean(np.abs(item_test[TARGET_COL] - item_test["pred"]))
    print(f"LightGBM MAE on {top_item} only (for fair comparison): {item_mae:.3f}")

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop 5 most important features:")
    print(importance.head(5))

    model.booster_.save_model("outputs/lgbm_model.txt")
    print("\nModel saved to outputs/lgbm_model.txt")