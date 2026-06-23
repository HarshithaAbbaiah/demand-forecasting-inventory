"""
inventory.py

Converts demand forecasts into actionable inventory decisions:
- Safety stock : extra buffer stock to cover demand uncertainty during lead time
- Reorder point : the stock level at which you must place a new order
- Business impact simulation : compares stockout days avoided / overstock $ saved
  when using our forecast vs a naive forecast

Formulas used (standard inventory management formulas):
- Safety stock = Z * sigma_demand * sqrt(lead_time)
    Z = service-level z-score (e.g. 1.65 for 95% service level)
    sigma_demand = standard deviation of daily demand (forecast uncertainty/volatility)
    lead_time = days between placing an order and receiving stock

- Reorder point = (avg_daily_demand * lead_time) + safety_stock
"""

import pandas as pd
import numpy as np
from scipy.stats import norm


def compute_safety_stock(demand_std: float, lead_time_days: int, service_level: float = 0.95) -> float:
    """
    service_level=0.95 means we want to avoid stockouts 95% of the time.
    Higher service level = higher Z = more safety stock = costs more to hold,
    but fewer stockouts. This is a real business tradeoff, not a fixed answer.
    """
    z = norm.ppf(service_level)
    safety_stock = z * demand_std * np.sqrt(lead_time_days)
    return safety_stock


def compute_reorder_point(avg_daily_demand: float, lead_time_days: int, safety_stock: float) -> float:
    return (avg_daily_demand * lead_time_days) + safety_stock


def compute_item_inventory_plan(
    item_forecast: pd.Series,
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> dict:
    """
    Takes a forecasted demand series for ONE item (e.g. predicted daily sales
    for the next 28 days) and returns its safety stock + reorder point.
    """
    avg_daily_demand = item_forecast.mean()
    demand_std = item_forecast.std()

    safety_stock = compute_safety_stock(demand_std, lead_time_days, service_level)
    reorder_point = compute_reorder_point(avg_daily_demand, lead_time_days, safety_stock)

    return {
        "avg_daily_demand": round(avg_daily_demand, 2),
        "demand_std": round(demand_std, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
    }


def simulate_stockouts(actual_demand: np.ndarray, forecast: np.ndarray, starting_stock: float, lead_time_days: int):
    """
    Simple simulation: start with `starting_stock` units, subtract actual daily
    demand. If stock would hit 0 before a restock arrives, count it as a stockout day.
    Assumes a restock arrives exactly every `lead_time_days`, sized to the forecast's
    avg daily demand * lead_time (i.e., following the reorder point logic).
    """
    stock = starting_stock
    restock_qty = forecast.mean() * lead_time_days
    stockout_days = 0

    for day, demand in enumerate(actual_demand):
        if day % lead_time_days == 0 and day != 0:
            stock += restock_qty
        stock -= demand
        if stock < 0:
            stockout_days += 1
            stock = 0  # can't go negative, lost sales

    return stockout_days


if __name__ == "__main__":
    import lightgbm as lgb
    from models.lgbm_model import FEATURE_COLS, add_item_features, train_test_split_by_date

    df = pd.read_parquet("data/processed/ca1_foods_features.parquet")
    top_item = "FOODS_3_090"

    # Recreate the same train/test split and item features used during LightGBM training
    _, _, cutoff_date = train_test_split_by_date(df, test_days=28)
    df = add_item_features(df, train_cutoff_date=cutoff_date)
    train, test, _ = train_test_split_by_date(df, test_days=28)

    # Load the trained LightGBM model
    booster = lgb.Booster(model_file="outputs/lgbm_model.txt")

    item_test = test[test["item_id"] == top_item].sort_values("date")
    actual_demand = item_test["sales"].values

    pred_log = booster.predict(item_test[FEATURE_COLS])
    our_forecast = np.expm1(pred_log)
    our_forecast = np.clip(our_forecast, 0, None)

    naive_forecast = np.full(28, train[train["item_id"] == top_item]["sales"].tail(7).mean())

    plan = compute_item_inventory_plan(pd.Series(our_forecast), lead_time_days=7, service_level=0.95)
    print(f"Inventory plan for {top_item} (95% service level, 7-day lead time):")
    print(plan)

    stockouts_ours = simulate_stockouts(actual_demand, our_forecast, starting_stock=plan["reorder_point"], lead_time_days=7)
    naive_plan = compute_item_inventory_plan(pd.Series(naive_forecast), lead_time_days=7, service_level=0.95)
    stockouts_naive = simulate_stockouts(actual_demand, naive_forecast, starting_stock=naive_plan["reorder_point"], lead_time_days=7)

    print(f"\nSimulated stockout days over 28 days:")
    print(f"  Using our (LightGBM) forecast-based reorder point: {stockouts_ours} stockout days")
    print(f"  Using naive forecast-based reorder point: {stockouts_naive} stockout days")