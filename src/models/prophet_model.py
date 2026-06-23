"""
prophet_model.py

Prophet (by Meta) is built specifically for business time series:
- handles trend + multiple seasonalities (weekly, yearly) automatically
- lets us add custom regressors, like SNAP days and event flags,
  which SARIMA can't easily do

Prophet requires a specific input format: columns must be named
'ds' (date) and 'y' (value to forecast).
"""

import pandas as pd
import numpy as np
from prophet import Prophet


def fit_and_forecast_prophet(train_df: pd.DataFrame, horizon: int, future_regressors: pd.DataFrame):
    """
    train_df must have columns: ds, y, is_snap, is_event
    future_regressors must have the same regressor columns for the forecast horizon dates
    """
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    model.add_regressor("is_snap")
    model.add_regressor("is_event")

    model.fit(train_df)

    future = model.make_future_dataframe(periods=horizon)
    future = future.merge(future_regressors, on="ds", how="left")
    future[["is_snap", "is_event"]] = future[["is_snap", "is_event"]].fillna(0)

    forecast = model.predict(future)
    return forecast.tail(horizon)["yhat"].values


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/ca1_foods_features.parquet")
    top_item = "FOODS_3_090"

    item_df = df[df["item_id"] == top_item].sort_values("date")
    train = item_df.iloc[:-28]
    test = item_df.iloc[-28:]

    train_prophet = train[["date", "sales", "is_snap", "is_event"]].rename(
        columns={"date": "ds", "sales": "y"}
    )
    future_regressors = test[["date", "is_snap", "is_event"]].rename(columns={"date": "ds"})

    preds = fit_and_forecast_prophet(train_prophet, horizon=28, future_regressors=future_regressors)
    mae = np.mean(np.abs(test["sales"].values - preds))
    print(f"Prophet MAE on last 28 days: {mae:.3f}")