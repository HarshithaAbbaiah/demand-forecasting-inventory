"""
sarima_model.py

SARIMA = Seasonal AutoRegressive Integrated Moving Average.
Classic statistical time series model - looks ONLY at the sales series itself
(no external features like price/events), learning patterns from its own past
values, trend, and seasonality.

We use a weekly seasonal period (s=7) since retail demand has strong
day-of-week patterns (weekends busier than weekdays).
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")  # SARIMA throws a lot of convergence warnings on noisy retail data


def fit_and_forecast_sarima(train_series: pd.Series, horizon: int):
    """
    Fit a SARIMA model and forecast `horizon` days ahead.

    order=(1,1,1): basic AR/I/MA terms - 1 lag of autoregression,
                   1 differencing to remove trend, 1 lag of moving average
    seasonal_order=(1,1,1,7): same idea but applied at the weekly (7-day) seasonal level
    """
    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=horizon)
    return forecast.values


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/ca1_foods_features.parquet")
    top_item = "FOODS_3_090"  # same item as baseline, for fair comparison

    item_df = df[df["item_id"] == top_item].sort_values("date")
    train = item_df.iloc[:-28]
    test = item_df.iloc[-28:]

    preds = fit_and_forecast_sarima(train["sales"], horizon=28)
    mae = np.mean(np.abs(test["sales"].values - preds))
    print(f"SARIMA MAE on last 28 days: {mae:.3f}")