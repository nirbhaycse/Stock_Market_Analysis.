#!/usr/bin/env python3
"""Evaluate Prophet stock-price forecasts on a chronological holdout set.

Example:
    python evaluate_forecast.py --ticker AAPL --start 2016-01-01

The final 20% of observations are held out by date. No random shuffling is used.
This is a simple educational evaluation, not a trading strategy or financial advice.
"""

import argparse
import math

import pandas as pd
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def download_close_prices(ticker: str, start: str) -> pd.DataFrame:
    """Download adjusted historical prices and return Date/Close columns."""
    raw = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for {ticker!r}. Check the ticker and date range."
        )

    # yfinance may return MultiIndex columns, depending on its version/options.
    if isinstance(raw.columns, pd.MultiIndex):
        close_block = raw.xs("Close", axis=1, level=0)
        close = close_block.iloc[:, 0]
    else:
        if "Close" not in raw.columns:
            raise ValueError("The downloaded data does not contain a Close column.")
        close = raw["Close"]

    frame = pd.DataFrame({"ds": pd.to_datetime(close.index), "y": close.to_numpy()})
    frame = frame.dropna().sort_values("ds").drop_duplicates("ds").reset_index(drop=True)

    if len(frame) < 30:
        raise ValueError(
            f"Only {len(frame)} valid observations found; use a longer date range."
        )
    return frame


def calculate_metrics(actual, predicted) -> dict:
    actual = pd.Series(actual, dtype="float64")
    predicted = pd.Series(predicted, dtype="float64")
    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    nonzero = actual.abs() > 1e-12
    mape = (
        ((actual[nonzero] - predicted[nonzero]).abs() / actual[nonzero].abs()).mean()
        * 100
        if nonzero.any()
        else float("nan")
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Prophet against a naive previous-close baseline."
    )
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker (default: AAPL)")
    parser.add_argument("--start", default="2016-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of the latest observations reserved for testing (default: 0.2)",
    )
    args = parser.parse_args()

    if not 0.05 <= args.test_fraction <= 0.5:
        parser.error("--test-fraction must be between 0.05 and 0.5")

    prices = download_close_prices(args.ticker.upper(), args.start)
    split_at = int(len(prices) * (1 - args.test_fraction))
    train = prices.iloc[:split_at].copy()
    test = prices.iloc[split_at:].copy()

    if len(train) < 20 or test.empty:
        raise ValueError("Not enough observations for the requested chronological split.")

    model = Prophet()
    model.fit(train[["ds", "y"]])

    # Predict only on actual held-out trading dates; do not generate calendar-day dates.
    forecast = model.predict(test[["ds"]])
    prophet_predictions = forecast["yhat"].to_numpy()

    # A simple baseline: tomorrow's close is estimated as the previous observed close.
    all_prices = prices["y"].reset_index(drop=True)
    naive_predictions = all_prices.iloc[split_at - 1 : -1].to_numpy()

    actual = test["y"].to_numpy()
    prophet_metrics = calculate_metrics(actual, prophet_predictions)
    naive_metrics = calculate_metrics(actual, naive_predictions)

    print(f"Ticker: {args.ticker.upper()}")
    print(f"Date range: {prices['ds'].min().date()} to {prices['ds'].max().date()}")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")
    print(f"Test period: {test['ds'].min().date()} to {test['ds'].max().date()}")
    print("\nMetrics on chronological holdout (lower is better):")
    results = pd.DataFrame(
        {"Prophet": prophet_metrics, "Previous-close baseline": naive_metrics}
    )
    print(results.to_string(float_format=lambda value: f"{value:.4f}"))
    print(
        "\nInterpretation: compare both models on the same holdout period. "
        "These metrics do not establish profitability or predict future performance."
    )


if __name__ == "__main__":
    main()
