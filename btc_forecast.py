"""BTC Price Prediction and Forecasting Script.

This module downloads historical Bitcoin price data and trains a machine
learning model to predict the next day's closing price. It then produces a
multi-day forecast using an iterative approach.

Example usage:
    python btc_forecast.py --period 3y --forecast-horizon 7

Requirements:
    pip install pandas numpy yfinance scikit-learn matplotlib

The script is intentionally lightweight and designed for demonstration
purposes. For production use, consider model tuning, hyperparameter
search, and more rigorous validation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import yfinance as yf


DEFAULT_TICKER = "BTC-USD"
DEFAULT_PERIOD = "5y"
DEFAULT_FORECAST_HORIZON = 7
RANDOM_STATE = 42


@dataclass
class ForecastResult:
    """Represents a single forecasted data point."""

    date: pd.Timestamp
    predicted_close: float
    predicted_return_pct: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "predicted_close": round(self.predicted_close, 2),
            "predicted_return_pct": round(self.predicted_return_pct, 2),
        }


def download_price_history(ticker: str, period: str) -> pd.DataFrame:
    """Download historical price data from Yahoo Finance."""
    data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(
            "No price data returned. Please verify the ticker and period arguments."
        )
    data = data.sort_index()
    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features needed for training."""
    df = data.copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_7d"] = df["Close"].pct_change(7)
    df["SMA_7"] = df["Close"].rolling(window=7).mean()
    df["SMA_21"] = df["Close"].rolling(window=21).mean()
    df["Volatility_7"] = df["Return_1d"].rolling(window=7).std()
    df["Target"] = df["Close"].shift(-1)  # next day's close
    df = df.dropna()
    return df


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split feature columns and target."""
    feature_cols = [
        "Close",
        "Volume",
        "Return_1d",
        "Return_7d",
        "SMA_7",
        "SMA_21",
        "Volatility_7",
    ]
    X = df[feature_cols]
    y = df["Target"]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def iterative_forecast(
    model: RandomForestRegressor,
    recent_data: pd.DataFrame,
    horizon: int,
) -> List[ForecastResult]:
    """Generate a forecast by iteratively predicting the next day."""
    forecast_results: List[ForecastResult] = []
    df = recent_data.copy()

    last_date = df.index[-1]
    for step in range(1, horizon + 1):
        features = df.iloc[[-1]][[
            "Close",
            "Volume",
            "Return_1d",
            "Return_7d",
            "SMA_7",
            "SMA_21",
            "Volatility_7",
        ]]
        next_close = float(model.predict(features)[0])

        next_date = last_date + dt.timedelta(days=1)
        predicted_return = (next_close - df.iloc[-1]["Close"]) / df.iloc[-1]["Close"] * 100

        forecast_results.append(
            ForecastResult(
                date=next_date,
                predicted_close=next_close,
                predicted_return_pct=predicted_return,
            )
        )

        # Update dataframe to include the predicted day for iterative forecasting
        next_row = df.iloc[[-1]].copy()
        next_row.index = [next_date]
        next_row.loc[next_date, "Close"] = next_close
        next_row.loc[next_date, "Target"] = np.nan
        df = pd.concat([df, next_row])

        # Recalculate rolling features for the new row
        df.loc[next_date, "Return_1d"] = df.loc[next_date, "Close"] / df.iloc[-2]["Close"] - 1
        df.loc[next_date, "Return_7d"] = (
            df.loc[next_date, "Close"] / df.iloc[-7]["Close"] - 1
            if len(df) >= 7
            else df.loc[next_date, "Return_1d"]
        )
        df.loc[next_date, "SMA_7"] = df["Close"].tail(7).mean()
        df.loc[next_date, "SMA_21"] = df["Close"].tail(min(21, len(df))).mean()
        df.loc[next_date, "Volatility_7"] = df["Return_1d"].tail(7).std()

        last_date = next_date

    return forecast_results


def print_evaluation(metrics: dict[str, float]) -> None:
    print("Model evaluation on hold-out set:")
    for name, value in metrics.items():
        print(f"  {name}: {value:,.4f}")


def print_forecast(results: Iterable[ForecastResult]) -> None:
    print("\nForecast:")
    for result in results:
        d = result.as_dict()
        print(
            f"  {d['date']}: ${d['predicted_close']:,} (Return: {d['predicted_return_pct']}%)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker symbol to download (default: BTC-USD)",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="Period of historical data to download (e.g., 1y, 5y, max).",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=DEFAULT_FORECAST_HORIZON,
        help="Number of days to forecast ahead.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for the evaluation set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_data = download_price_history(args.ticker, args.period)
    feature_data = engineer_features(raw_data)
    X, y = split_features_targets(feature_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, shuffle=False
    )

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    print_evaluation(metrics)

    recent_data = feature_data.iloc[-60:].copy()
    forecast_results = iterative_forecast(model, recent_data, args.forecast_horizon)
    print_forecast(forecast_results)


if __name__ == "__main__":
    main()
