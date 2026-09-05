from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd


DEFAULT_CONFIG: Dict[str, Any] = {
    "initial_capital": 10_000_000.0,
    "moving_average_days": 120,
    "entry_exit_buffer": 0.02,
    "realized_volatility_days": 20,
    "target_annual_volatility": 0.80,
    "minimum_exposure": 0.25,
    "maximum_exposure": 1.0,
    "transaction_cost_per_turnover": 0.001,
    "annual_financing_rate": 0.10,
}


@dataclass
class Adaptive120Result:
    daily: pd.DataFrame
    metrics: Dict[str, float]


def prepare_signals(data: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    cfg = {**DEFAULT_CONFIG, **(dict(config) if config else {})}
    required = ["open", "high", "low", "close", "volume"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    frame = data.copy().sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame[required] = frame[required].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=required)

    ma_days = int(cfg["moving_average_days"])
    vol_days = int(cfg["realized_volatility_days"])
    buffer = float(cfg["entry_exit_buffer"])
    frame["sma"] = frame["close"].rolling(ma_days).mean()
    frame["upper_band"] = frame["sma"] * (1.0 + buffer)
    frame["lower_band"] = frame["sma"] * (1.0 - buffer)
    frame["realized_volatility"] = (
        frame["close"].pct_change().rolling(vol_days).std(ddof=1) * np.sqrt(365.0)
    )

    enter = frame["close"] > frame["upper_band"]
    exit_ = frame["close"] < frame["lower_band"]
    trend_state = pd.Series(np.nan, index=frame.index, dtype=float)
    trend_state.loc[enter] = 1.0
    trend_state.loc[exit_] = 0.0
    frame["trend_state"] = trend_state.ffill().fillna(0.0)

    volatility_multiplier = (
        float(cfg["target_annual_volatility"]) / frame["realized_volatility"]
    ).clip(
        lower=float(cfg["minimum_exposure"]),
        upper=float(cfg["maximum_exposure"]),
    )
    frame["desired_exposure"] = (frame["trend_state"] * volatility_multiplier).fillna(0.0)
    frame["signal"] = np.select(
        [enter, exit_], ["ENTER_OR_HOLD", "EXIT_OR_CASH"], default="HYSTERESIS_HOLD"
    )
    return frame


def _metrics(returns: pd.Series, position: pd.Series) -> Dict[str, float]:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    years = max((wealth.index[-1] - wealth.index[0]).days / 365.25, 1 / 365.25)
    volatility = returns.std(ddof=1) * np.sqrt(365.0)
    total_return = float(wealth.iloc[-1] - 1.0)
    return {
        "total_return": total_return,
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "annualized_volatility": float(volatility),
        "sharpe_zero_rf": float(returns.mean() * 365.0 / volatility) if volatility > 0 else float("nan"),
        "average_exposure": float(position.mean()),
    }


def backtest(
    data: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    evaluation_start: str = "2018-05-01",
    evaluation_end: str | None = None,
) -> Adaptive120Result:
    """Run a next-open backtest of the selected adaptive 120-day strategy."""
    cfg = {**DEFAULT_CONFIG, **(dict(config) if config else {})}
    frame = prepare_signals(data, cfg)
    if len(frame) < int(cfg["moving_average_days"]) + 2:
        raise ValueError("Not enough history for the selected moving average")

    # A signal observed at today's completed close becomes the position at the
    # following day's open. The position then earns the next open-to-open return.
    frame["position"] = frame["desired_exposure"].shift(1).fillna(0.0)
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["forward_open_return"] = frame["open"].shift(-1) / frame["open"] - 1.0
    frame["transaction_cost"] = (
        frame["turnover"] * float(cfg["transaction_cost_per_turnover"])
    )
    frame["financing_cost"] = (
        frame["position"].sub(1.0).clip(lower=0.0)
        * float(cfg["annual_financing_rate"])
        / 365.0
    )
    frame["strategy_return"] = (
        frame["position"] * frame["forward_open_return"].fillna(0.0)
        - frame["transaction_cost"]
        - frame["financing_cost"]
    )
    frame["buy_hold_return"] = frame["forward_open_return"].fillna(0.0)

    end = evaluation_end or frame.index[-1].strftime("%Y-%m-%d")
    sample = frame.loc[evaluation_start:end].copy()
    if sample.empty:
        raise ValueError("The evaluation range contains no rows")
    # Treat each requested evaluation window as a fresh portfolio. Charge the
    # initial allocation cost even when the trend state was established before
    # the selected start date; buy-and-hold pays the same one-time entry cost.
    initial_cost_rate = float(cfg["transaction_cost_per_turnover"])
    sample.iloc[0, sample.columns.get_loc("strategy_return")] -= (
        float(sample["position"].iloc[0]) * initial_cost_rate
    )
    sample.iloc[0, sample.columns.get_loc("buy_hold_return")] -= initial_cost_rate
    sample["strategy_equity"] = (
        float(cfg["initial_capital"]) * (1.0 + sample["strategy_return"]).cumprod()
    )
    sample["buy_hold_equity"] = (
        float(cfg["initial_capital"]) * (1.0 + sample["buy_hold_return"]).cumprod()
    )

    strategy_metrics = _metrics(sample["strategy_return"], sample["position"])
    hold_metrics = _metrics(sample["buy_hold_return"], pd.Series(1.0, index=sample.index))
    metrics = {
        **{f"strategy_{key}": value for key, value in strategy_metrics.items()},
        **{f"buy_hold_{key}": value for key, value in hold_metrics.items()},
        "evaluation_start": sample.index[0].strftime("%Y-%m-%d"),
        "evaluation_end": sample.index[-1].strftime("%Y-%m-%d"),
    }
    return Adaptive120Result(daily=sample, metrics=metrics)
