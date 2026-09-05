from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]


DEFAULT_CONFIG: Dict[str, Any] = {
    "initial_capital": 10_000_000.0,
    "fee_rate": 0.0005,
    "slippage_rate": 0.0005,
    "risk_per_trade": 0.005,
    "transition_exposure_cap": 0.35,
    "confirmed_exposure_cap": 0.60,
    "breakout_lookback": 20,
    "breakout_volume_multiple": 1.20,
    "breakout_stop_atr": 1.10,
    "pullback_stop_atr": 1.60,
    "trailing_stop_atr": 2.00,
    "partial_take_profit_r": 2.00,
    "partial_take_profit_fraction": 0.35,
    "shallow_rsi_min": 50.0,
    "shallow_rsi_max": 70.0,
    "deep_rsi_min": 45.0,
    "deep_rsi_max": 65.0,
}


def _validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    frame = data.copy().sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame[list(required)] = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=list(required))
    if frame.empty:
        raise ValueError("No valid OHLCV rows")
    return frame


def add_indicators(data: pd.DataFrame, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    cfg = {**DEFAULT_CONFIG, **(dict(config) if config else {})}
    frame = _validate_ohlcv(data)
    close = frame["close"]

    frame["sma10"] = close.rolling(10).mean()
    frame["sma20"] = close.rolling(20).mean()
    frame["sma50"] = close.rolling(50).mean()
    frame["sma200"] = close.rolling(200).mean()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr14"] = true_range.rolling(14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    frame["rsi14"] = 100.0 - 100.0 / (1.0 + relative_strength)
    frame.loc[average_loss.eq(0.0) & average_gain.gt(0.0), "rsi14"] = 100.0

    lookback = int(cfg["breakout_lookback"])
    # Shifted values ensure today's signal only sees levels known before today.
    frame["prior_high"] = frame["high"].shift(1).rolling(lookback).max()
    frame["prior_volume_avg"] = frame["volume"].shift(1).rolling(lookback).mean()

    frame["regime"] = (close > frame["sma200"]) & (frame["sma20"] > frame["sma200"])
    frame["confirmed_regime"] = frame["sma50"] > frame["sma200"]
    frame["breakout_signal"] = (
        frame["regime"]
        & (close > frame["prior_high"])
        & (frame["volume"] >= frame["prior_volume_avg"] * float(cfg["breakout_volume_multiple"]))
    )
    bullish_reversal = (close > frame["open"]) & (close > previous_close)
    frame["shallow_pullback_signal"] = (
        frame["regime"]
        & bullish_reversal
        & (frame["low"] <= frame["sma10"] * 1.005)
        & (close > frame["sma10"])
        & frame["rsi14"].between(float(cfg["shallow_rsi_min"]), float(cfg["shallow_rsi_max"]))
    )
    frame["deep_pullback_signal"] = (
        frame["regime"]
        & bullish_reversal
        & (frame["low"] <= frame["sma20"] * 1.01)
        & (close > frame["sma20"])
        & frame["rsi14"].between(float(cfg["deep_rsi_min"]), float(cfg["deep_rsi_max"]))
    )
    return frame


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "entry_date", "exit_date", "setup", "entry_price", "exit_price",
            "btc_quantity", "net_pnl", "return_on_cost", "exit_reason",
        ]
    )


def calculate_metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, float]:
    if daily.empty:
        return {}
    equity = daily["equity"].astype(float)
    daily_return = equity.pct_change().fillna(0.0)
    drawdown = equity / equity.cummax() - 1.0
    elapsed_days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = elapsed_days / 365.25
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    buy_hold = daily["close"].astype(float) / float(daily["close"].iloc[0])
    buy_hold_drawdown = buy_hold / buy_hold.cummax() - 1.0
    buy_hold_total_return = float(buy_hold.iloc[-1] - 1.0)
    volatility = daily_return.std(ddof=1) * np.sqrt(365.0)
    sharpe = np.nan if not np.isfinite(volatility) or volatility == 0 else daily_return.mean() * 365.0 / volatility
    completed = trades[trades["exit_date"].notna()] if not trades.empty else trades
    wins = completed["net_pnl"] > 0 if not completed.empty else pd.Series(dtype=bool)
    return {
        "initial_equity": float(equity.iloc[0]),
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(volatility),
        "sharpe_zero_rf": float(sharpe) if np.isfinite(sharpe) else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "buy_hold_total_return": buy_hold_total_return,
        "buy_hold_cagr": float((1.0 + buy_hold_total_return) ** (1.0 / years) - 1.0),
        "buy_hold_max_drawdown": float(buy_hold_drawdown.min()),
        "completed_trades": float(len(completed)),
        "win_rate": float(wins.mean()) if len(wins) else float("nan"),
        "exposure_days": float((daily["btc_quantity"] > 0).mean()),
    }


def backtest(data: pd.DataFrame, config: Mapping[str, Any] | None = None) -> BacktestResult:
    """Backtest the long-only strategy using next-open signal execution.

    Signals are computed from a completed daily candle. Entries and close-based
    exits execute at the following candle's open. An intraday protective stop
    uses that day's OHLC; if a gap opens below the stop, the open is used.
    When both a stop and profit target are touched in one candle, the stop is
    processed first, which is deliberately conservative for daily data.
    """
    cfg = {**DEFAULT_CONFIG, **(dict(config) if config else {})}
    frame = add_indicators(data, cfg)
    if len(frame) < 202:
        raise ValueError("At least 202 daily candles are required")

    fee = float(cfg["fee_rate"])
    slip = float(cfg["slippage_rate"])
    cash = float(cfg["initial_capital"])
    quantity = 0.0
    entry_quantity = 0.0
    entry_price = np.nan
    entry_cost = 0.0
    entry_date: pd.Timestamp | None = None
    setup = ""
    stop_price = np.nan
    initial_risk = np.nan
    target_price = np.nan
    highest_close = -np.inf
    partial_taken = False
    pending_entry: Dict[str, Any] | None = None
    pending_exit_reason: str | None = None
    rows: list[Dict[str, Any]] = []
    trades: list[Dict[str, Any]] = []

    def sell(qty: float, raw_price: float) -> tuple[float, float]:
        nonlocal cash, quantity
        fill = max(raw_price * (1.0 - slip), 0.0)
        proceeds = qty * fill * (1.0 - fee)
        cash += proceeds
        quantity = max(quantity - qty, 0.0)
        return fill, proceeds

    for i, (date, row) in enumerate(frame.iterrows()):
        exit_reason_today = ""

        if quantity > 0 and pending_exit_reason:
            remaining = quantity
            fill, proceeds = sell(remaining, float(row["open"]))
            trade = trades[-1]
            trade["sale_proceeds"] += proceeds
            trade["sold_quantity"] = entry_quantity
            trade["exit_date"] = date
            trade["exit_price"] = fill
            trade["net_pnl"] = trade["sale_proceeds"] - entry_cost
            trade["return_on_cost"] = trade["net_pnl"] / entry_cost
            trade["exit_reason"] = pending_exit_reason
            exit_reason_today = pending_exit_reason
            pending_exit_reason = None

        if quantity == 0 and pending_entry is not None:
            raw_open = float(row["open"])
            if raw_open <= pending_entry["signal_close"] * 1.02:
                fill = raw_open * (1.0 + slip)
                atr = float(pending_entry["atr"])
                stop_multiple = float(
                    cfg["breakout_stop_atr"]
                    if pending_entry["setup"] == "breakout"
                    else cfg["pullback_stop_atr"]
                )
                stop_distance = atr * stop_multiple
                exposure_cap = float(
                    cfg["confirmed_exposure_cap"]
                    if pending_entry["confirmed"]
                    else cfg["transition_exposure_cap"]
                )
                equity_before = cash
                risk_fraction = float(cfg["risk_per_trade"])
                risk_sized_value = equity_before * risk_fraction * fill / stop_distance
                purchase_value = min(risk_sized_value, equity_before * exposure_cap)
                quantity = purchase_value / (fill * (1.0 + fee))
                entry_quantity = quantity
                entry_price = fill
                entry_cost = quantity * fill * (1.0 + fee)
                cash -= entry_cost
                entry_date = date
                setup = str(pending_entry["setup"])
                stop_price = fill - stop_distance
                initial_risk = stop_distance
                target_price = fill + float(cfg["partial_take_profit_r"]) * stop_distance
                highest_close = float(row["close"])
                partial_taken = False
                trades.append({
                    "entry_date": date,
                    "exit_date": pd.NaT,
                    "setup": setup,
                    "entry_price": fill,
                    "exit_price": np.nan,
                    "btc_quantity": entry_quantity,
                    "entry_cost": entry_cost,
                    "sale_proceeds": 0.0,
                    "sold_quantity": 0.0,
                    "net_pnl": np.nan,
                    "return_on_cost": np.nan,
                    "exit_reason": "",
                })
            pending_entry = None

        if quantity > 0:
            # Conservative ordering for an OHLC-only backtest: stop before target.
            if float(row["low"]) <= stop_price:
                raw_fill = min(float(row["open"]), stop_price)
                remaining = quantity
                fill, proceeds = sell(remaining, raw_fill)
                trade = trades[-1]
                trade["sale_proceeds"] += proceeds
                trade["sold_quantity"] += remaining
                trade["exit_date"] = date
                trade["exit_price"] = fill
                trade["net_pnl"] = trade["sale_proceeds"] - entry_cost
                trade["return_on_cost"] = trade["net_pnl"] / entry_cost
                trade["exit_reason"] = "protective_stop"
                exit_reason_today = "protective_stop"
            else:
                if not partial_taken and float(row["high"]) >= target_price:
                    qty_to_sell = quantity * float(cfg["partial_take_profit_fraction"])
                    fill, proceeds = sell(qty_to_sell, target_price)
                    trade = trades[-1]
                    trade["sale_proceeds"] += proceeds
                    trade["sold_quantity"] += qty_to_sell
                    partial_taken = True
                highest_close = max(highest_close, float(row["close"]))
                if pd.notna(row["atr14"]):
                    stop_price = max(
                        stop_price,
                        highest_close - float(cfg["trailing_stop_atr"]) * float(row["atr14"]),
                    )
                below_sma20_twice = (
                    i >= 1
                    and float(row["close"]) < float(row["sma20"])
                    and float(frame.iloc[i - 1]["close"]) < float(frame.iloc[i - 1]["sma20"])
                )
                if below_sma20_twice:
                    pending_exit_reason = "two_closes_below_sma20"

        if quantity == 0 and pending_exit_reason is None and i < len(frame) - 1:
            if bool(row["breakout_signal"]):
                pending_entry = {
                    "setup": "breakout", "signal_close": float(row["close"]),
                    "atr": float(row["atr14"]), "confirmed": bool(row["confirmed_regime"]),
                }
            elif bool(row["deep_pullback_signal"]):
                pending_entry = {
                    "setup": "deep_pullback", "signal_close": float(row["close"]),
                    "atr": float(row["atr14"]), "confirmed": bool(row["confirmed_regime"]),
                }
            elif bool(row["shallow_pullback_signal"]):
                pending_entry = {
                    "setup": "shallow_pullback", "signal_close": float(row["close"]),
                    "atr": float(row["atr14"]), "confirmed": bool(row["confirmed_regime"]),
                }

        mark_price = float(row["close"])
        equity = cash + quantity * mark_price
        rows.append({
            "date": date,
            "equity": equity,
            "cash": cash,
            "btc_quantity": quantity,
            "close": mark_price,
            "stop_price": stop_price if quantity > 0 else np.nan,
            "breakout_signal": bool(row["breakout_signal"]),
            "shallow_pullback_signal": bool(row["shallow_pullback_signal"]),
            "deep_pullback_signal": bool(row["deep_pullback_signal"]),
            "exit_reason": exit_reason_today,
        })

    daily = pd.DataFrame(rows).set_index("date")
    trades_frame = pd.DataFrame(trades) if trades else _empty_trades()
    if not trades_frame.empty:
        trades_frame = trades_frame.drop(columns=["entry_cost", "sale_proceeds", "sold_quantity"], errors="ignore")
    return BacktestResult(daily=daily, trades=trades_frame, metrics=calculate_metrics(daily, trades_frame))
