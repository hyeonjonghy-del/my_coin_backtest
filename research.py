from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


HALVINGS = pd.to_datetime(["2016-07-09", "2020-05-11", "2024-04-20"])


def load_prices(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date_utc"]).set_index("date_utc").sort_index()
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["open", "close"])


def hysteresis_position(close: pd.Series, ma_days: int, buffer: float, confirm_days: int) -> pd.Series:
    average = close.rolling(ma_days).mean()
    above = close > average * (1.0 + buffer)
    below = close < average * (1.0 - buffer)
    enter = above.rolling(confirm_days).sum().eq(confirm_days)
    exit_ = below.rolling(confirm_days).sum().eq(confirm_days)
    position = pd.Series(np.nan, index=close.index, dtype=float)
    position.loc[enter] = 1.0
    position.loc[exit_] = 0.0
    return position.ffill().fillna(0.0)


def make_position(
    prices: pd.DataFrame,
    ma_days: int,
    buffer: float,
    confirm_days: int,
    mode: str,
    max_exposure: float,
    target_volatility: float,
    use_halving_overlay: bool,
) -> pd.Series:
    close = prices["close"]
    base = hysteresis_position(close, ma_days, buffer, confirm_days)
    if mode == "binary":
        desired = base.copy()
    elif mode == "staged":
        ma = close.rolling(ma_days).mean()
        strong = (
            (close.rolling(20).mean() > close.rolling(60).mean())
            & (close.rolling(60).mean() > ma)
            & (ma > ma.shift(20))
        )
        desired = base.where(strong, base * 0.50)
    elif mode == "vol_target":
        daily_vol = close.pct_change().rolling(20).std(ddof=1) * np.sqrt(365.0)
        multiplier = (target_volatility / daily_vol).clip(lower=0.25, upper=max_exposure)
        desired = base * multiplier
    else:
        raise ValueError(f"Unknown mode: {mode}")

    desired = desired.clip(lower=0.0, upper=max_exposure)
    if use_halving_overlay:
        dates = pd.Series(prices.index, index=prices.index)
        latest_halving = pd.Series(pd.NaT, index=prices.index, dtype="datetime64[ns]")
        for halving in HALVINGS:
            latest_halving.loc[prices.index >= halving] = halving
        age = (dates - latest_halving).dt.days
        # A deliberately coarse overlay: it changes only the exposure ceiling.
        # Price trend remains the actual entry/exit signal.
        cap = pd.Series(1.0, index=prices.index)
        cap.loc[age.between(0, 540)] = 1.25
        cap.loc[age.between(541, 900)] = 1.00
        cap.loc[age > 900] = 0.75
        desired = np.minimum(desired, cap).astype(float)
    return desired


def equity_curve(
    prices: pd.DataFrame,
    desired_position: pd.Series,
    cost_rate: float,
    annual_financing_rate: float = 0.10,
) -> pd.DataFrame:
    # A close signal becomes executable at the next open. That exposure earns
    # the following open-to-open return, avoiding a same-close look-ahead bias.
    position = desired_position.shift(1).fillna(0.0)
    forward_open_return = prices["open"].shift(-1) / prices["open"] - 1.0
    turnover = position.diff().abs().fillna(position.abs())
    financing_cost = position.sub(1.0).clip(lower=0.0) * annual_financing_rate / 365.0
    strategy_return = (
        position * forward_open_return.fillna(0.0)
        - turnover * cost_rate
        - financing_cost
    )
    equity = (1.0 + strategy_return).cumprod()
    return pd.DataFrame(
        {
            "position": position,
            "turnover": turnover,
            "financing_cost": financing_cost,
            "return": strategy_return,
            "equity": equity,
        },
        index=prices.index,
    )


def metrics(curve: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sample = curve.loc[start:end].copy()
    if len(sample) < 2:
        return {key: np.nan for key in ["return", "cagr", "mdd", "volatility", "sharpe", "exposure"]}
    wealth = (1.0 + sample["return"]).cumprod()
    years = max((sample.index[-1] - sample.index[0]).days / 365.25, 1 / 365.25)
    total_return = wealth.iloc[-1] - 1.0
    volatility = sample["return"].std(ddof=1) * np.sqrt(365.0)
    return {
        "return": float(total_return),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
        "mdd": float((wealth / wealth.cummax() - 1.0).min()),
        "volatility": float(volatility),
        "sharpe": float(sample["return"].mean() * 365.0 / volatility) if volatility > 0 else np.nan,
        "exposure": float(sample["position"].mean()),
    }


def evaluate(
    prices: pd.DataFrame,
    cost_rate: float,
    annual_financing_rate: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    evaluation_start = "2019-01-01"
    train_end = "2022-12-31"
    test_start = "2023-01-01"
    evaluation_end = prices.index[-1].strftime("%Y-%m-%d")
    specifications = []

    for ma_days, buffer, confirm_days, mode, max_exposure, target_volatility, halving in itertools.product(
        [90, 105, 120, 135, 150, 180, 200],
        [0.0, 0.01, 0.015, 0.02],
        [1, 2, 3],
        ["binary", "staged", "vol_target"],
        [1.0, 1.25],
        [0.60, 0.80],
        [False, True],
    ):
        if mode != "vol_target" and target_volatility != 0.60:
            continue
        if mode in {"binary", "staged"} and max_exposure != 1.0:
            continue
        position = make_position(
            prices, ma_days, buffer, confirm_days, mode,
            max_exposure, target_volatility, halving,
        )
        curve = equity_curve(prices, position, cost_rate, annual_financing_rate)
        whole = metrics(curve, evaluation_start, evaluation_end)
        train = metrics(curve, evaluation_start, train_end)
        test = metrics(curve, test_start, evaluation_end)
        specifications.append(
            {
                "name": f"ma{ma_days}_{mode}_b{buffer}_c{confirm_days}_x{max_exposure}_v{target_volatility}_h{halving}",
                "ma_days": ma_days,
                "buffer": buffer,
                "confirm_days": confirm_days,
                "mode": mode,
                "max_exposure": max_exposure,
                "target_volatility": target_volatility,
                "halving_overlay": halving,
                **{f"all_{k}": v for k, v in whole.items()},
                **{f"train_{k}": v for k, v in train.items()},
                **{f"test_{k}": v for k, v in test.items()},
            }
        )

    hold_position = pd.Series(1.0, index=prices.index)
    hold_curve = equity_curve(prices, hold_position, cost_rate, annual_financing_rate)
    hold = {
        "all": metrics(hold_curve, evaluation_start, evaluation_end),
        "train": metrics(hold_curve, evaluation_start, train_end),
        "test": metrics(hold_curve, test_start, evaluation_end),
    }
    results = pd.DataFrame(specifications)
    results["beats_hold_all"] = (
        (results["all_cagr"] > hold["all"]["cagr"])
        & (results["all_mdd"] > hold["all"]["mdd"])
    )
    results["beats_hold_test"] = (
        (results["test_cagr"] > hold["test"]["cagr"])
        & (results["test_mdd"] > hold["test"]["mdd"])
    )
    results["qualifies"] = results["beats_hold_all"] & results["beats_hold_test"]
    results = results.sort_values(
        ["qualifies", "test_cagr", "all_cagr", "all_mdd"], ascending=[False, False, False, False]
    )
    return results, {"hold": hold, "hold_curve": hold_curve}


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Research robust MA/halving BTC allocation rules")
    parser.add_argument("--csv", default=str(here / "results" / "upbit_btc_krw_daily.csv"))
    parser.add_argument("--cost-rate", type=float, default=0.001)
    parser.add_argument("--annual-financing-rate", type=float, default=0.10)
    parser.add_argument("--output", default=str(here / "research_results"))
    args = parser.parse_args()

    prices = load_prices(args.csv)
    results, benchmarks = evaluate(prices, args.cost_rate, args.annual_financing_rate)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "parameter_comparison.csv", index=False, encoding="utf-8-sig")
    results.head(30).to_csv(output / "top_30.csv", index=False, encoding="utf-8-sig")
    (output / "buy_and_hold.json").write_text(
        json.dumps(benchmarks["hold"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Buy and hold:")
    print(json.dumps(benchmarks["hold"], ensure_ascii=False, indent=2))
    print(f"Qualified variants: {int(results['qualifies'].sum())} / {len(results)}")
    columns = [
        "name", "all_cagr", "all_mdd", "test_cagr", "test_mdd",
        "all_exposure", "qualifies",
    ]
    print(results[columns].head(15).to_string(index=False))
    print(f"Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
