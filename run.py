from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

try:
    from .strategy import add_indicators, backtest
except ImportError:
    from strategy import add_indicators, backtest


API_URL = "https://api.upbit.com/v1/candles/days"


def _request_candles(to: str | None, count: int = 200) -> list[dict[str, Any]]:
    query: dict[str, Any] = {"market": "KRW-BTC", "count": min(count, 200)}
    if to:
        query["to"] = to
    request = Request(
        f"{API_URL}?{urlencode(query)}",
        headers={"Accept": "application/json", "User-Agent": "upbit-btc-daily-backtest/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_upbit_daily(start: str, end: str, include_incomplete: bool = False) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    cursor = end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    records: list[dict[str, Any]] = []

    while True:
        batch = _request_candles(cursor)
        if not batch:
            break
        records.extend(batch)
        oldest = pd.Timestamp(batch[-1]["candle_date_time_utc"], tz="UTC")
        if oldest <= start_ts:
            break
        cursor = (oldest - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        time.sleep(0.12)

    if not records:
        raise RuntimeError("Upbit returned no candle data")
    raw = pd.DataFrame(records)
    index = pd.to_datetime(raw["candle_date_time_utc"], utc=True).dt.tz_localize(None)
    frame = pd.DataFrame(
        {
            "open": raw["opening_price"].to_numpy(),
            "high": raw["high_price"].to_numpy(),
            "low": raw["low_price"].to_numpy(),
            "close": raw["trade_price"].to_numpy(),
            "volume": raw["candle_acc_trade_volume"].to_numpy(),
        },
        index=index,
    ).sort_index()
    frame.index.name = "date_utc"
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.loc[(frame.index >= start_ts.tz_localize(None)) & (frame.index < end_ts.tz_localize(None))]

    # Upbit's current UTC-date candle remains open until the next 00:00 UTC
    # (09:00 KST). Exclude it by default to avoid unstable signals.
    if not include_incomplete:
        current_utc_date = datetime.now(timezone.utc).date()
        frame = frame.loc[frame.index.date < current_utc_date]
    return frame


def load_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = "date_utc" if "date_utc" in frame.columns else frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column])
    return frame.set_index(date_column).sort_index()


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Backtest the Upbit BTC/KRW daily strategy")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--config", default=str(here / "config.json"))
    parser.add_argument("--csv", help="Optional local OHLCV CSV; skips the Upbit download")
    parser.add_argument("--output", default=str(here / "results"))
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    prices = load_csv(args.csv) if args.csv else fetch_upbit_daily(
        args.start, args.end, include_incomplete=args.include_incomplete
    )
    result = backtest(prices, config)
    indicators = add_indicators(prices, config)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output / "upbit_btc_krw_daily.csv", encoding="utf-8-sig")
    indicators.to_csv(output / "signals_and_indicators.csv", encoding="utf-8-sig")
    result.daily.to_csv(output / "equity_curve.csv", encoding="utf-8-sig")
    result.trades.to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")
    (output / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=True))
    print(f"Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()

