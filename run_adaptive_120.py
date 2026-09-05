from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .adaptive_120_strategy import backtest
    from .run import fetch_upbit_daily, load_csv
except ImportError:
    from adaptive_120_strategy import backtest
    from run import fetch_upbit_daily, load_csv


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Backtest the adaptive Upbit BTC 120-day strategy")
    parser.add_argument("--download-start", default="2018-01-01")
    parser.add_argument("--evaluation-start", default="2018-05-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--config", default=str(here / "adaptive_120_config.json"))
    parser.add_argument("--csv", help="Optional local OHLCV CSV; skips the Upbit download")
    parser.add_argument("--output", default=str(here / "adaptive_120_results"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    prices = load_csv(args.csv) if args.csv else fetch_upbit_daily(args.download_start, args.end)
    result = backtest(
        prices,
        config,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.end,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(output / "daily_signals_and_equity.csv", encoding="utf-8-sig")
    (output / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=True))
    print(f"Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
