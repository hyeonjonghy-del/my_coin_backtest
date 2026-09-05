"""Controlled experiments for the two requested Adaptive 120 enhancements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

try:
    from .adaptive_120_strategy import backtest
    from .analysis_tools import chronological_splits
except ImportError:
    from adaptive_120_strategy import backtest
    from analysis_tools import chronological_splits


EXPERIMENTS = {
    "기존 Adaptive 120": {},
    "회복 재진입만": {"enable_recovery_reentry": True},
    "안정형 변동성 비중만": {"enable_stable_volatility_sizing": True},
    "두 로직 결합": {
        "enable_recovery_reentry": True,
        "enable_stable_volatility_sizing": True,
    },
}


def compare_logic_experiments(
    prices: pd.DataFrame,
    base_config: Mapping[str, Any],
    evaluation_start: str,
    evaluation_end: str,
) -> pd.DataFrame:
    """Compare baseline, each enhancement alone, and both together."""
    rows = []
    for name, overrides in EXPERIMENTS.items():
        cfg = {
            **dict(base_config),
            "moving_average_days": 120,
            "enable_recovery_reentry": False,
            "enable_stable_volatility_sizing": False,
            **overrides,
        }
        result = backtest(prices, cfg, evaluation_start, evaluation_end)
        m = result.metrics
        rows.append(
            {
                "실험": name,
                "CAGR": m["strategy_cagr"],
                "MDD": m["strategy_max_drawdown"],
                "Sharpe": m["strategy_sharpe_zero_rf"],
                "누적수익률": m["strategy_total_return"],
                "평균 익스포저": m["strategy_average_exposure"],
                "총 회전율": float(result.daily["turnover"].sum()),
                "재진입 횟수": int(result.daily["recovery_entry"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("실험")


def split_logic_experiments(
    prices: pd.DataFrame,
    base_config: Mapping[str, Any],
    evaluation_start: str,
    evaluation_end: str,
) -> pd.DataFrame:
    """Repeat the four controlled experiments in three independent windows."""
    rows = []
    for period, start, end in chronological_splits(
        pd.DatetimeIndex(prices.index), evaluation_start, evaluation_end
    ):
        comparison = compare_logic_experiments(
            prices, base_config, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        ).reset_index()
        comparison.insert(0, "기간", period)
        comparison.insert(1, "시작일", start.strftime("%Y-%m-%d"))
        comparison.insert(2, "종료일", end.strftime("%Y-%m-%d"))
        rows.append(comparison)
    return pd.concat(rows, ignore_index=True)
