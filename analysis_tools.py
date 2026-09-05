"""Robustness analysis helpers for the Adaptive moving-average strategy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

try:
    from .adaptive_120_strategy import backtest
except ImportError:
    from adaptive_120_strategy import backtest


MA_CANDIDATES = (100, 110, 120, 130, 140)


def compare_moving_averages(
    prices: pd.DataFrame,
    base_config: Mapping[str, Any],
    evaluation_start: str,
    evaluation_end: str,
    candidates: Iterable[int] = MA_CANDIDATES,
) -> pd.DataFrame:
    """Backtest several moving-average lengths under identical assumptions."""
    rows: list[dict[str, float | int]] = []
    for days in candidates:
        config = {**dict(base_config), "moving_average_days": int(days)}
        metrics = backtest(
            prices,
            config,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        ).metrics
        rows.append(
            {
                "이동평균": int(days),
                "CAGR": metrics["strategy_cagr"],
                "MDD": metrics["strategy_max_drawdown"],
                "Sharpe": metrics["strategy_sharpe_zero_rf"],
                "누적수익률": metrics["strategy_total_return"],
                "평균 익스포저": metrics["strategy_average_exposure"],
            }
        )
    return pd.DataFrame(rows).set_index("이동평균")


def chronological_splits(
    index: pd.DatetimeIndex,
    evaluation_start: str,
    evaluation_end: str,
    parts: int = 3,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Return equal-observation, non-overlapping chronological validation windows."""
    selected = index[(index >= pd.Timestamp(evaluation_start)) & (index <= pd.Timestamp(evaluation_end))]
    if len(selected) < parts * 30:
        raise ValueError("기간분할 검증에는 구간당 최소 30개의 일봉이 필요합니다.")

    groups = np.array_split(selected, parts)
    return [
        (f"구간 {number}", pd.Timestamp(group[0]), pd.Timestamp(group[-1]))
        for number, group in enumerate(groups, start=1)
        if len(group)
    ]


def period_split_validation(
    prices: pd.DataFrame,
    base_config: Mapping[str, Any],
    evaluation_start: str,
    evaluation_end: str,
    candidates: Iterable[int] = MA_CANDIDATES,
    parts: int = 3,
) -> pd.DataFrame:
    """Evaluate every moving-average candidate in independent time windows."""
    rows: list[dict[str, Any]] = []
    windows = chronological_splits(
        pd.DatetimeIndex(prices.index), evaluation_start, evaluation_end, parts=parts
    )
    for period, start, end in windows:
        for days in candidates:
            config = {**dict(base_config), "moving_average_days": int(days)}
            metrics = backtest(
                prices,
                config,
                evaluation_start=start.strftime("%Y-%m-%d"),
                evaluation_end=end.strftime("%Y-%m-%d"),
            ).metrics
            rows.append(
                {
                    "기간": period,
                    "시작일": start.strftime("%Y-%m-%d"),
                    "종료일": end.strftime("%Y-%m-%d"),
                    "이동평균": int(days),
                    "CAGR": metrics["strategy_cagr"],
                    "MDD": metrics["strategy_max_drawdown"],
                    "Sharpe": metrics["strategy_sharpe_zero_rf"],
                    "단순보유 CAGR": metrics["buy_hold_cagr"],
                    "단순보유 MDD": metrics["buy_hold_max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def downsample_for_chart(frame: pd.DataFrame, max_points: int = 900) -> pd.DataFrame:
    """Reduce chart payload while preserving endpoints and each series' extrema."""
    if len(frame) <= max_points:
        return frame

    positions = np.linspace(0, len(frame) - 1, max_points, dtype=int)
    keep = set(positions.tolist())
    keep.update({0, len(frame) - 1})
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column]
        if values.notna().any():
            keep.add(frame.index.get_loc(values.idxmin()))
            keep.add(frame.index.get_loc(values.idxmax()))
    return frame.iloc[sorted(keep)]
