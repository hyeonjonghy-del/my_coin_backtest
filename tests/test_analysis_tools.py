from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adaptive_120_strategy import DEFAULT_CONFIG
from analysis_tools import (
    chronological_splits,
    compare_moving_averages,
    downsample_for_chart,
    period_split_validation,
)


class AnalysisToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        index = pd.date_range("2018-01-01", periods=900, freq="D")
        close = pd.Series(10_000 + np.linspace(0, 20_000, len(index)), index=index)
        cls.prices = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1.0,
            }
        )

    def test_comparison_contains_all_candidates(self) -> None:
        result = compare_moving_averages(
            self.prices, DEFAULT_CONFIG, "2018-06-01", "2020-06-01"
        )
        self.assertEqual(result.index.tolist(), [100, 110, 120, 130, 140])
        self.assertIn("CAGR", result.columns)

    def test_period_split_has_three_windows_per_candidate(self) -> None:
        result = period_split_validation(
            self.prices, DEFAULT_CONFIG, "2018-06-01", "2020-06-01"
        )
        self.assertEqual(len(result), 15)
        self.assertEqual(result["기간"].nunique(), 3)

    def test_splits_do_not_overlap(self) -> None:
        windows = chronological_splits(
            self.prices.index, "2018-06-01", "2020-06-01"
        )
        self.assertLess(windows[0][2], windows[1][1])
        self.assertLess(windows[1][2], windows[2][1])

    def test_downsampling_preserves_extrema_and_last_row(self) -> None:
        frame = pd.DataFrame({"value": np.arange(2_000, dtype=float)})
        frame.loc[777, "value"] = -100.0
        sampled = downsample_for_chart(frame, max_points=100)
        self.assertIn(777, sampled.index)
        self.assertIn(1999, sampled.index)
        self.assertLessEqual(len(sampled), 104)


if __name__ == "__main__":
    unittest.main()
