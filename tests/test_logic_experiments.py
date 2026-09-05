from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adaptive_120_strategy import DEFAULT_CONFIG, prepare_signals
from logic_experiments import compare_logic_experiments, split_logic_experiments


class LogicExperimentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        index = pd.date_range("2017-01-01", periods=1_500, freq="D")
        trend = np.linspace(10_000, 40_000, len(index))
        cycle = 4_000 * np.sin(np.linspace(0, 30, len(index)))
        close = pd.Series(trend + cycle, index=index)
        cls.prices = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1.0,
            }
        )

    def test_only_four_controlled_variants_are_compared(self) -> None:
        result = compare_logic_experiments(
            self.prices, DEFAULT_CONFIG, "2018-01-01", "2020-12-31"
        )
        self.assertEqual(len(result), 4)
        self.assertIn("두 로직 결합", result.index)

    def test_split_experiment_has_three_windows(self) -> None:
        result = split_logic_experiments(
            self.prices, DEFAULT_CONFIG, "2018-01-01", "2020-12-31"
        )
        self.assertEqual(result["기간"].nunique(), 3)
        self.assertEqual(len(result), 12)

    def test_stable_sizing_never_exceeds_spot_limit(self) -> None:
        signals = prepare_signals(
            self.prices,
            {**DEFAULT_CONFIG, "enable_stable_volatility_sizing": True},
        )
        self.assertLessEqual(float(signals["desired_exposure"].max()), 1.0)
        self.assertGreaterEqual(float(signals["desired_exposure"].min()), 0.0)

    def test_enhancements_do_not_look_into_future(self) -> None:
        cfg = {
            **DEFAULT_CONFIG,
            "enable_recovery_reentry": True,
            "enable_stable_volatility_sizing": True,
        }
        cutoff = 1_100
        original = prepare_signals(self.prices, cfg).iloc[:cutoff]
        changed = self.prices.copy()
        changed.iloc[cutoff:, changed.columns.get_loc("close")] *= 3.0
        changed.iloc[cutoff:, changed.columns.get_loc("high")] *= 3.0
        changed.iloc[cutoff:, changed.columns.get_loc("low")] *= 3.0
        altered = prepare_signals(changed, cfg).iloc[:cutoff]
        pd.testing.assert_series_equal(
            original["desired_exposure"], altered["desired_exposure"]
        )


if __name__ == "__main__":
    unittest.main()
