import pandas as pd

from telegram_alert import build_alert


def sample(position: float, target: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": [position],
            "desired_exposure": [target],
            "close": [100_000_000.0],
            "sma": [95_000_000.0],
            "realized_volatility": [0.5],
        },
        index=pd.to_datetime(["2026-09-08"]),
    )


def test_skips_unchanged_exposure():
    changed, message = build_alert(sample(0.75, 0.75))
    assert not changed
    assert message == ""


def test_formats_changed_exposure():
    changed, message = build_alert(sample(0.25, 0.75))
    assert changed
    assert "추가 매수" in message
    assert "75.00%" in message
    assert "+50.00%p" in message


def test_alerts_on_small_exposure_change():
    changed, message = build_alert(sample(0.7500, 0.7501))
    assert changed
    assert "+0.01%p" in message
