"""Send a Telegram alert when the Adaptive 120 model exposure changes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from adaptive_120_strategy import DEFAULT_CONFIG, backtest
from run import fetch_upbit_daily


CHANGE_TOLERANCE = 0.005


def build_alert(daily: pd.DataFrame) -> tuple[bool, str]:
    latest = daily.iloc[-1]
    current = float(latest["position"])
    target = float(latest["desired_exposure"])
    change = target - current
    if abs(change) <= CHANGE_TOLERANCE:
        return False, ""

    if target <= CHANGE_TOLERANCE:
        action = "전량 매도"
    elif current <= CHANGE_TOLERANCE:
        action = "매수"
    elif change > 0:
        action = "추가 매수"
    else:
        action = "일부 매도"

    signal_date = pd.Timestamp(latest.name)
    execution_date = signal_date + pd.DateOffset(days=1)
    message = (
        "🔔 <b>BTC Adaptive 120 비중 변경</b>\n\n"
        f"판단: <b>{action}</b>\n"
        f"현재 모델 비중: {current * 100:.1f}%\n"
        f"새 목표 비중: <b>{target * 100:.1f}%</b>\n"
        f"조정폭: {change * 100:+.1f}%p\n\n"
        f"신호 확정일: {signal_date:%Y-%m-%d} UTC\n"
        f"적용 기준: {execution_date:%Y-%m-%d} 09:00 KST 이후\n"
        f"확정 종가: ₩{latest['close']:,.0f}\n"
        f"120일 이동평균: ₩{latest['sma']:,.0f}\n"
        f"20일 실현변동성: {latest['realized_volatility'] * 100:.1f}%\n\n"
        "주문 참고금액 = 현재 총 평가금액 × 조정폭"
    )
    return True, message


def send_telegram(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured as GitHub Actions secrets"
        )

    payload = urlencode(
        {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    ).encode("utf-8")
    request = Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError("Telegram API request failed") from exc
    if not result.get("ok"):
        raise RuntimeError("Telegram API rejected the alert")


def main() -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    prices = fetch_upbit_daily("2017-09-25", today, include_incomplete=False)
    result = backtest(
        prices,
        DEFAULT_CONFIG,
        evaluation_start="2018-05-01",
        evaluation_end=today,
    )
    changed, message = build_alert(result.daily)
    if not changed:
        print("No model exposure change; Telegram alert skipped.")
        return
    send_telegram(message)
    print("Telegram exposure-change alert sent.")


if __name__ == "__main__":
    main()
