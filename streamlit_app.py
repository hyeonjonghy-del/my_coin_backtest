from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .adaptive_120_strategy import DEFAULT_CONFIG, backtest
    from .run import fetch_upbit_daily
except ImportError:  # Direct `streamlit run` execution from this directory.
    from adaptive_120_strategy import DEFAULT_CONFIG, backtest
    from run import fetch_upbit_daily


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_START = date(2018, 1, 1)
DEFAULT_EVALUATION_START = date(2018, 5, 1)


st.set_page_config(
    page_title="Upbit BTC Adaptive 120 Backtest",
    page_icon="₿",
    layout="wide",
)


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(start: str, end: str) -> pd.DataFrame:
    return fetch_upbit_daily(start, end, include_incomplete=False)


def metric_percent(label: str, value: float, comparison: float | None = None) -> None:
    delta = None if comparison is None else f"{(value - comparison) * 100:+.2f}%p vs 보유"
    st.metric(label, f"{value * 100:,.2f}%", delta)


st.title("업비트 BTC/KRW Adaptive 120 백테스트")
st.caption(
    "확정 일봉의 120일 추세와 20일 실현변동성으로 익스포저를 조절합니다. "
    "반감기는 참고 정보이며 매매 신호에는 사용하지 않습니다."
)

with st.sidebar:
    st.header("데이터")
    source = st.radio("데이터 소스", ["업비트 최신 데이터", "CSV 업로드"])
    download_start = st.date_input("데이터 시작일", DEFAULT_DATA_START)
    evaluation_start = st.date_input("평가 시작일", DEFAULT_EVALUATION_START)
    evaluation_end = st.date_input("평가 종료일", datetime.now(timezone.utc).date())

    uploaded = None
    if source == "CSV 업로드":
        uploaded = st.file_uploader(
            "OHLCV CSV",
            type=["csv"],
            help="date_utc, open, high, low, close, volume 열이 필요합니다.",
        )

    st.header("전략 설정")
    ma_days = st.slider("이동평균 기간", 60, 240, 120, 5)
    buffer_pct = st.slider("진입·청산 완충폭", 0.0, 5.0, 2.0, 0.25)
    vol_days = st.slider("실현변동성 기간", 10, 60, 20)
    target_vol_pct = st.slider("목표 연 변동성", 20, 120, 80, 5)
    max_exposure = st.slider("최대 익스포저", 0.50, 1.50, 1.25, 0.05)
    min_exposure = st.slider("최소 익스포저", 0.0, min(0.75, max_exposure), 0.25, 0.05)

    st.header("비용 설정")
    turnover_cost_pct = st.number_input(
        "비중 변경 비용(%)", min_value=0.0, max_value=2.0, value=0.10, step=0.01
    )
    financing_pct = st.number_input(
        "100% 초과분 연 금융비용(%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5
    )
    run_clicked = st.button("백테스트 실행", type="primary", use_container_width=True)

if max_exposure > 1.0:
    st.warning(
        "최대 익스포저가 100%를 초과합니다. 업비트 현물만으로는 구현할 수 없으며, "
        "파생상품·차입에는 청산, 펀딩비, 추적오차 위험이 있습니다."
    )

if not run_clicked:
    st.info("왼쪽에서 조건을 설정한 뒤 ‘백테스트 실행’을 누르세요.")
    st.stop()

if evaluation_start < download_start:
    st.error("평가 시작일은 데이터 시작일보다 빠를 수 없습니다.")
    st.stop()
if evaluation_end <= evaluation_start:
    st.error("평가 종료일은 평가 시작일보다 늦어야 합니다.")
    st.stop()

try:
    with st.spinner("일봉 데이터를 준비하고 백테스트하는 중입니다..."):
        if source == "업비트 최신 데이터":
            prices = download_prices(download_start.isoformat(), evaluation_end.isoformat())
        else:
            if uploaded is None:
                st.error("CSV 파일을 먼저 업로드하세요.")
                st.stop()
            raw = pd.read_csv(uploaded)
            date_column = "date_utc" if "date_utc" in raw.columns else raw.columns[0]
            raw[date_column] = pd.to_datetime(raw[date_column])
            prices = raw.set_index(date_column).sort_index()

        config = {
            **DEFAULT_CONFIG,
            "moving_average_days": ma_days,
            "entry_exit_buffer": buffer_pct / 100.0,
            "realized_volatility_days": vol_days,
            "target_annual_volatility": target_vol_pct / 100.0,
            "minimum_exposure": min_exposure,
            "maximum_exposure": max_exposure,
            "transaction_cost_per_turnover": turnover_cost_pct / 100.0,
            "annual_financing_rate": financing_pct / 100.0,
        }
        result = backtest(
            prices,
            config,
            evaluation_start=evaluation_start.isoformat(),
            evaluation_end=evaluation_end.isoformat(),
        )
except Exception as exc:
    st.exception(exc)
    st.stop()

m = result.metrics
st.subheader("성과 비교")
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_percent("전략 CAGR", m["strategy_cagr"], m["buy_hold_cagr"])
with c2:
    metric_percent("전략 MDD", m["strategy_max_drawdown"], m["buy_hold_max_drawdown"])
with c3:
    st.metric("전략 Sharpe", f"{m['strategy_sharpe_zero_rf']:.2f}")
with c4:
    st.metric("평균 익스포저", f"{m['strategy_average_exposure'] * 100:.1f}%")

comparison = pd.DataFrame(
    {
        "Adaptive 120": {
            "누적수익률": m["strategy_total_return"],
            "CAGR": m["strategy_cagr"],
            "MDD": m["strategy_max_drawdown"],
            "연 변동성": m["strategy_annualized_volatility"],
            "Sharpe": m["strategy_sharpe_zero_rf"],
        },
        "단순보유": {
            "누적수익률": m["buy_hold_total_return"],
            "CAGR": m["buy_hold_cagr"],
            "MDD": m["buy_hold_max_drawdown"],
            "연 변동성": m["buy_hold_annualized_volatility"],
            "Sharpe": m["buy_hold_sharpe_zero_rf"],
        },
    }
)
display = comparison.copy()
for row in ["누적수익률", "CAGR", "MDD", "연 변동성"]:
    display.loc[row] = comparison.loc[row].map(lambda value: f"{value * 100:,.2f}%")
display.loc["Sharpe"] = comparison.loc["Sharpe"].map(lambda value: f"{value:.2f}")
st.dataframe(display, use_container_width=True)

st.subheader("자산곡선")
equity = result.daily[["strategy_equity", "buy_hold_equity"]].rename(
    columns={"strategy_equity": "Adaptive 120", "buy_hold_equity": "단순보유"}
)
st.line_chart(equity)

st.subheader("가격과 추세 밴드")
st.line_chart(
    result.daily[["close", "sma", "upper_band", "lower_band"]].rename(
        columns={"close": "BTC/KRW", "sma": "SMA", "upper_band": "상단", "lower_band": "하단"}
    )
)

st.subheader("투자비중")
st.area_chart(result.daily[["position"]].rename(columns={"position": "익스포저"}))

latest = result.daily.iloc[-1]
st.subheader("최신 확정 일봉 신호")
l1, l2, l3 = st.columns(3)
l1.metric("확정 종가", f"₩{latest['close']:,.0f}")
l2.metric("120일선", f"₩{latest['sma']:,.0f}")
l3.metric("다음 거래일 목표 익스포저", f"{latest['desired_exposure'] * 100:.1f}%")
st.write(f"신호 상태: **{latest['signal']}**")

st.download_button(
    "일별 결과 CSV 다운로드",
    data=result.daily.to_csv(index=True).encode("utf-8-sig"),
    file_name="upbit_btc_adaptive_120_backtest.csv",
    mime="text/csv",
)
st.download_button(
    "성과지표 JSON 다운로드",
    data=json.dumps(m, ensure_ascii=False, indent=2, allow_nan=True).encode("utf-8"),
    file_name="upbit_btc_adaptive_120_metrics.json",
    mime="application/json",
)

st.caption(
    "연구·교육용 백테스트입니다. 과거 성과는 미래 성과를 보장하지 않으며, "
    "파생상품의 실제 펀딩비·청산·체결오차는 단순화되어 있습니다."
)

