from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .adaptive_120_strategy import DEFAULT_CONFIG, backtest
    from .analysis_tools import (
        MA_CANDIDATES,
        compare_moving_averages,
        downsample_for_chart,
        period_split_validation,
    )
    from .run import fetch_upbit_daily
except ImportError:  # Direct `streamlit run` execution from this directory.
    from adaptive_120_strategy import DEFAULT_CONFIG, backtest
    from analysis_tools import (
        MA_CANDIDATES,
        compare_moving_averages,
        downsample_for_chart,
        period_split_validation,
    )
    from run import fetch_upbit_daily


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_START = date(2018, 1, 1)
DEFAULT_EVALUATION_START = date(2018, 5, 1)


st.set_page_config(
    page_title="Upbit BTC Adaptive 120 Backtest",
    page_icon="₿",
    layout="wide",
)


@st.cache_data(ttl=21600, max_entries=4, show_spinner=False)
def download_prices(start: str, end: str) -> pd.DataFrame:
    return fetch_upbit_daily(start, end, include_incomplete=False)


def metric_percent(label: str, value: float, comparison: float | None = None) -> None:
    delta = None if comparison is None else f"{(value - comparison) * 100:+.2f}%p vs 보유"
    st.metric(label, f"{value * 100:,.2f}%", delta)


st.title("코인 전용 BTC/KRW Adaptive 120 백테스트")
st.caption(
    "확정 일봉의 120일 추세와 20일 실현변동성으로 익스포저를 조절합니다. "
    "기본값은 업비트 현물 운용 기준이며, 반감기는 참고 정보로만 사용합니다."
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
    max_exposure = st.slider("최대 익스포저", 0.50, 1.00, 1.00, 0.05)
    min_exposure = st.slider("최소 익스포저", 0.0, min(0.75, max_exposure), 0.25, 0.05)
    st.caption("현물 전용 설정: 투자비중은 원금의 100%를 초과하지 않습니다.")

    st.header("비용 설정")
    turnover_cost_pct = st.number_input(
        "비중 변경 비용(%)", min_value=0.0, max_value=2.0, value=0.10, step=0.01
    )
    run_clicked = st.button("백테스트 실행", type="primary", width="stretch")

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
            "annual_financing_rate": 0.0,
        }
        result = backtest(
            prices,
            config,
            evaluation_start=evaluation_start.isoformat(),
            evaluation_end=evaluation_end.isoformat(),
        )
        ma_comparison = compare_moving_averages(
            prices,
            config,
            evaluation_start.isoformat(),
            evaluation_end.isoformat(),
        )
        split_results = period_split_validation(
            prices,
            config,
            evaluation_start.isoformat(),
            evaluation_end.isoformat(),
        )
except Exception as exc:
    st.exception(exc)
    st.stop()

m = result.metrics
daily = result.daily

latest = daily.iloc[-1]
current_exposure = float(latest["position"])
target_exposure = float(latest["desired_exposure"])
adjustment = target_exposure - current_exposure
signal_date = pd.Timestamp(latest.name)
execution_date = signal_date + pd.DateOffset(days=1)
tolerance = 0.005

if abs(adjustment) <= tolerance:
    action = "보유 유지" if target_exposure > tolerance else "현금 유지"
elif target_exposure <= tolerance:
    action = "전량 매도"
elif current_exposure <= tolerance:
    action = "매수"
elif adjustment > 0:
    action = "추가 매수"
else:
    action = "일부 매도"

signal_name = {
    "ENTER_OR_HOLD": "상승 추세 진입·유지",
    "EXIT_OR_CASH": "추세 이탈·현금화",
    "HYSTERESIS_HOLD": "완충 구간·기존 상태 유지",
}.get(str(latest["signal"]), str(latest["signal"]))

monthly_returns = (
    (1.0 + daily["strategy_return"].fillna(0.0))
    .groupby([daily.index.year, daily.index.month])
    .prod()
    .sub(1.0)
    .unstack(level=1)
    .reindex(columns=range(1, 13))
)
monthly_returns.columns = [f"{month}월" for month in range(1, 13)]
annual_returns = (
    (1.0 + daily["strategy_return"].fillna(0.0))
    .groupby(daily.index.year)
    .prod()
    .sub(1.0)
)
monthly_returns["연 수익률"] = annual_returns
monthly_returns.index.name = "연도"
monthly_display = monthly_returns.apply(
    lambda column: column.map(lambda value: "-" if pd.isna(value) else f"{value * 100:+.2f}%")
)

performance_tab, comparison_tab, validation_tab, monthly_tab = st.tabs(
    ["성과", "이평선 비교", "기간분할 검증", "월별 수익률"]
)

with performance_tab:
    st.subheader("실전 시그널")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("실행 판단", action)
    s2.metric("현재 모델 비중", f"{current_exposure * 100:.1f}%")
    s3.metric("다음 목표 비중", f"{target_exposure * 100:.1f}%")
    s4.metric("조정폭", f"{adjustment * 100:+.1f}%p")

    execution_message = (
        f"{execution_date:%Y-%m-%d} UTC 00:00(KST 09:00) 일봉 시가부터 "
        f"BTC 비중을 {target_exposure * 100:.1f}%로 맞춥니다."
    )
    if action in {"매수", "추가 매수"}:
        st.success(f"매수 신호: {execution_message}")
    elif action in {"전량 매도", "일부 매도"}:
        st.warning(f"매도 신호: {execution_message}")
    else:
        st.info(f"유지 신호: {execution_message}")

    signal_details = pd.DataFrame(
        {
            "항목": [
                "신호 기준 확정 일봉",
                "전략 상태",
                "확정 종가",
                f"{ma_days}일 이동평균",
                "매수 진입 기준",
                "매도 청산 기준",
                f"{vol_days}일 실현변동성",
            ],
            "값": [
                f"{signal_date:%Y-%m-%d} UTC",
                signal_name,
                f"₩{latest['close']:,.0f}",
                f"₩{latest['sma']:,.0f}",
                f"종가 > ₩{latest['upper_band']:,.0f}",
                f"종가 < ₩{latest['lower_band']:,.0f}",
                f"{latest['realized_volatility'] * 100:.1f}%",
            ],
        }
    )
    st.dataframe(signal_details, hide_index=True, width="stretch")
    st.caption(
        "주문 기준: 현재 총 평가금액 × 조정폭입니다. 조정폭이 +이면 매수, -이면 매도합니다. "
        "신호는 확정 일봉으로 계산하며 백테스트와 동일하게 다음 일봉 시가에 적용합니다."
    )

    st.divider()
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

    strategy_label = f"Adaptive {ma_days}"
    comparison = pd.DataFrame(
        {
            strategy_label: {
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
    display = comparison.astype(object).copy()
    for row in ["누적수익률", "CAGR", "MDD", "연 변동성"]:
        display.loc[row] = comparison.loc[row].map(lambda value: f"{value * 100:,.2f}%")
    display.loc["Sharpe"] = comparison.loc["Sharpe"].map(lambda value: f"{value:.2f}")
    st.dataframe(display, width="stretch")

    st.subheader("자산곡선")
    equity = daily[["strategy_equity", "buy_hold_equity"]].rename(
        columns={"strategy_equity": f"Adaptive {ma_days}", "buy_hold_equity": "단순보유"}
    )
    st.line_chart(downsample_for_chart(equity))

    st.subheader("낙폭(MDD) 차트")
    drawdown = equity.div(equity.cummax()).sub(1.0)
    st.line_chart(downsample_for_chart(drawdown))
    st.caption(
        "각 시점의 이전 최고 자산 대비 하락률입니다. 0%에 가까울수록 고점 부근이며, "
        "가장 낮은 값이 해당 전략의 MDD입니다."
    )

    st.subheader("가격과 추세 밴드")
    st.line_chart(
        downsample_for_chart(daily[["close", "sma", "upper_band", "lower_band"]]).rename(
            columns={"close": "BTC/KRW", "sma": "SMA", "upper_band": "상단", "lower_band": "하단"}
        )
    )

    st.subheader("투자비중")
    st.area_chart(
        downsample_for_chart(daily[["position"]]).rename(columns={"position": "익스포저"})
    )

    st.download_button(
        "일별 결과 CSV 다운로드",
        data=daily.to_csv(index=True).encode("utf-8-sig"),
        file_name="upbit_btc_adaptive_120_backtest.csv",
        mime="text/csv",
    )
    st.download_button(
        "성과지표 JSON 다운로드",
        data=json.dumps(m, ensure_ascii=False, indent=2, allow_nan=True).encode("utf-8"),
        file_name="upbit_btc_adaptive_120_metrics.json",
        mime="application/json",
    )

with comparison_tab:
    st.subheader("100·110·120·130·140일 이동평균 비교")
    st.caption(
        "모든 후보에 동일한 기간, 완충폭, 변동성 목표, 비용 조건을 적용합니다. "
        "수익률만이 아니라 MDD와 Sharpe를 함께 확인하세요."
    )
    comparison_display = ma_comparison.copy()
    for column in ["CAGR", "MDD", "누적수익률", "평균 익스포저"]:
        comparison_display[column] = comparison_display[column].map(lambda value: f"{value * 100:,.2f}%")
    comparison_display["Sharpe"] = comparison_display["Sharpe"].map(lambda value: f"{value:.2f}")
    comparison_display.index = [f"{days}일" for days in comparison_display.index]
    st.dataframe(comparison_display, width="stretch")

    chart_values = ma_comparison[["CAGR", "MDD"]].mul(100.0)
    chart_values.index = [f"{days}일" for days in chart_values.index]
    st.bar_chart(chart_values)

    best_cagr = int(ma_comparison["CAGR"].idxmax())
    best_sharpe = int(ma_comparison["Sharpe"].idxmax())
    st.info(
        f"전체 기간 최고 CAGR은 {best_cagr}일, 최고 Sharpe는 {best_sharpe}일입니다. "
        "전체 기간 1등만으로 확정하지 말고 아래 기간분할 결과도 함께 판단해야 합니다."
    )

    st.download_button(
        "이평선 비교 CSV 다운로드",
        data=ma_comparison.to_csv(index=True).encode("utf-8-sig"),
        file_name="btc_moving_average_comparison.csv",
        mime="text/csv",
    )

with validation_tab:
    st.subheader("시간순 3구간 기간분할 검증")
    st.caption(
        "평가기간을 일봉 개수가 비슷한 세 구간으로 나누어 독립적으로 다시 시작합니다. "
        "여러 구간에서 반복해서 양호한 후보가 한 구간에서만 높은 후보보다 견고합니다."
    )

    split_summary = split_results.groupby("이동평균").agg(
        기간별_중앙_CAGR=("CAGR", "median"),
        기간별_최저_CAGR=("CAGR", "min"),
        최악_MDD=("MDD", "min"),
        중앙_Sharpe=("Sharpe", "median"),
    )
    outperform_count = (
        split_results.assign(
            outperform=split_results["CAGR"] > split_results["단순보유 CAGR"]
        )
        .groupby("이동평균")["outperform"]
        .sum()
        .astype(int)
    )
    split_summary["단순보유보다_높은_CAGR_구간"] = outperform_count
    summary_display = split_summary.copy()
    for column in ["기간별_중앙_CAGR", "기간별_최저_CAGR", "최악_MDD"]:
        summary_display[column] = summary_display[column].map(lambda value: f"{value * 100:,.2f}%")
    summary_display["중앙_Sharpe"] = summary_display["중앙_Sharpe"].map(lambda value: f"{value:.2f}")
    summary_display["단순보유보다_높은_CAGR_구간"] = summary_display[
        "단순보유보다_높은_CAGR_구간"
    ].map(lambda value: f"{value}/3")
    summary_display.index = [f"{days}일" for days in summary_display.index]
    st.dataframe(summary_display, width="stretch")

    cagr_pivot = split_results.pivot(index="기간", columns="이동평균", values="CAGR")
    cagr_pivot.columns = [f"{days}일" for days in cagr_pivot.columns]
    cagr_display = cagr_pivot.apply(
        lambda column: column.map(lambda value: f"{value * 100:+.2f}%")
    )
    date_labels = split_results.drop_duplicates("기간").set_index("기간")[["시작일", "종료일"]]
    st.markdown("#### 구간별 CAGR")
    st.dataframe(date_labels.join(cagr_display), width="stretch")

    mdd_pivot = split_results.pivot(index="기간", columns="이동평균", values="MDD")
    mdd_pivot.columns = [f"{days}일" for days in mdd_pivot.columns]
    st.markdown("#### 구간별 MDD")
    mdd_display = mdd_pivot.apply(
        lambda column: column.map(lambda value: f"{value * 100:.2f}%")
    )
    st.dataframe(mdd_display, width="stretch")

    st.download_button(
        "기간분할 결과 CSV 다운로드",
        data=split_results.to_csv(index=False).encode("utf-8-sig"),
        file_name="btc_period_split_validation.csv",
        mime="text/csv",
    )

with monthly_tab:
    st.subheader("월별 수익률")
    st.dataframe(monthly_display, width="stretch")
    st.caption(
        "각 월의 전략 수익률이며, 맨 오른쪽 ‘연 수익률’은 해당 연도의 월 수익률을 복리로 합산한 값입니다."
    )
    st.download_button(
        "월별 수익률 CSV 다운로드",
        data=monthly_returns.to_csv(index=True).encode("utf-8-sig"),
        file_name="upbit_btc_adaptive_120_monthly_returns.csv",
        mime="text/csv",
    )

st.caption(
    "연구·교육용 백테스트입니다. 과거 성과는 미래 성과를 보장하지 않으며, "
    "파생상품의 실제 펀딩비·청산·체결오차는 단순화되어 있습니다."
)

