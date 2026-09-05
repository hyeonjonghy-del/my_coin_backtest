# Upbit BTC/KRW Daily Trend-Recovery Strategy

업비트 BTC/KRW 일봉으로 돌파와 눌림목을 함께 테스트하는 현물 롱온리 전략입니다.

## 핵심 규칙

- 추세 필터: 종가 > SMA200, SMA20 > SMA200
- 돌파: 종가가 직전 20일 최고가를 넘고 거래량이 직전 20일 평균의 1.2배 이상
- 얕은 눌림: SMA10을 장중 시험한 뒤 SMA10 위 양봉 마감, RSI 50~70
- 깊은 눌림: SMA20을 장중 시험한 뒤 SMA20 위 양봉 마감, RSI 45~65
- 모든 신호는 확정 일봉에서 계산하고 다음 날 시가에 체결
- SMA50 <= SMA200이면 최대 투자비중 35%, 골든크로스 이후 60%
- 거래당 허용손실 0.5%, ATR 기반 초기·추적 손절
- 2R 도달 시 35% 분할익절, SMA20 아래 2일 연속 마감 시 다음 날 청산
- 수수료 0.05%와 슬리피지 0.05%를 매수·매도 양쪽에 적용

일봉 OHLC만으로 손절과 목표가가 같은 날 모두 닿으면 정확한 장중 순서를 알 수 없으므로 손절을 먼저 처리합니다.

## 실행

저장소 루트에서:

```powershell
python run.py --start 2018-01-01
```

인터넷 없이 자체 CSV로 실행하려면 `date_utc,open,high,low,close,volume` 열을 준비합니다.

```powershell
python run.py --csv C:\path\btc_daily.csv
```

결과는 기본적으로 저장소 루트의 `results`에 저장됩니다.

- `upbit_btc_krw_daily.csv`: 원본 일봉
- `signals_and_indicators.csv`: 지표와 매수 신호
- `equity_curve.csv`: 일별 평가금액
- `trades.csv`: 거래 내역
- `metrics.json`: 전략 및 매수후보유 CAGR·MDD, 변동성, 승률 등

연구·교육용 코드이며 실제 수익을 보장하지 않습니다.

## Adaptive 120 V2

단순보유보다 높은 장기 수익률과 낮은 MDD를 함께 목표로 한 연구 후보입니다.

- 종가가 SMA120의 2% 위로 올라가면 추세 진입
- 종가가 SMA120의 2% 아래로 내려가면 현금화
- 두 밴드 사이에서는 기존 상태 유지
- 20일 실현변동성을 기준으로 연 80% 변동성을 목표
- 익스포저 범위 0.25~1.00배의 업비트 현물 운용
- 거래비용은 비중 변경액의 0.1%
- 반감기는 결과 보고용 정보로만 사용하고 매매 신호에는 사용하지 않음

```powershell
python run_adaptive_120.py --download-start 2018-01-01 --evaluation-start 2018-05-01
```

이미 받은 CSV를 사용할 수도 있습니다.

```powershell
python run_adaptive_120.py --csv results/upbit_btc_krw_daily.csv
```

## Streamlit 웹 앱

로컬 실행:

```powershell
python -m streamlit run streamlit_app.py
```

GitHub에 푸시한 후 Streamlit Community Cloud에서 다음 값을 선택합니다.

- Repository: 이 프로젝트를 올린 GitHub 저장소
- Branch: `master` 또는 실제 배포 브랜치
- Main file path: `streamlit_app.py`

배포용 `requirements.txt`는 앱 파일과 같은 폴더에 있습니다. 앱은 업비트 공개 API를
6시간 캐시하고 결과를 다운로드 버튼으로 제공하므로 API 키와 영구 파일 저장소가
필요하지 않습니다.

웹 앱에는 다음 견고성 검증 기능이 포함됩니다.

- 100·110·120·130·140일 이동평균을 동일 조건으로 비교
- 평가기간을 시간순 3개 구간으로 나누어 CAGR·MDD·Sharpe 재검증
- 전체 기간과 기간분할 결과를 각각 CSV로 다운로드
- 장기 차트의 표시 점을 줄이되 시작·종료점과 최저·최고점은 보존
- 급락 후 회복 재진입과 안정형 변동성 비중 조절을 각각·결합 형태로 통제 실험
- 두 추가 로직도 동일한 시간순 3구간에서 재검증하고 CSV로 다운로드

### 두 로직 실험의 고정 규칙

- 회복 재진입: 최근 90일 고점 대비 20% 이상 급락한 이력이 있는 상태에서
  최근 30일 저점 대비 10% 반등하고 SMA120 하단 밴드로 복귀하면 재진입
- 안정형 변동성 비중: 20일과 60일 연율 변동성 중 높은 값을 사용하며,
  비중 축소는 즉시 반영하고 비중 증액은 목표와의 차이 중 25%씩 반영
- 결과가 기존 Adaptive 120의 CAGR·MDD·Sharpe를 모두 개선하지 못하면
  기본 전략으로 자동 채택하지 않고 연구 후보로만 표시
