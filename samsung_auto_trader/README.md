# Samsung Auto Trader — 상세 설명서

이 문서는 `samsung_auto_trader` 서브프로젝트의 동작 원리와 구성 파일 역할, 트레이딩 로직을 상세히 설명합니다. 로컬 환경에서 EOD(종가) 기반 신호를 생성하고, 트레이더가 그 신호를 읽어 주문을 실행하는 워크플로우를 전제로 합니다.

※ 본 소프트웨어는 교육/데모용이며, 실제 자금 운용 전에는 충분한 검증이 필요합니다.

## 목차

- 개요
- 1) 구체적인 트레이딩 로직(Decision rules)
- 2) 각 파이썬 파일의 역할 및 상세 설명
- 3) 전체 구조 다이어그램(흐름도)
- 설치 및 실행(간단 예)

## 개요

기본 흐름:

1. `export_history.py`가 KIS API(모의 또는 실제)에서 일별 가격을 가져와 `Samsung_Daily_Data_yfinance.csv`를 업데이트합니다.
2. `tiny_gpt_trading_signal_real_cli.py`(AI 신호 생성기)가 CSV를 읽어 모델을 학습/추론하고 `latest_trading_signal.json` 및 `trading_signals_history.csv`를 생성합니다.
3. `main.py`가 실행되는 동안 `trader.py`가 최신 신호 파일을 읽어 매매를 결정·전송합니다. 또한 장 종료 후 23:00(Asia/Seoul)에 자동으로 1~2번의 일괄 갱신(내부 루프)을 수행하도록 설계되어 있습니다.

## 1) 구체적인 트레이딩 로직

아래는 `trader.py`에 구현된 핵심 규칙(요약, 구현세부는 코드 참고)입니다.

- 신호 로드 및 검증
  - `latest_trading_signal.json`의 필수 항목: `symbol`, `as_of_date`, `prediction.{trading_signal,confidence,normalized_entropy}`, `training_summary.best_validation_balanced_accuracy`
  - 날짜가 오늘보다 이후이면 무시
  - `confidence` 최소값: 0.45 (이하면 실행 차단)
  - `normalized_entropy` 최대값: 0.95 (초과 시 불확실하다고 간주해 차단)
  - `best_validation_balanced_accuracy` 최소값: 0.36 (모델 검증 성능이 낮으면 차단)

- 신호 해석
  - `prediction.trading_signal`이 `BUY` / `HOLD` / `SELL` 중 하나여야 함
  - `HOLD`이면 아무 작업도 하지 않음

- 매수 규칙
  - 사용 가능한 현금(`available_cash`)의 최대 10% 또는 전체 주식 가치의 30% 한도(현재 보유량 고려) 중 작은 값으로 주문금액 한도를 정함
  - 그 한도 내에서 1주 이상 구매 가능한 수량을 산정(정수 수량)
  - 가격: 현재 시세에 `buy_offset`(예: -2000) 적용 (구성값 참조)
  - 그 결과 수량이 0이면 매수 생략

- 매도 규칙
  - 전량(holding_qty 전량) 매도하도록 설계(간단한 청산 전략)
  - 가격: 현재 시세에 `sell_offset` 적용

- 안전 장치
  - 주문 실패 또는 API 오류 발생 시 로그로 남기고 재시도 로직은 적절히 확장 가능(현재는 단순 실패 로깅)
  - 거래 시간(기본: 09:10 ~ 15:30 KST) 외에는 매매 사이클을 수행하지 않음

## 2) 각 파이썬 파일의 역할 및 상세 설명

다음은 `samsung_auto_trader` 폴더 아래 주요 파일과 그 상세 역할입니다.

- `main.py`
  - 애플리케이션 진입점. 로깅 구성과 `Settings` 로드 후 `AutoTrader`를 생성하여 `run()`을 호출합니다.

- `trader.py`
  - `AutoTrader` 클래스의 구현체가 위치합니다.
  - 신호 파일 로드(`_load_latest_signal`) → 검증(`_validate_signal`) → `_signal_order`로 BUY/SELL 결정
  - `_build_buy_order`, `_build_sell_order`에서 주문 파라미터 생성 후 `orders.place_order` 호출
  - 일중 주기적 `_trade_cycle()` 호출로 시장 감시 및 주문 실행
  - 장 종료 후 일일 업데이트 타이밍(23:00 KST)에 `export_historical_prices`와 `generate_signal`을 수행하도록 통합되어 있습니다.

- `export_history.py`
  - KIS API를 호출하여 지정한 기간/심볼의 히스토리(일봉/주봉/월봉)를 가져옵니다.
  - 기존 CSV와 병합(_merge_rows)하여 중복 제거 및 날짜 정규화 후 저장합니다.
  - 재사용 가능한 함수 `export_historical_prices(...)`를 제공합니다.

- `tiny_gpt_trading_signal_real_cli.py`
  - 데이터 전처리, 작은 GPT 스타일 모델 구성, 학습 루프, 예측 루틴이 포함되어 있습니다.
  - `generate_signal(...)` 함수는 CSV를 읽고 모델을 학습/평가하여 최신 신호 JSON 및 히스토리 CSV를 생성합니다.
  - `main()`은 CLI 인자를 받아 위 함수를 호출합니다.

- `api_client.py`
  - Korea Investment API 호출용 래퍼입니다. 인증 토큰 자동 갱신, 기본 URL, 요청/응답 로깅을 처리합니다.

- `auth.py` / `TokenManager`
  - OAuth 토큰 발급/갱신 및 `token_cache.json`에 캐시하는 로직을 포함합니다. 민감정보는 커밋 금지입니다.

- `market_data.py`
  - `get_current_price()`와 `get_historical_prices()` 같은 헬퍼를 제공하여 실시간 시세 및 히스토리 호출을 표준화합니다.

- `account.py`
  - 계좌 잔고/예수금, 보유종목 조회를 위한 간단한 래퍼 함수가 위치합니다.

- `orders.py`
  - 매수/매도 주문을 단일 함수(`place_order`)로 추상화합니다. 주문 결과(`OrderResult`)를 반환합니다.

- `config.py`
  - `Settings` dataclass로 환경 변수/설정값을 중앙에서 관리합니다.

- `logger.py`
  - 로깅 포맷/레벨 초기화 도우미를 제공합니다.

- `test_trading_logic.py`
  - 신호→주문 결정 로직의 단위 테스트 및 시나리오 테스트가 포함되어 있습니다. (로컬에서 실행하여 보정 가능)

- 데이터 파일
  - `Samsung_Daily_Data_yfinance.csv`: export_history가 갱신하는 내부 EOD CSV
  - `latest_trading_signal.json`: AI가 생성한 최신 신호(로컬 보존, .gitignore에 추가됨)
  - `trading_signals_history.csv`: 과거 예측 로그(로컬 보존)

> 중요한 유의사항: 대부분 내부 모듈이 `from auth import TokenManager`와 같이 상대 모듈 임포트를 전제로 합니다. 레포를 분리할 경우 패키징(예: `pip install -e .` 또는 `python -m samsung_auto_trader.main`)을 권장합니다.

## 3) 전체 구조 다이어그램

아래 mermaid 다이어그램은 전체 데이터 흐름과 매칭되는 구성 요소를 보여 줍니다.

```mermaid
flowchart LR
  subgraph Data
    CSV[Samsung_Daily_Data_yfinance.csv]
    JSON[latest_trading_signal.json]
    HIST[trading_signals_history.csv]
  end

  subgraph Ingest
    EH[export_history.py]
  end

  subgraph AI
    TG[tiny_gpt_trading_signal_real_cli.py]
  end

  subgraph Trading
    MAIN[main.py]
    TR[trader.py]
    AC[account.py]
    MD[market_data.py]
    ORD[orders.py]
  end

  EH --> CSV
  CSV --> TG
  TG --> JSON
  TG --> HIST
  JSON --> TR
  TR --> MD
  TR --> AC
  TR --> ORD

  %% Scheduled nightly update
  MAIN -->|runs TR.run()| TR
  TR -->|23:00 nightly| EH
  TR -->|after export| TG

  style Data fill:#f9f,stroke:#333,stroke-width:1px
  style AI fill:#bbf,stroke:#333
  style Trading fill:#bfb,stroke:#333
```

## 설치 및 빠른 시작 (예)

1) 의존성 설치

```bash
cd samsung_auto_trader
pip install -r requirements.txt
```

2) 환경 변수 설정(샘플)

```bash
export GH_ACCOUNT="your_account"
export GH_APPKEY="your_appkey"
export GH_APPSECRET="your_appsecret"
export PRODUCT_CODE="01"
```

3) (수동) 히스토리 갱신 → 신호 생성 → 트레이더 실행

```bash
# 1) 히스토리 갱신
PYTHONPATH=./samsung_auto_trader python samsung_auto_trader/export_history.py --symbol 005930

# 2) 신호 생성
PYTHONPATH=./samsung_auto_trader python samsung_auto_trader/tiny_gpt_trading_signal_real_cli.py

# 3) 트레이더 실행
PYTHONPATH=./samsung_auto_trader python samsung_auto_trader/main.py
```

