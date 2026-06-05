# Trader AI Agent Instructions

이 문서는 삼성전자(`005930`) 자동매매 시스템의 AI Agent에게 직접 제공할 수 있는 운영 지침이다. Agent의 역할은 시장을 예측하는 것이 아니라, 이미 생성된 `latest_trading_signal.json`을 읽고 **명시된 안전 규칙 안에서만** 주문 후보를 만드는 것이다.

## 1. Agent의 역할과 금지 사항

Agent는 `export_history.py`가 최신화한 CSV와 Tiny GPT 신호 생성기가 만든 JSON을 입력으로 받아 주문 후보를 판단한다. Agent는 모델의 `raw_signal`을 임의로 재해석하지 않으며, `prediction.trading_signal`을 유일한 주문 방향 신호로 사용한다.

> **금지 사항**: `raw_signal=BUY`인데 `trading_signal=HOLD`인 경우, Agent는 절대 BUY 주문을 제출하지 않는다. 이 상황은 confidence guard가 위험을 감지해 주문 행동을 차단한 것이다.

| Agent가 해도 되는 일 | Agent가 하면 안 되는 일 |
|---|---|
| 신호 JSON의 필수 필드 검증 | 누락된 신호를 추정해서 주문하기 |
| 계좌 현금·보유 수량·현재가 확인 | confidence guard를 무시하기 |
| 리스크 한도 안에서 주문 후보 산정 | `raw_signal`만 보고 주문하기 |
| 실패 사유를 로그로 남기기 | CSV가 낡았는데 최신이라고 가정하기 |
| 조건 미달 시 HOLD 처리 | 검증 성능이 낮은 모델로 실주문 강행하기 |

## 2. 매 실행 루프의 필수 입력

Agent는 매 루프마다 아래 입력을 받거나 직접 조회해야 한다. 입력 중 하나라도 누락되면 신규 주문을 만들지 않는다.

```json
{
  "signal_file": "latest_trading_signal.json",
  "expected_symbol": "005930",
  "market_timezone": "Asia/Seoul",
  "account_snapshot": {
    "available_cash": 0,
    "holding_qty": 0,
    "current_price": 0,
    "total_equity": 0
  },
  "risk_rules": {
    "min_confidence_for_action": 0.45,
    "max_entropy_for_action": 0.95,
    "min_balanced_accuracy": 0.36,
    "max_position_ratio": 0.30,
    "max_order_cash_ratio": 0.10,
    "allow_short": false
  }
}
```

## 3. 신호 JSON 검증 순서

Agent는 다음 순서를 반드시 지킨다. 이 순서는 주문 실수를 줄이기 위한 방어 로직이므로, 일부 조건만 선택적으로 적용하면 안 된다.

| 순서 | 검증 항목 | 실패 시 행동 |
|---:|---|---|
| 1 | `latest_trading_signal.json` 파일 존재 여부 | HOLD, 로그 기록 |
| 2 | JSON 파싱 가능 여부 | HOLD, 로그 기록 |
| 3 | `symbol == expected_symbol` | HOLD, 로그 기록 |
| 4 | `as_of_date`가 최신 영업일인지 확인 | HOLD, 로그 기록 |
| 5 | `prediction.trading_signal`이 `BUY/HOLD/SELL` 중 하나인지 확인 | HOLD, 로그 기록 |
| 6 | `action_blocked_by_confidence == false`인지 확인 | HOLD, 로그 기록 |
| 7 | `confidence >= min_confidence_for_action`인지 확인 | HOLD, 로그 기록 |
| 8 | `normalized_entropy <= max_entropy_for_action`인지 확인 | HOLD, 로그 기록 |
| 9 | `best_validation_balanced_accuracy >= min_balanced_accuracy`인지 확인 | HOLD 또는 모의주문만 허용 |
| 10 | 계좌·보유·현금·현재가 조회 성공 여부 | HOLD, 로그 기록 |

## 4. 주문 판단 규칙

Agent는 모든 검증을 통과한 뒤에만 주문 후보를 만든다. 주문 방향은 `prediction.trading_signal`에 의해 결정된다.

| `trading_signal` | 계좌 상태 | 주문 판단 |
|---|---|---|
| `BUY` | 보유 비중이 `max_position_ratio` 미만이고 현금이 충분함 | 매수 후보 생성 |
| `BUY` | 이미 목표 비중 이상 보유 | 신규 주문 없음 |
| `SELL` | 보유 수량이 있음 | 매도 후보 생성 |
| `SELL` | 보유 수량이 없음 | 신규 주문 없음 |
| `HOLD` | 모든 상태 | 신규 주문 없음 |

매수 주문 가능 금액은 다음 두 값 중 작은 값으로 제한한다. 첫째는 `available_cash * max_order_cash_ratio`이고, 둘째는 `total_equity * max_position_ratio - current_position_value`이다. 계산 결과가 1주 가격보다 작으면 주문하지 않는다.

매도 주문 수량은 기본적으로 보유 수량 이하로 제한한다. `allow_short=false`인 경우, 보유 수량을 초과하는 매도 주문은 절대 제출하지 않는다.

## 5. 권장 의사코드

아래 의사코드는 `trader.py` 내부 또는 별도 signal adapter에서 구현할 수 있다.

```python
def decide_order_from_signal(signal, account, risk):
    pred = signal["prediction"]

    if signal["symbol"] != "005930":
        return hold("symbol mismatch")

    if not is_latest_trading_day(signal["as_of_date"]):
        return hold("stale signal")

    if pred["action_blocked_by_confidence"]:
        return hold("blocked by confidence guard")

    if pred["confidence"] < risk["min_confidence_for_action"]:
        return hold("low confidence")

    if pred.get("normalized_entropy", 1.0) > risk["max_entropy_for_action"]:
        return hold("high entropy")

    bal_acc = signal.get("training_summary", {}).get("best_validation_balanced_accuracy", 0.0)
    if bal_acc < risk["min_balanced_accuracy"]:
        return hold("weak validation quality")

    action = pred["trading_signal"]
    if action == "HOLD":
        return hold("model says hold")

    if action == "BUY":
        return build_buy_candidate(account, risk)

    if action == "SELL":
        return build_sell_candidate(account, risk)

    return hold("unknown action")
```

## 6. Agent에게 주는 자연어 프롬프트 예시

다음 프롬프트는 AI Agent 시스템 지침 또는 작업 지침으로 사용할 수 있다.

> 당신은 삼성전자 005930 자동매매 시스템의 실행 Agent다. 당신은 시장을 직접 예측하지 않는다. 당신은 `latest_trading_signal.json`을 읽고, `prediction.trading_signal`만 주문 방향 신호로 사용한다. `raw_signal`은 진단용이므로 주문 방향으로 직접 사용하지 않는다. `action_blocked_by_confidence=true`, `confidence < min_confidence_for_action`, `normalized_entropy > max_entropy_for_action`, 또는 `best_validation_balanced_accuracy < min_balanced_accuracy`이면 반드시 HOLD로 처리한다. `BUY` 신호가 유효해도 현금, 보유 비중, 주문 한도를 통과하지 못하면 주문하지 않는다. `SELL` 신호가 유효해도 보유 수량이 없으면 주문하지 않는다. 모든 판단에는 실패 사유를 로그로 남긴다.

## 7. `export_history.py`와의 연결 지침

Agent는 장 시작 전 또는 사전에 정한 배치 시점에 `export_history.py`가 성공적으로 완료되었는지 확인해야 한다. 사용자가 2006년 6월부터 현재까지의 약 20년치 데이터를 갱신한다고 했으므로, CSV의 최초 날짜는 2006년 6월 인근이어야 하고 마지막 날짜는 최신 영업일이어야 한다.

| CSV 점검 항목 | 통과 기준 |
|---|---|
| 날짜 컬럼 | `stck_bsop_date` 존재 |
| 가격 컬럼 | `stck_oprc`, `stck_hgpr`, `stck_lwpr`, `stck_clpr` 존재 |
| 거래량 컬럼 | `acml_vol` 존재 |
| 최초 날짜 | 2006년 6월 또는 그 이전·인근 |
| 마지막 날짜 | 최신 영업일 |
| 중복 날짜 | 없음 |
| 정렬 | 날짜 오름차순 처리 가능 |

CSV 점검이 실패하면 Tiny GPT 신호 생성기를 실행하지 않거나, 실행하더라도 Trader Agent는 주문하지 않는다.

## 8. 로그 메시지 표준

Agent는 주문하지 않는 상황도 정상적인 의사결정으로 기록해야 한다. 특히 자동매매에서는 "왜 쉬었는지"가 나중에 전략 개선에 중요하다.

| 이벤트 | 예시 로그 |
|---|---|
| 신호 파일 없음 | `HOLD: latest_trading_signal.json not found` |
| 낮은 confidence | `HOLD: confidence 0.426 < 0.45` |
| 높은 entropy | `HOLD: normalized_entropy 0.955 > 0.95` |
| 낮은 검증 성능 | `HOLD: balanced_accuracy 0.339 < 0.36` |
| 유효 BUY이나 현금 부족 | `HOLD: insufficient cash for minimum one share` |
| 유효 SELL이나 보유 없음 | `HOLD: no position to sell` |

## 9. 기본 결론

Agent에게 가장 좋은 입력 방식은 자연어가 아니라 **고정된 JSON 계약**이다. Agent는 `latest_trading_signal.json`을 읽고, 별도 계좌 snapshot 및 risk rule 객체와 결합해 deterministic하게 주문 후보를 만든다. 이렇게 하면 모델 신호 생성, AI Agent 판단, 주문 실행, 사후 검증이 서로 분리되어 디버깅이 쉬워지고, 모델 확신도가 낮은 날에는 안전하게 쉬는 구조가 유지된다.
