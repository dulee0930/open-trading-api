#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiny_gpt_trading_signal_real_cli.py의 거래 로직 테스트.

Trader AI Agent Instructions.md의 지침을 따르는 거래 판단 함수들의 동작을 검증합니다.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple


# ====== 거래 로직 클래스 및 함수 (tiny_gpt_trading_signal_real_cli.py에서 추출) ======

@dataclass
class AccountSnapshot:
    """거래 판단에 필요한 계좌 상태 스냅샷."""
    available_cash: float
    holding_qty: int
    current_price: float
    total_equity: float


@dataclass
class RiskRules:
    """거래 판단 시 적용할 리스크 규칙."""
    min_confidence_for_action: float = 0.45
    max_entropy_for_action: float = 0.95
    min_balanced_accuracy: float = 0.36
    max_position_ratio: float = 0.30
    max_order_cash_ratio: float = 0.10
    allow_short: bool = False


@dataclass
class OrderCandidate:
    """주문 생성 결과."""
    order_type: str  # "BUY", "SELL", "HOLD"
    quantity: int
    reason: str  # 주문 또는 거절 사유
    estimated_value: float = 0.0  # BUY/SELL 시 예상 거래 금액


def validate_signal(
    signal: Dict[str, object],
    expected_symbol: str,
    as_of_date_str: str,
) -> Tuple[bool, str]:
    """신호 JSON 검증 (10단계 순서)."""
    
    if signal.get("symbol") != expected_symbol:
        return False, f"HOLD: symbol mismatch. expected={expected_symbol}, got={signal.get('symbol')}"
    
    signal_date = signal.get("as_of_date", "")
    if signal_date != as_of_date_str:
        return False, f"HOLD: stale signal. as_of_date={signal_date}, expected={as_of_date_str}"
    
    pred = signal.get("prediction")
    if not isinstance(pred, dict):
        return False, "HOLD: prediction field missing or not a dict"
    
    trading_signal = pred.get("trading_signal")
    if trading_signal not in ["BUY", "HOLD", "SELL"]:
        return False, f"HOLD: invalid trading_signal={trading_signal}"
    
    if pred.get("action_blocked_by_confidence", False):
        return False, "HOLD: blocked by confidence guard"
    
    confidence = pred.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        return False, "HOLD: confidence is not numeric"
    
    normalized_entropy = pred.get("normalized_entropy", 0.0)
    if not isinstance(normalized_entropy, (int, float)):
        return False, "HOLD: normalized_entropy is not numeric"
    
    training_summary = signal.get("training_summary", {})
    bal_acc = training_summary.get("best_validation_balanced_accuracy", 0.0)
    if not isinstance(bal_acc, (int, float)):
        return False, "HOLD: best_validation_balanced_accuracy is not numeric"
    
    return True, ""


def check_risk_conditions(
    signal: Dict[str, object],
    risk: RiskRules,
) -> Tuple[bool, str]:
    """신호의 위험 조건들을 확인 (검증 단계 7~9)."""
    
    pred = signal.get("prediction", {})
    confidence = pred.get("confidence", 0.0)
    normalized_entropy = pred.get("normalized_entropy", 1.0)
    training_summary = signal.get("training_summary", {})
    bal_acc = training_summary.get("best_validation_balanced_accuracy", 0.0)
    
    if confidence < risk.min_confidence_for_action:
        return False, f"HOLD: confidence {confidence:.3f} < {risk.min_confidence_for_action}"
    
    if normalized_entropy > risk.max_entropy_for_action:
        return False, f"HOLD: normalized_entropy {normalized_entropy:.3f} > {risk.max_entropy_for_action}"
    
    if bal_acc < risk.min_balanced_accuracy:
        return False, f"HOLD: balanced_accuracy {bal_acc:.3f} < {risk.min_balanced_accuracy}"
    
    return True, ""


def build_buy_candidate(
    account: AccountSnapshot,
    risk: RiskRules,
    current_price: float,
) -> OrderCandidate:
    """매수 후보 생성 (trading_signal=BUY일 때)."""
    
    current_position_value = account.holding_qty * current_price
    max_by_cash = account.available_cash * risk.max_order_cash_ratio
    target_position_value = account.total_equity * risk.max_position_ratio
    max_by_position = max(0, target_position_value - current_position_value)
    max_order_cash = min(max_by_cash, max_by_position)
    
    if max_order_cash < current_price:
        return OrderCandidate(
            order_type="HOLD",
            quantity=0,
            reason="insufficient cash for minimum one share",
            estimated_value=0.0,
        )
    
    quantity = int(max_order_cash / current_price)
    estimated_value = quantity * current_price
    
    return OrderCandidate(
        order_type="BUY",
        quantity=quantity,
        reason=f"buy signal accepted: {quantity} shares @ {current_price}",
        estimated_value=estimated_value,
    )


def build_sell_candidate(
    account: AccountSnapshot,
    risk: RiskRules,
    current_price: float,
) -> OrderCandidate:
    """매도 후보 생성 (trading_signal=SELL일 때)."""
    
    if account.holding_qty <= 0:
        return OrderCandidate(
            order_type="HOLD",
            quantity=0,
            reason="no position to sell",
            estimated_value=0.0,
        )
    
    quantity = account.holding_qty
    estimated_value = quantity * current_price
    
    return OrderCandidate(
        order_type="SELL",
        quantity=quantity,
        reason=f"sell signal accepted: {quantity} shares @ {current_price}",
        estimated_value=estimated_value,
    )


def decide_order_from_signal(
    signal: Dict[str, object],
    account: AccountSnapshot,
    risk: RiskRules,
    expected_symbol: str,
    as_of_date_str: str,
) -> OrderCandidate:
    """권장 의사코드 기반 주문 판단."""
    
    signal_valid, signal_fail_reason = validate_signal(signal, expected_symbol, as_of_date_str)
    if not signal_valid:
        return OrderCandidate(order_type="HOLD", quantity=0, reason=signal_fail_reason)
    
    risk_ok, risk_fail_reason = check_risk_conditions(signal, risk)
    if not risk_ok:
        return OrderCandidate(order_type="HOLD", quantity=0, reason=risk_fail_reason)
    
    if account.total_equity <= 0:
        return OrderCandidate(order_type="HOLD", quantity=0, reason="HOLD: total_equity is zero or negative")
    
    pred = signal.get("prediction", {})
    action = pred.get("trading_signal", "HOLD")
    current_price = account.current_price
    
    if action == "HOLD":
        return OrderCandidate(
            order_type="HOLD",
            quantity=0,
            reason="model signal is HOLD",
        )
    
    if action == "BUY":
        return build_buy_candidate(account, risk, current_price)
    
    if action == "SELL":
        return build_sell_candidate(account, risk, current_price)
    
    return OrderCandidate(
        order_type="HOLD",
        quantity=0,
        reason=f"unknown trading_signal: {action}",
    )


def test_validate_signal():
    """신호 검증 함수 테스트."""
    print("=" * 60)
    print("TEST 1: validate_signal() - 신호 검증")
    print("=" * 60)
    
    # 정상 신호
    valid_signal = {
        "symbol": "005930",
        "as_of_date": "2024-01-15",
        "prediction": {
            "trading_signal": "BUY",
            "action_blocked_by_confidence": False,
        },
        "training_summary": {
            "best_validation_balanced_accuracy": 0.50,
        },
    }
    
    passed, reason = validate_signal(valid_signal, "005930", "2024-01-15")
    print(f"✓ 정상 신호: passed={passed}, reason='{reason}'")
    
    # 심볼 불일치
    invalid_symbol = {**valid_signal, "symbol": "000270"}
    passed, reason = validate_signal(invalid_symbol, "005930", "2024-01-15")
    print(f"✓ 심볼 불일치: passed={passed}")
    print(f"  → {reason}")
    
    # 오래된 신호
    passed, reason = validate_signal(valid_signal, "005930", "2024-01-16")
    print(f"✓ 오래된 신호: passed={passed}")
    print(f"  → {reason}")
    
    print()


def test_check_risk_conditions():
    """위험 조건 확인 테스트."""
    print("=" * 60)
    print("TEST 2: check_risk_conditions() - 위험 조건 확인")
    print("=" * 60)
    
    risk = RiskRules(
        min_confidence_for_action=0.45,
        max_entropy_for_action=0.95,
        min_balanced_accuracy=0.36,
    )
    
    # 높은 신뢰도 신호
    good_signal = {
        "prediction": {
            "confidence": 0.65,
            "normalized_entropy": 0.72,
        },
        "training_summary": {
            "best_validation_balanced_accuracy": 0.50,
        },
    }
    
    passed, reason = check_risk_conditions(good_signal, risk)
    print(f"✓ 좋은 신호: passed={passed}")
    
    # 낮은 신뢰도 신호
    low_conf_signal = {
        "prediction": {
            "confidence": 0.40,  # < 0.45
            "normalized_entropy": 0.72,
        },
        "training_summary": {
            "best_validation_balanced_accuracy": 0.50,
        },
    }
    
    passed, reason = check_risk_conditions(low_conf_signal, risk)
    print(f"✓ 낮은 신뢰도: passed={passed}")
    print(f"  → {reason}")
    
    # 높은 엔트로피 신호
    high_entropy_signal = {
        "prediction": {
            "confidence": 0.65,
            "normalized_entropy": 0.98,  # > 0.95
        },
        "training_summary": {
            "best_validation_balanced_accuracy": 0.50,
        },
    }
    
    passed, reason = check_risk_conditions(high_entropy_signal, risk)
    print(f"✓ 높은 엔트로피: passed={passed}")
    print(f"  → {reason}")
    
    print()


def test_buy_logic():
    """매수 로직 테스트."""
    print("=" * 60)
    print("TEST 3: build_buy_candidate() - 매수 판단")
    print("=" * 60)
    
    risk = RiskRules(
        max_position_ratio=0.30,
        max_order_cash_ratio=0.10,
        allow_short=False,
    )
    
    # 케이스 1: 충분한 현금
    account = AccountSnapshot(
        available_cash=1000000,
        holding_qty=0,
        current_price=60000,
        total_equity=1000000,
    )
    
    order = build_buy_candidate(account, risk, 60000)
    print(f"✓ 충분한 현금 케이스:")
    print(f"  - 주문 타입: {order.order_type}")
    print(f"  - 주문 수량: {order.quantity} 주")
    print(f"  - 예상 거래금: {order.estimated_value:,.0f}원")
    print(f"  - 사유: {order.reason}")
    
    # 케이스 2: 보유 비중 제한
    account2 = AccountSnapshot(
        available_cash=1000000,
        holding_qty=5000,  # 이미 30% 이상 보유
        current_price=60000,
        total_equity=1000000,
    )
    
    order2 = build_buy_candidate(account2, risk, 60000)
    print(f"\n✓ 보유 비중 제한 케이스:")
    print(f"  - 주문 타입: {order2.order_type}")
    print(f"  - 주문 수량: {order2.quantity} 주")
    print(f"  - 사유: {order2.reason}")
    
    # 케이스 3: 현금 부족
    account3 = AccountSnapshot(
        available_cash=10000,  # 충분하지 않음
        holding_qty=0,
        current_price=60000,
        total_equity=1000000,
    )
    
    order3 = build_buy_candidate(account3, risk, 60000)
    print(f"\n✓ 현금 부족 케이스:")
    print(f"  - 주문 타입: {order3.order_type}")
    print(f"  - 주문 수량: {order3.quantity} 주")
    print(f"  - 사유: {order3.reason}")
    
    print()


def test_sell_logic():
    """매도 로직 테스트."""
    print("=" * 60)
    print("TEST 4: build_sell_candidate() - 매도 판단")
    print("=" * 60)
    
    risk = RiskRules(allow_short=False)
    
    # 케이스 1: 보유 있음
    account = AccountSnapshot(
        available_cash=500000,
        holding_qty=100,
        current_price=60000,
        total_equity=1000000,
    )
    
    order = build_sell_candidate(account, risk, 60000)
    print(f"✓ 보유 있음 케이스:")
    print(f"  - 주문 타입: {order.order_type}")
    print(f"  - 주문 수량: {order.quantity} 주")
    print(f"  - 예상 거래금: {order.estimated_value:,.0f}원")
    print(f"  - 사유: {order.reason}")
    
    # 케이스 2: 보유 없음
    account2 = AccountSnapshot(
        available_cash=1000000,
        holding_qty=0,  # 보유 없음
        current_price=60000,
        total_equity=1000000,
    )
    
    order2 = build_sell_candidate(account2, risk, 60000)
    print(f"\n✓ 보유 없음 케이스:")
    print(f"  - 주문 타입: {order2.order_type}")
    print(f"  - 주문 수량: {order2.quantity} 주")
    print(f"  - 사유: {order2.reason}")
    
    print()


def test_full_decision_flow():
    """전체 의사결정 흐름 테스트."""
    print("=" * 60)
    print("TEST 5: decide_order_from_signal() - 전체 거래 판단")
    print("=" * 60)
    
    risk = RiskRules(
        min_confidence_for_action=0.45,
        max_entropy_for_action=0.95,
        min_balanced_accuracy=0.36,
        max_position_ratio=0.30,
        max_order_cash_ratio=0.10,
    )
    
    account = AccountSnapshot(
        available_cash=1000000,
        holding_qty=0,
        current_price=60000,
        total_equity=1000000,
    )
    
    # 시나리오 1: 유효한 BUY 신호
    signal_buy = {
        "symbol": "005930",
        "as_of_date": "2024-01-15",
        "prediction": {
            "trading_signal": "BUY",
            "confidence": 0.65,
            "normalized_entropy": 0.72,
            "action_blocked_by_confidence": False,
        },
        "training_summary": {
            "best_validation_balanced_accuracy": 0.50,
        },
    }
    
    order = decide_order_from_signal(
        signal_buy, account, risk, "005930", "2024-01-15"
    )
    print(f"✓ 유효한 BUY 신호:")
    print(f"  - 주문 타입: {order.order_type}")
    print(f"  - 주문 수량: {order.quantity} 주")
    print(f"  - 사유: {order.reason}")
    
    # 시나리오 2: HOLD 신호
    signal_hold = {
        "symbol": "005930",
        "as_of_date": "2024-01-15",
        "prediction": {
            "trading_signal": "HOLD",
            "confidence": 0.55,
            "normalized_entropy": 0.72,
            "action_blocked_by_confidence": False,
        },
        "training_summary": {
            "best_validation_balanced_accuracy": 0.50,
        },
    }
    
    order = decide_order_from_signal(
        signal_hold, account, risk, "005930", "2024-01-15"
    )
    print(f"\n✓ HOLD 신호:")
    print(f"  - 주문 타입: {order.order_type}")
    print(f"  - 주문 수량: {order.quantity} 주")
    print(f"  - 사유: {order.reason}")
    
    # 시나리오 3: confidence guard 활성화
    signal_blocked = {
        "symbol": "005930",
        "as_of_date": "2024-01-15",
        "prediction": {
            "trading_signal": "BUY",
            "confidence": 0.30,  # < 0.45
            "normalized_entropy": 0.72,
            "action_blocked_by_confidence": True,
        },
        "training_summary": {
            "best_validation_balanced_accuracy": 0.50,
        },
    }
    
    order = decide_order_from_signal(
        signal_blocked, account, risk, "005930", "2024-01-15"
    )
    print(f"\n✓ Confidence guard 활성화:")
    print(f"  - 주문 타입: {order.order_type}")
    print(f"  - 주문 수량: {order.quantity} 주")
    print(f"  - 사유: {order.reason}")
    
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "거래 로직 검증 테스트".center(58) + "║")
    print("║" + "(Trader AI Agent Instructions 기반)".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    test_validate_signal()
    test_check_risk_conditions()
    test_buy_logic()
    test_sell_logic()
    test_full_decision_flow()
    
    print("=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)
