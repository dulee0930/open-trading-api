import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from api_client import ApiClient

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    side: str
    symbol: str
    quantity: int
    price: int
    success: bool
    raw: Dict[str, Any]
    message: Optional[str] = None
    order_reference: Optional[str] = None


def _extract_output(data: Dict[str, Any]) -> Dict[str, Any]:
    output = data.get("output") or data.get("output1") or data.get("output2") or {}
    if isinstance(output, list):
        return output[0] if output else {}
    if isinstance(output, dict):
        return output
    return {}


def _extract_order_reference(output: Dict[str, Any]) -> Optional[str]:
    for key in ["ord_no", "odr_no", "ODNO", "odr_no", "ord_gno_brno", "odno"]:
        if key in output and output[key]:
            return str(output[key])
    return None


def place_order(
    api_client: ApiClient,
    account_number: str,
    product_code: str,
    symbol: str,
    side: str,
    price: int,
    quantity: int = 1,
) -> OrderResult:
    logger.info("Placing %s order for %s: qty=%s price=%s", side, symbol, quantity, price)

    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")

    tr_id = "VTTC0012U" if side == "buy" else "VTTC0011U"
    params = {
        "CANO": account_number,
        "ACNT_PRDT_CD": product_code,
        "PDNO": symbol,
        "ORD_DVSN": "00",
        "ORD_QTY": str(quantity),
        "ORD_UNPR": str(price),
        "EXCG_ID_DVSN_CD": "KRX",
        "SLL_TYPE": "",
        "CNDT_PRIC": "",
    }

    try:
        response = api_client.post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=tr_id,
            params=params,
        )
    except Exception as exc:
        logger.error("Order request failed: %s", exc)
        return OrderResult(
            side=side,
            symbol=symbol,
            quantity=quantity,
            price=price,
            success=False,
            raw={},
            message=str(exc),
        )

    output = _extract_output(response)
    order_reference = _extract_order_reference(output)
    message = output.get("msg1") or output.get("msg2") or response.get("msg1") or response.get("msg2")
    success = bool(output)
    logger.info(
        "%s order completed for %s (reference=%s, success=%s)",
        side,
        symbol,
        order_reference,
        success,
    )

    return OrderResult(
        side=side,
        symbol=symbol,
        quantity=quantity,
        price=price,
        success=success,
        raw=response,
        message=str(message) if message else None,
        order_reference=order_reference,
    )
