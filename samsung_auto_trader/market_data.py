import logging
from typing import Any, Dict

from api_client import ApiClient

logger = logging.getLogger(__name__)

PRICE_KEYS = ["stck_prpr", "STCK_PRPR", "prpr", "PRPR", "price"]


def _extract_row(data: Dict[str, Any]) -> Dict[str, Any]:
    if "output" in data:
        output = data["output"]
    elif "output1" in data:
        output = data["output1"]
    else:
        output = data

    if isinstance(output, list) and output:
        return output[0]
    if isinstance(output, dict):
        return output
    return {}


def _find_price(row: Dict[str, Any]) -> int:
    for key in PRICE_KEYS:
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(float(value))
        except (ValueError, TypeError):
            continue
    raise ValueError("Unable to parse current price from API response")


def get_current_price(api_client: ApiClient, symbol: str) -> int:
    logger.info("Fetching current market price for %s", symbol)
    response = api_client.get(
        "/uapi/domestic-stock/v1/quotations/inquire-price",
        tr_id="FHKST01010100",
        params={
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
        },
    )

    row = _extract_row(response)
    price = _find_price(row)
    logger.info("Current price for %s is %s KRW", symbol, price)
    return price
