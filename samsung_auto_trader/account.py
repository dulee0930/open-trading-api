import logging
from typing import Any, Dict, List, Optional

from api_client import ApiClient

logger = logging.getLogger(__name__)

HOLDING_KEYS = ["pdno", "PDNO"]
QUANTITY_KEYS = ["hldg_qty", "hldt_qty", "qty", "HLDG_QTY"]
PRICE_KEYS = ["pchs_avg_pric", "prpr", "stck_prpr", "avg_prc", "avg_prc"]
CASH_KEYS = [
    "ord_psbl_cash",
    "ord_psbl_cash_amt",
    "dpsl_amt",
    "avail_cash",
    "cash_avail",
    "dnca_tot_amt",
    "nass_amt",
    "tot_evlu_amt",
]


def _find_numeric(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _find_first_key(item: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _find_holding_row(data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    output = data.get("output1") or data.get("output") or []
    if isinstance(output, dict):
        output = [output]

    for row in output:
        code = _find_first_key(row, HOLDING_KEYS)
        if not code:
            continue
        if str(code).strip() == symbol:
            return row
    return None


def get_account_holdings(api_client: ApiClient, account_number: str, product_code: str, symbol: str) -> Dict[str, Any]:
    logger.info("Checking holdings for symbol %s", symbol)
    response = api_client.get(
        "/uapi/domestic-stock/v1/trading/inquire-balance",
        tr_id="VTTC8434R",
        params={
            "CANO": account_number,
            "ACNT_PRDT_CD": product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
    )

    position = _find_holding_row(response, symbol)
    if not position:
        logger.info("No holding found for %s", symbol)
        return {
            "symbol": symbol,
            "quantity": 0,
            "average_price": 0,
            "current_price": 0,
            "raw": response,
        }

    quantity = _find_numeric(_find_first_key(position, QUANTITY_KEYS))
    average_price = _find_numeric(_find_first_key(position, [PRICE_KEYS[0]]))
    current_price = _find_numeric(_find_first_key(position, [PRICE_KEYS[1]]))

    return {
        "symbol": symbol,
        "quantity": quantity,
        "average_price": average_price,
        "current_price": current_price,
        "raw": position,
    }


def get_account_summary(api_client: ApiClient, account_number: str, product_code: str) -> Dict[str, Any]:
    logger.info("Fetching account summary for %s", account_number)
    response = api_client.get(
        "/uapi/domestic-stock/v1/trading/inquire-balance",
        tr_id="VTTC8434R",
        params={
            "CANO": account_number,
            "ACNT_PRDT_CD": product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
    )

    output2 = response.get("output2") or response.get("output") or {}
    if isinstance(output2, list):
        output2 = output2[0] if output2 else {}

    available_cash = _find_numeric(_find_first_key(output2, CASH_KEYS))
    return {
        "available_cash": available_cash,
        "raw": output2,
    }
