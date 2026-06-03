import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

from api_client import ApiClient

logger = logging.getLogger(__name__)

PRICE_KEYS = ["stck_prpr", "STCK_PRPR", "prpr", "PRPR", "price"]

HISTORICAL_FIELD_DESCRIPTIONS = {
    "acml_prtt_rate": "누적수익률",
    "acml_vol": "누적거래량",
    "flng_cls_code": "외국인구분코드",
    "frgn_ntby_qty": "외국인순매수수량",
    "hts_frgn_ehrt": "HTS외국인보유율",
    "prdy_ctrt": "전일대비율",
    "prdy_vrss": "전일대비",
    "prdy_vrss_sign": "전일대비부호",
    "prdy_vrss_vol_rate": "전일대비거래량증감율",
    "stck_bsop_date": "영업일자",
    "stck_clpr": "종가",
    "stck_hgpr": "고가",
    "stck_lwpr": "저가",
    "stck_oprc": "시가",
}


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


def get_historical_prices(
    api_client: ApiClient,
    symbol: str,
    period_div: str = "D",
    org_adj_prc: str = "1",
    market_div: str = "J",
) -> List[Dict[str, Any]]:
    logger.info(
        "Fetching historical prices for %s (period=%s, adj=%s)",
        symbol,
        period_div,
        org_adj_prc,
    )

    response = api_client.get(
        "/uapi/domestic-stock/v1/quotations/inquire-daily-price",
        tr_id="FHKST01010400",
        params={
            "FID_COND_MRKT_DIV_CODE": market_div,
            "FID_INPUT_ISCD": symbol,
            "FID_PERIOD_DIV_CODE": period_div,
            "FID_ORG_ADJ_PRC": org_adj_prc,
        },
    )

    output = response.get("output") or response.get("output1") or response.get("output2") or []
    if isinstance(output, dict):
        output = [output]
    if not isinstance(output, list):
        raise ValueError("Unexpected historical price response format")

    logger.info("Retrieved %s historical price rows for %s", len(output), symbol)
    return output


def _write_field_descriptions(file_path: Path) -> Path:
    schema_path = file_path.with_name(f"{file_path.stem}.schema.csv")
    logger.info("Writing field description schema to %s", schema_path)

    with schema_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["field", "description"])
        for field, description in sorted(HISTORICAL_FIELD_DESCRIPTIONS.items()):
            writer.writerow([field, description])

    logger.info("Schema export complete: %s", schema_path)
    return schema_path


def export_historical_prices(
    api_client: ApiClient,
    symbol: str,
    output_path: str,
    period_div: str = "D",
    org_adj_prc: str = "1",
    market_div: str = "J",
    include_schema: bool = True,
) -> Path:
    rows = get_historical_prices(
        api_client,
        symbol,
        period_div=period_div,
        org_adj_prc=org_adj_prc,
        market_div=market_div,
    )

    file_path = Path(output_path)
    if not rows:
        raise ValueError("No historical price data available to export")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    logger.info("Exporting historical prices to %s", file_path)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if include_schema:
        _write_field_descriptions(file_path)

    logger.info("Export complete: %s", file_path)
    return file_path
