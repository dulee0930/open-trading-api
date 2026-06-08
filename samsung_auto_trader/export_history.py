import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from auth import TokenManager
from api_client import ApiClient
from config import Settings
from market_data import get_historical_prices

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "Samsung_Daily_Data_yfinance.csv"
DATE_FIELD = "stck_bsop_date"
DATE_FORMAT = "%Y-%m-%d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Samsung Electronics historical price data using Korea Investment REST API."
    )
    parser.add_argument(
        "--symbol",
        default="005930",
        help="Stock symbol to export (default: 005930).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--period",
        default="D",
        choices=["D", "W", "M"],
        help="Period granularity (D=day, W=week, M=month).",
    )
    parser.add_argument(
        "--adj",
        default="1",
        choices=["0", "1"],
        help="Adjusted price flag (0=unadjusted, 1=adjusted).",
    )
    parser.add_argument(
        "--market",
        default="J",
        choices=["J", "NX", "UN"],
        help="Market division code (J=KRX, NX=NXT, UN=Unified).",
    )
    return parser.parse_args()


def _parse_date(value: str) -> datetime:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {value}")


def _normalize_date(value: str) -> str:
    return _parse_date(value).strftime(DATE_FORMAT)


def _load_existing_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


def _combine_fieldnames(existing: List[str], fetched: List[str]) -> List[str]:
    fieldnames = list(existing)
    for field in fetched:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def _merge_rows(
    existing_rows: List[Dict[str, str]],
    fetched_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for row in existing_rows:
        date_key = row.get(DATE_FIELD)
        if not date_key:
            continue
        normalized = _normalize_date(date_key)
        row[DATE_FIELD] = normalized
        merged[normalized] = row

    for row in fetched_rows:
        date_key = row.get(DATE_FIELD)
        if not date_key:
            continue
        normalized = _normalize_date(date_key)
        row[DATE_FIELD] = normalized
        merged[normalized] = row

    sorted_rows = sorted(
        merged.values(),
        key=lambda row: _parse_date(row[DATE_FIELD]),
    )
    return sorted_rows


def _write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_historical_prices(
    client: ApiClient,
    symbol: str = "005930",
    output: str | Path = DEFAULT_OUTPUT,
    period: str = "D",
    adj: str = "1",
    market: str = "J",
) -> Path:
    output_path = Path(output)

    fetched_rows = get_historical_prices(
        client,
        symbol=symbol,
        period_div=period,
        org_adj_prc=adj,
        market_div=market,
    )

    existing_rows = _load_existing_rows(output_path)
    fetched_dates = {row[DATE_FIELD] for row in fetched_rows if row.get(DATE_FIELD)}
    existing_dates = {row[DATE_FIELD] for row in existing_rows if row.get(DATE_FIELD)}
    new_dates = sorted(fetched_dates - existing_dates, key=lambda d: _parse_date(d))

    merged_rows = _merge_rows(existing_rows, fetched_rows)

    if existing_rows:
        fieldnames = _combine_fieldnames(
            list(existing_rows[0].keys()),
            [field for row in fetched_rows for field in row.keys()],
        )
    else:
        fieldnames = sorted({field for row in fetched_rows for field in row.keys()})

    _write_rows(output_path, merged_rows, fieldnames)

    if new_dates:
        logger.info("Appended %d new historical rows to %s", len(new_dates), output_path)
    else:
        logger.info("No new historical rows were available to append to %s", output_path)

    logger.info("Saved historical prices to %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    settings = Settings.load()
    token_manager = TokenManager(settings)
    client = ApiClient(settings, token_manager)

    args = parse_args()
    output_path = Path(args.output)

    export_historical_prices(
        client=client,
        symbol=args.symbol,
        output=output_path,
        period=args.period,
        adj=args.adj,
        market=args.market,
    )


if __name__ == "__main__":
    main()
