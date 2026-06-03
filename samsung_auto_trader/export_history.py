import argparse
import logging

from auth import TokenManager
from api_client import ApiClient
from config import Settings
from market_data import export_historical_prices

logger = logging.getLogger(__name__)


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
        default="samsung_price_history.csv",
        help="Output CSV file path.",
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


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    settings = Settings.load()
    token_manager = TokenManager(settings)
    client = ApiClient(settings, token_manager)

    args = parse_args()
    path = export_historical_prices(
        client,
        symbol=args.symbol,
        output_path=args.output,
        period_div=args.period,
        org_adj_prc=args.adj,
        market_div=args.market,
    )

    logger.info("Saved historical prices to %s", path)


if __name__ == "__main__":
    main()
