import logging
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Optional

from account import get_account_holdings, get_account_summary
from config import Settings
from market_data import get_current_price
from orders import OrderResult, place_order
from api_client import ApiClient

logger = logging.getLogger(__name__)


class AutoTrader:
    def __init__(self, settings: Settings, api_client: ApiClient) -> None:
        self.settings = settings
        self.api_client = api_client
        self.local_tz = ZoneInfo("Asia/Seoul")

    def run(self) -> None:
        logger.info("Starting Samsung Auto Trader")

        while True:
            now = self._now()
            if now.time() >= self.settings.trading_end:
                logger.info(
                    "Trading window ended at %s. Stopping trading loop.",
                    self.settings.trading_end.strftime("%H:%M"),
                )
                break

            if now.time() < self.settings.trading_start:
                sleep_seconds = self._seconds_until(self.settings.trading_start)
                logger.info(
                    "Waiting for trading window to open at %s. Sleeping %s seconds.",
                    self.settings.trading_start.strftime("%H:%M"),
                    sleep_seconds,
                )
                time.sleep(min(sleep_seconds, 60))
                continue

            logger.info(
                "Trading window open: %s - %s (local time %s)",
                self.settings.trading_start.strftime("%H:%M"),
                self.settings.trading_end.strftime("%H:%M"),
                now.strftime("%H:%M"),
            )

            self._trade_cycle()

            if datetime.now().time() >= self.settings.trading_end:
                continue

            logger.info("Sleeping %s seconds before next cycle", self.settings.polling_interval_seconds)
            time.sleep(self.settings.polling_interval_seconds)

    def _trade_cycle(self) -> None:
        symbol = self.settings.symbol
        current_price: Optional[int] = None

        try:
            current_price = get_current_price(self.api_client, symbol)
        except Exception as exc:
            logger.error("Unable to fetch current price: %s", exc)
            return

        try:
            summary_before = get_account_summary(
                self.api_client,
                self.settings.account_number,
                self.settings.product_code,
            )
            holdings_before = get_account_holdings(
                self.api_client,
                self.settings.account_number,
                self.settings.product_code,
                symbol,
            )
        except Exception as exc:
            logger.error("Unable to fetch account information: %s", exc)
            return

        logger.info("Pre-order summary: available_cash=%s", summary_before["available_cash"])
        logger.info("Pre-order holdings: quantity=%s average_price=%s",
                    holdings_before["quantity"], holdings_before["average_price"])

        buy_price = max(current_price + self.settings.buy_offset, 1)
        sell_price = current_price + self.settings.sell_offset

        buy_result = self._attempt_buy(buy_price, holdings_before, summary_before, symbol)
        sell_result = self._attempt_sell(sell_price, holdings_before, symbol)

        try:
            summary_after = get_account_summary(
                self.api_client,
                self.settings.account_number,
                self.settings.product_code,
            )
            holdings_after = get_account_holdings(
                self.api_client,
                self.settings.account_number,
                self.settings.product_code,
                symbol,
            )
        except Exception as exc:
            logger.error("Unable to fetch account information after orders: %s", exc)
            return

        logger.info("Post-order summary: available_cash=%s", summary_after["available_cash"])
        logger.info("Post-order holdings: quantity=%s average_price=%s",
                    holdings_after["quantity"], holdings_after["average_price"])

        self._report_execution(holdings_before, holdings_after, summary_before, summary_after,
                               buy_result, sell_result)

    def _attempt_buy(
        self,
        buy_price: int,
        holdings: dict,
        summary: dict,
        symbol: str,
    ) -> Optional[OrderResult]:
        available_cash = summary.get("available_cash", 0)
        if available_cash < buy_price:
            logger.info(
                "Skipping buy order: available cash %s is below buy price %s",
                available_cash,
                buy_price,
            )
            return None

        return place_order(
            self.api_client,
            self.settings.account_number,
            self.settings.product_code,
            symbol,
            side="buy",
            price=buy_price,
            quantity=1,
        )

    def _attempt_sell(self, sell_price: int, holdings: dict, symbol: str) -> Optional[OrderResult]:
        if holdings.get("quantity", 0) < 1:
            logger.info("Skipping sell order: no shares held for %s", symbol)
            return None

        return place_order(
            self.api_client,
            self.settings.account_number,
            self.settings.product_code,
            symbol,
            side="sell",
            price=sell_price,
            quantity=1,
        )

    def _report_execution(
        self,
        holdings_before: dict,
        holdings_after: dict,
        summary_before: dict,
        summary_after: dict,
        buy_result: Optional[OrderResult],
        sell_result: Optional[OrderResult],
    ) -> None:
        executed = False

        if buy_result and buy_result.success:
            executed = True
            logger.info("Buy request succeeded: %s", buy_result)
        elif buy_result is not None:
            logger.warning("Buy request failed or returned no output: %s", buy_result)

        if sell_result and sell_result.success:
            executed = True
            logger.info("Sell request succeeded: %s", sell_result)
        elif sell_result is not None:
            logger.warning("Sell request failed or returned no output: %s", sell_result)

        if holdings_after["quantity"] != holdings_before["quantity"]:
            executed = True
            logger.info(
                "Holdings changed from %s to %s shares",
                holdings_before["quantity"],
                holdings_after["quantity"],
            )

        if summary_after["available_cash"] != summary_before["available_cash"]:
            executed = True
            logger.info(
                "Available cash changed from %s to %s",
                summary_before["available_cash"],
                summary_after["available_cash"],
            )

        if not executed:
            logger.info("No execution detected in this cycle")

    @staticmethod
    def _seconds_until(self, target_time: dt_time) -> int:
        now = self._now()
        target = datetime.combine(now.date(), target_time, tzinfo=self.local_tz)
        if target <= now:
            target = target.replace(day=now.day + 1)
        return int((target - now).total_seconds())

    def _now(self) -> datetime:
        return datetime.now(self.local_tz)
