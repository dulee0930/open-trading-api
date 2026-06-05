import json
import logging
import os
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional, Tuple

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

    def _signal_file_path(self) -> Path:
        env_path = os.environ.get("SIGNAL_JSON_PATH")
        if env_path:
            return Path(env_path).expanduser().resolve()
        return Path(__file__).resolve().parent / "latest_trading_signal.json"

    def _load_latest_signal(self) -> Optional[Dict[str, Any]]:
        signal_path = self._signal_file_path()
        if not signal_path.exists():
            logger.info("Signal file not found: %s", signal_path)
            return None

        try:
            with signal_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Unable to load signal file %s: %s", signal_path, exc)
            return None

    def _validate_signal(self, signal: Dict[str, Any]) -> Tuple[bool, str]:
        if signal.get("symbol") != self.settings.symbol:
            return False, f"HOLD: symbol mismatch (expected {self.settings.symbol})"

        as_of_date = signal.get("as_of_date")
        if not isinstance(as_of_date, str):
            return False, "HOLD: as_of_date missing or invalid"

        try:
            signal_date = datetime.fromisoformat(as_of_date).date()
        except ValueError:
            return False, f"HOLD: invalid as_of_date format {as_of_date}"

        today = self._now().date()
        if signal_date > today:
            return False, f"HOLD: signal date {signal_date} is in the future"

        prediction = signal.get("prediction")
        if not isinstance(prediction, dict):
            return False, "HOLD: prediction missing or invalid"

        trading_signal = prediction.get("trading_signal")
        if trading_signal not in {"BUY", "HOLD", "SELL"}:
            return False, f"HOLD: invalid trading_signal {trading_signal}"

        if prediction.get("action_blocked_by_confidence", False):
            return False, "HOLD: blocked by confidence guard"

        confidence = prediction.get("confidence")
        if not isinstance(confidence, (int, float)):
            return False, "HOLD: confidence missing or invalid"
        if confidence < 0.45:
            return False, f"HOLD: confidence {confidence:.3f} < 0.45"

        normalized_entropy = prediction.get("normalized_entropy")
        if not isinstance(normalized_entropy, (int, float)):
            return False, "HOLD: normalized_entropy missing or invalid"
        if normalized_entropy > 0.95:
            return False, f"HOLD: normalized_entropy {normalized_entropy:.3f} > 0.95"

        training_summary = signal.get("training_summary", {})
        bal_acc = training_summary.get("best_validation_balanced_accuracy")
        if not isinstance(bal_acc, (int, float)):
            return False, "HOLD: best_validation_balanced_accuracy missing or invalid"
        if bal_acc < 0.36:
            return False, f"HOLD: balanced_accuracy {bal_acc:.3f} < 0.36"

        return True, ""

    def _build_buy_order(self, available_cash: int, holding_qty: int, current_price: int, total_equity: float) -> Optional[OrderResult]:
        max_order_cash = min(
            available_cash * 0.10,
            max(0.0, total_equity * 0.30 - holding_qty * current_price),
        )
        if max_order_cash < current_price:
            logger.info("HOLD: insufficient cash or position limit for one share")
            return None

        quantity = int(max_order_cash // current_price)
        if quantity <= 0:
            logger.info("HOLD: insufficient planned order cash for one share")
            return None

        return place_order(
            self.api_client,
            self.settings.account_number,
            self.settings.product_code,
            self.settings.symbol,
            side="buy",
            price=current_price,
            quantity=quantity,
        )

    def _build_sell_order(self, holding_qty: int, current_price: int) -> Optional[OrderResult]:
        if holding_qty <= 0:
            logger.info("HOLD: no position to sell")
            return None

        return place_order(
            self.api_client,
            self.settings.account_number,
            self.settings.product_code,
            self.settings.symbol,
            side="sell",
            price=current_price,
            quantity=holding_qty,
        )

    def _signal_order(self, signal: Dict[str, Any], summary: Dict[str, Any], holdings: Dict[str, Any], current_price: int) -> Optional[OrderResult]:
        valid, message = self._validate_signal(signal)
        if not valid:
            logger.info(message)
            return None

        trading_signal = signal["prediction"]["trading_signal"]
        available_cash = int(summary.get("available_cash", 0))
        holding_qty = int(holdings.get("quantity", 0))
        total_equity = available_cash + holding_qty * current_price

        if trading_signal == "HOLD":
            logger.info("HOLD: model signal is HOLD")
            return None

        if trading_signal == "BUY":
            return self._build_buy_order(available_cash, holding_qty, current_price, total_equity)

        if trading_signal == "SELL":
            return self._build_sell_order(holding_qty, current_price)

        logger.info("HOLD: unknown trading signal %s", trading_signal)
        return None

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

        signal = self._load_latest_signal()
        if signal is None:
            logger.info("Skipping trade cycle because latest signal is unavailable")
            return

        buy_result = self._signal_order(signal, summary_before, holdings_before, current_price)
        sell_result = None

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
