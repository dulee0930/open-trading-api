import json
import logging
from typing import Any, Dict, Optional

import requests

from auth import TokenManager
from config import Settings

logger = logging.getLogger(__name__)


class ApiClient:
    def __init__(self, settings: Settings, token_manager: TokenManager) -> None:
        self.settings = settings
        self.token_manager = token_manager
        self.token = token_manager.get_token()

    def _build_headers(self, tr_id: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "koreainvestment-samsung-auto-trader/1.0",
            "authorization": f"Bearer {self.token}",
            "appkey": self.settings.app_key,
            "appsecret": self.settings.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
            "tr_cont": "",
        }

    def _send_request(
        self,
        method: str,
        api_path: str,
        tr_id: str,
        params: Dict[str, Any],
        retries: int = 2,
    ) -> Dict[str, Any]:
        url = f"{self.settings.api_domain}{api_path}"

        for attempt in range(1, retries + 1):
            try:
                headers = self._build_headers(tr_id)
                if method == "post":
                    response = requests.post(url, headers=headers, json=params, timeout=15)
                else:
                    response = requests.get(url, headers=headers, params=params, timeout=15)

                if response.status_code == 401 and attempt == 1:
                    logger.warning("Received 401 from API, refreshing token and retrying")
                    self.token = self.token_manager.refresh_token()
                    continue

                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                logger.warning(
                    "API request attempt %s failed for %s: %s",
                    attempt,
                    api_path,
                    exc,
                )
                if attempt == retries:
                    logger.error("API request permanently failed for %s", api_path)
                    raise

        raise RuntimeError("API request failed after retries")

    def get(self, api_path: str, tr_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._send_request("get", api_path, tr_id, params)

    def post(self, api_path: str, tr_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._send_request("post", api_path, tr_id, params)
