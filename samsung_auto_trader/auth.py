import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from config import Settings


logger = logging.getLogger(__name__)


class TokenManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_path: Path = settings.token_cache_path
        self.token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            if self.cache_path.exists():
                with self.cache_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.token = payload.get("token")
                expires_at = payload.get("expires_at")
                if expires_at:
                    self.expires_at = datetime.fromisoformat(expires_at)
        except (ValueError, OSError) as exc:
            logger.warning("Token cache load failed: %s", exc)
            self.token = None
            self.expires_at = None

    def _save_cache(self, token: str, expires_at: datetime) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token": token,
            "expires_at": expires_at.isoformat(),
        }
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _token_is_valid(self) -> bool:
        if not self.token or not self.expires_at:
            return False

        now = datetime.utcnow()
        return now < self.expires_at

    def get_token(self) -> str:
        if self._token_is_valid():
            logger.info("Reusing cached token until %s UTC", self.expires_at)
            return self.token  # type: ignore

        logger.info("Cached token missing or expired, requesting new token")
        return self._request_token()

    def refresh_token(self) -> str:
        logger.info("Refreshing token now")
        return self._request_token()

    def _request_token(self) -> str:
        body = {
            "grant_type": "client_credentials",
            "appkey": self.settings.app_key,
            "appsecret": self.settings.app_secret,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "koreainvestment-samsung-auto-trader/1.0",
        }

        response = requests.post(
            self.settings.get_auth_url(),
            headers=headers,
            json=body,
            timeout=15,
        )

        if response.status_code != 200:
            logger.error(
                "Token request failed: %s %s",
                response.status_code,
                response.text,
            )
            raise RuntimeError("Unable to authenticate with Korea Investment API")

        data = response.json()
        token = self._extract_token(data)
        expires_at = self._extract_expiration(data)
        self.token = token
        self.expires_at = expires_at
        self._save_cache(token, expires_at)
        logger.info("Saved new token valid until %s UTC", expires_at)
        return token

    def _extract_token(self, data: Dict[str, Any]) -> str:
        token = data.get("access_token") or data.get("accessToken")
        if not token:
            raise ValueError("Authentication response did not include an access token")
        return token

    def _extract_expiration(self, data: Dict[str, Any]) -> datetime:
        expires_at = data.get("access_token_token_expired")
        if expires_at:
            try:
                return datetime.fromisoformat(expires_at)
            except ValueError:
                pass

        expires_in = data.get("expires_in")
        if isinstance(expires_in, (int, float)):
            return datetime.utcnow() + timedelta(seconds=int(expires_in))

        # Default to same-day reuse with conservative expiry, as required.
        tomorrow = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return tomorrow
