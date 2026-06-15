"""
Kalshi API client.
Handles authentication (PSS signing) and all API calls.
"""

import base64
import requests
from datetime import datetime
from kalshi_terminal.config import KALSHI_BASE_URL, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY


class KalshiAPI:
    def __init__(self):
        self.base_url    = KALSHI_BASE_URL
        self.key_id      = KALSHI_KEY_ID
        self.private_key = None
        self.connected   = False
        self._load_key()

    def _load_key(self):
        if not self.key_id or "your-key" in self.key_id:
            return
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            key_str = KALSHI_PRIVATE_KEY.replace("\\n", "\n")
            self.private_key = serialization.load_pem_private_key(
                key_str.encode("utf-8"),
                password=None,
                backend=default_backend(),
            )
            self.connected = True
        except Exception as e:
            print(f"  API key load failed: {e}")

    def _sign(self, method, path):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts  = str(int(datetime.now().timestamp() * 1000))
        msg = (ts + method + path).encode("utf-8")
        sig = self.private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY":       self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
        }

    def _get(self, path, params=None):
        if not self.connected:
            return None, "Not connected"
        try:
            r = requests.get(
                self.base_url + path,
                headers=self._sign("GET", path),
                params=params,
                timeout=8,
            )
            if r.status_code == 200:
                return r.json(), None
            return None, f"API {r.status_code}: {r.text[:120]}"
        except Exception as e:
            return None, str(e)

    # ── Public methods ──────────────────────────────────────

    def search_markets(self, query="", limit=10):
        params = {"limit": limit, "status": "open"}
        if query:
            params["search"] = query
        data, err = self._get("/trade-api/v2/markets", params)
        if err:
            return [], err
        return data.get("markets", []), None

    def get_market(self, ticker):
        data, err = self._get(f"/trade-api/v2/markets/{ticker}")
        if err:
            return None, err
        return data.get("market", {}), None

    def get_balance(self):
        data, err = self._get("/trade-api/v2/portfolio/balance")
        if err:
            return None, err
        return data, None

    def get_positions(self):
        data, err = self._get("/trade-api/v2/portfolio/positions", {"limit": 20})
        if err:
            return [], err
        return data.get("market_positions", []), None
