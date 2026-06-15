"""
Market data store.
Holds local market data. Syncs from Kalshi API when connected.
Falls back to dummy data in demo mode.
"""

from kalshi_terminal.api import KalshiAPI

api = KalshiAPI()

# ── Local data store ────────────────────────────────────────

markets = {
    "BTC": {
        "name": "Bitcoin above $100k?", "price": 62, "volume": 120000,
        "open_interest": 10000, "status": "Open",
        "history": [40, 45, 52, 58, 62],
        "bids": [(60, 100), (59, 250), (58, 400)],
        "asks": [(62, 120), (63, 300), (64, 500)],
        "category": "Crypto",
    },
    "FED": {
        "name": "Fed rate cut in 2026?", "price": 41, "volume": 80000,
        "open_interest": 7000, "status": "Open",
        "history": [35, 38, 39, 42, 41],
        "bids": [(40, 150), (39, 300), (38, 600)],
        "asks": [(42, 200), (43, 350), (44, 700)],
        "category": "Macro",
    },
    "ETH": {
        "name": "Ethereum above $5k?", "price": 48, "volume": 95000,
        "open_interest": 8200, "status": "Open",
        "history": [30, 35, 40, 45, 48],
        "bids": [(47, 120), (46, 280), (45, 500)],
        "asks": [(49, 150), (50, 320), (51, 600)],
        "category": "Crypto",
    },
    "REC": {
        "name": "US recession in 2026?", "price": 24, "volume": 60000,
        "open_interest": 5500, "status": "Open",
        "history": [30, 28, 26, 25, 24],
        "bids": [(23, 200), (22, 400), (21, 700)],
        "asks": [(25, 180), (26, 350), (27, 600)],
        "category": "Macro",
    },
}

watchlist = ["BTC", "FED", "ETH"]

portfolio = {
    "BTC": {"shares": 100, "side": "YES", "entry_price": 55},
    "FED": {"shares": 200, "side": "NO",  "entry_price": 50},
}

news = [
    ("BTC", "Bitcoin ETF inflows hit record high"),
    ("FED", "Powell signals data-dependent rate path"),
    ("ETH", "Ethereum upgrade scheduled for Q2"),
    ("REC", "Jobs report shows resilient labour market"),
]

command_history = []


# ── API sync helpers ────────────────────────────────────────

def _clean_raw(raw):
    """Convert raw Kalshi API response into local market format."""
    bid = raw.get("yes_bid", 0) or 0
    ask = raw.get("yes_ask", 100) or 100
    mid = int((bid + ask) / 2)
    return {
        "name": raw.get("title", "")[:60],
        "price": mid,
        "volume": raw.get("volume", 0) or 0,
        "open_interest": raw.get("open_interest", 0) or 0,
        "status": raw.get("status", "Open"),
        "history": [mid],
        "bids": [(bid, 100)],
        "asks": [(ask, 100)],
        "category": raw.get("category", "Other"),
    }


def sync_search(query, limit=10):
    """Search Kalshi live, store results locally, return tickers."""
    raw_markets, err = api.search_markets(query=query, limit=limit)
    if err:
        return [], err
    tickers = []
    for raw in raw_markets:
        ticker = raw.get("ticker", "")
        markets[ticker] = _clean_raw(raw)
        tickers.append(ticker)
    return tickers, None


def sync_market(ticker):
    """Fetch one market from Kalshi, update local store."""
    raw, err = api.get_market(ticker)
    if err:
        return False, err
    cleaned = _clean_raw(raw)
    if ticker in markets and "history" in markets[ticker]:
        cleaned["history"] = markets[ticker]["history"] + [cleaned["price"]]
    markets[ticker] = cleaned
    return True, None


def ensure_market(ticker):
    """Make sure we have data for this ticker. Refresh if live."""
    if api.connected:
        if ticker not in markets:
            ok, _ = sync_market(ticker)
            return ok
        sync_market(ticker)
        return True
    return ticker in markets
