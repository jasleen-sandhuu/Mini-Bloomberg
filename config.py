import os
from dotenv import load_dotenv

load_dotenv()

KALSHI_KEY_ID      = os.getenv("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY = os.getenv("KALSHI_PRIVATE_KEY", "")
KALSHI_BASE_URL    = "https://api.elections.kalshi.com"

# Math defaults
MONTE_CARLO_RUNS = 10_000
CONFIDENCE_LEVEL = 0.95
PINN_EPOCHS      = 2000
PINN_LR          = 0.05
