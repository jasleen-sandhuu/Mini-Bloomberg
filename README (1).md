# Kalshi Terminal

A Bloomberg-style quantitative research terminal for prediction markets. Connects to Kalshi's live API and runs edge detection using Bayesian inference, Monte Carlo simulation, Kelly criterion, and a physics-informed neural network.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/kalshi-terminal.git
cd kalshi-terminal
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configure

Copy the example env file and add your Kalshi credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Key ID and private key from [Kalshi API Settings](https://kalshi.com).

To run in demo mode without an API key, leave `.env` unchanged.

## Run

```bash
python main.py
```

## Commands

| Stage | Command | Description |
|-------|---------|-------------|
| UX | `HELP` | Command reference |
| UX | `SEARCH <query>` | Search markets (live or local) |
| UX | `WATCHLIST` | Tracked markets |
| UX | `STATUS` | API connection info |
| Trading | `MARKETS` | List all markets |
| Trading | `MARKET <ticker>` | Market detail |
| Trading | `BOOK <ticker>` | Order book |
| Trading | `SPREAD <ticker>` | Bid/ask spread |
| Trading | `PRICEHIST <ticker>` | Price history chart |
| Trading | `NEWS` | Market news feed |
| Trading | `BALANCE` | Account balance (live) |
| Quant | `EV <ticker>` | Expected value |
| Quant | `KELLY <ticker>` | Kelly criterion sizing |
| Quant | `ARB` | Arbitrage scanner |
| Quant | `PORTFOLIO` | Portfolio P&L |
| Engineering | `SIM <ticker>` | Monte Carlo (10k runs) |
| Engineering | `VAR <ticker>` | Value at Risk |
| Engineering | `CORR` | Correlation matrix |
| Engineering | `BAYES <ticker>` | Bayesian posterior |
| Engineering | `KALMAN <ticker>` | Kalman filter |
| Neural Net | `PROB <ticker>` | Probability analysis |
| Neural Net | `EDGE <ticker>` | Edge detection |
| Neural Net | `RISK <ticker>` | Risk report |
| Neural Net | `MODEL <ticker>` | Physics-informed NN |

## Architecture

```
main.py                     Entry point
kalshi_terminal/
  config.py                 Credentials + settings
  api.py                    Kalshi API client (PSS signing)
  data.py                   Market data store + dummy data
  math_engine.py            Probability, Bayesian, Monte Carlo, risk
  neural_net.py             Physics-informed neural network
  display.py                Rich formatting + all show_ functions
  commands.py               Command dispatcher
```

## License

MIT
