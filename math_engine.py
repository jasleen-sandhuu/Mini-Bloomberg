"""
Math engine.
Probability, Bayesian inference, Monte Carlo, Kelly, VaR, Kalman.
All functions take a market dict, not a ticker — keeps them testable.
"""

import numpy as np
from kalshi_terminal.config import MONTE_CARLO_RUNS, CONFIDENCE_LEVEL


# ── Probability ─────────────────────────────────────────────

def implied_prob(market):
    return market["price"] / 100


def prob_analysis(market):
    p   = implied_prob(market)
    vol = max(market["volume"], 1)
    var = p * (1 - p)
    std = var ** 0.5
    margin = 1.96 * (var / vol) ** 0.5
    return {
        "prob": p, "expect": p, "var": var, "std": std,
        "ci_low": max(0, p - margin), "ci_high": min(1, p + margin),
    }


# ── Bayesian ────────────────────────────────────────────────

def bayesian_update(market):
    p   = implied_prob(market)
    vol = max(market["volume"], 1)
    yes = int(p * vol)
    no  = vol - yes
    a   = 1 + yes
    b   = 1 + no
    mean = a / (a + b)
    std  = ((a * b) / ((a + b) ** 2 * (a + b + 1))) ** 0.5
    return {
        "prior": 0.5, "posterior": mean, "std": std,
        "ci_low": max(0, mean - 1.96 * std),
        "ci_high": min(1, mean + 1.96 * std),
        "shift": mean - p,
    }


# ── Monte Carlo ─────────────────────────────────────────────

def monte_carlo(market, n=MONTE_CARLO_RUNS):
    p = implied_prob(market)
    a = max(p * 20, 0.1)
    b = max((1 - p) * 20, 0.1)
    samples = np.random.beta(a, b, n)
    return {
        "n": n,
        "mean":   float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std":    float(np.std(samples)),
        "p5":     float(np.percentile(samples, 5)),
        "p95":    float(np.percentile(samples, 95)),
        "var":    float(np.percentile(samples, 5)),
        "samples": samples,
    }


# ── Edge + Kelly ────────────────────────────────────────────

def edge_calc(market):
    mkt   = implied_prob(market)
    model = bayesian_update(market)["posterior"]
    edge  = model - mkt
    kelly = max(0, model - (1 - model))
    abs_e = abs(edge)
    risk  = ("MINIMAL" if abs_e < 0.03 else
             "LOW" if abs_e < 0.07 else
             "MEDIUM" if abs_e < 0.12 else "HIGH")
    return {"market": mkt, "model": model, "edge": edge, "kelly": kelly, "risk": risk}


def expected_value(market):
    p   = implied_prob(market)
    ask = market["asks"][0][0] / 100
    win = (1 - ask) * p
    loss = -ask * (1 - p)
    return {"prob": p, "cost": ask, "ev": win + loss,
            "win_payoff": 1 - ask, "loss_payoff": -ask}


def kelly_criterion(market):
    p   = bayesian_update(market)["posterior"]
    ask = market["asks"][0][0] / 100
    b   = (1 - ask) / ask if ask > 0 else 0
    f   = (p * b - (1 - p)) / b if b > 0 else 0
    return {"model_prob": p, "kelly_fraction": max(0, f),
            "half_kelly": max(0, f / 2), "quarter_kelly": max(0, f / 4)}


# ── VaR ─────────────────────────────────────────────────────

def value_at_risk(market, confidence=CONFIDENCE_LEVEL):
    sim     = monte_carlo(market)
    samples = sim["samples"]
    var_pct = float(np.percentile(samples, (1 - confidence) * 100))
    cvar    = float(np.mean(samples[samples <= var_pct]))
    return {"confidence": confidence, "var": var_pct, "cvar": cvar,
            "mean": sim["mean"], "std": sim["std"]}


# ── Correlation ─────────────────────────────────────────────

def correlation_matrix(markets_dict):
    tickers = [t for t in markets_dict if len(markets_dict[t]["history"]) >= 2]
    if len(tickers) < 2:
        return tickers, np.eye(len(tickers))
    max_len = min(len(markets_dict[t]["history"]) for t in tickers)
    data = np.array([markets_dict[t]["history"][-max_len:] for t in tickers])
    return tickers, np.corrcoef(data)


# ── Kalman Filter ───────────────────────────────────────────

class KalmanFilter:
    def __init__(self, init_estimate=0.5, process_var=0.001, measurement_var=0.01):
        self.x = init_estimate
        self.P = 1.0
        self.Q = process_var
        self.R = measurement_var

    def update(self, measurement):
        self.P += self.Q
        K       = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P  = (1 - K) * self.P
        return self.x, self.P


def kalman_track(market):
    kf = KalmanFilter(init_estimate=market["history"][0] / 100)
    return [(p / 100, *kf.update(p / 100)) for p in market["history"]]


# ── Arbitrage ───────────────────────────────────────────────

def arbitrage_check(markets_dict):
    opps = []
    for ticker, d in markets_dict.items():
        bid = d["bids"][0][0]
        ask = d["asks"][0][0]
        s   = bid + (100 - ask)
        if s > 100:
            opps.append((ticker, s - 100))
    return opps


# ── Portfolio ───────────────────────────────────────────────

def portfolio_exposure(portfolio, markets_dict):
    rows = []
    total_cost = total_value = 0
    for ticker, pos in portfolio.items():
        if ticker not in markets_dict:
            continue
        current = markets_dict[ticker]["price"]
        entry   = pos["entry_price"]
        shares  = pos["shares"]
        if pos["side"] == "YES":
            cost  = entry * shares / 100
            value = current * shares / 100
        else:
            cost  = (100 - entry) * shares / 100
            value = (100 - current) * shares / 100
        pnl = value - cost
        total_cost  += cost
        total_value += value
        rows.append({"ticker": ticker, "side": pos["side"], "shares": shares,
                     "entry": entry, "current": current, "cost": cost,
                     "value": value, "pnl": pnl})
    return rows, total_cost, total_value
