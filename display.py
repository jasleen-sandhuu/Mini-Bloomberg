"""
Display layer.
All Rich console output — panels, tables, formatting.
No business logic here, just presentation.
"""

import os
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kalshi_terminal.data import (
    api, markets, watchlist, portfolio, news, command_history,
    sync_search, ensure_market,
)
from kalshi_terminal.math_engine import (
    implied_prob, prob_analysis, bayesian_update, monte_carlo,
    edge_calc, expected_value, kelly_criterion, value_at_risk,
    correlation_matrix, kalman_track, arbitrage_check, portfolio_exposure,
)
from kalshi_terminal.neural_net import build_pinn

console = Console()
PINN = build_pinn(markets)


def _cp(p):
    """Color code for a probability value."""
    if p >= 0.6: return "green"
    if p <= 0.4: return "red"
    return "yellow"


def _mode():
    return "[green]LIVE[/green]" if api.connected else "[yellow]DEMO[/yellow]"


# ── Stage 1 — UX ───────────────────────────────────────────

def show_header():
    os.system("clear" if os.name != "nt" else "cls")
    h = Text()
    h.append("  KALSHI TERM  ", style="bold black on yellow")
    h.append("  Quantitative Prediction Market Research Terminal", style="dim")
    console.print(h)
    console.print("─" * 80, style="dim")


def show_status_bar():
    parts = []
    for t in watchlist[:4]:
        if t in markets:
            p = markets[t]["price"]
            c = "green" if p >= 50 else "red"
            parts.append(f"[dim]{t}[/dim] [{c}]{p}¢[/{c}]")
    ts = datetime.now().strftime("%H:%M:%S")
    console.print("─" * 80, style="dim")
    console.print(f"[dim]│[/dim] {'  │  '.join(parts)}  [dim]│[/dim]  [yellow]{ts}[/yellow]  [dim]│[/dim]  {_mode()}  [dim]│[/dim]")


def show_help():
    console.print(Panel.fit(
        "[bold yellow]STAGE 1 — Terminal UX[/bold yellow]\n"
        "  HELP                       Show this menu\n"
        "  SEARCH <query>             Search markets\n"
        "  WATCHLIST                  Tracked markets\n"
        "  HISTORY                    Command history\n"
        "  STATUS                     Connection info\n"
        "  CLEAR                      Clear screen\n\n"
        "[bold yellow]STAGE 2 — Trading[/bold yellow]\n"
        "  MARKETS                    List all markets\n"
        "  MARKET <ticker>            Market detail\n"
        "  BOOK <ticker>              Order book\n"
        "  SPREAD <ticker>            Bid/ask spread\n"
        "  PRICEHIST <ticker>         Price history\n"
        "  NEWS                       News feed\n"
        "  BALANCE                    Account balance (live)\n\n"
        "[bold yellow]STAGE 3 — Quant Tools[/bold yellow]\n"
        "  EV <ticker>                Expected value\n"
        "  KELLY <ticker>             Kelly criterion\n"
        "  ARB                        Arbitrage scan\n"
        "  PORTFOLIO                  Portfolio P&L\n\n"
        "[bold yellow]STAGE 4 — Financial Engineering[/bold yellow]\n"
        "  SIM <ticker>               Monte Carlo\n"
        "  VAR <ticker>               Value at Risk\n"
        "  CORR                       Correlation matrix\n"
        "  BAYES <ticker>             Bayesian update\n"
        "  KALMAN <ticker>            Kalman filter\n\n"
        "[bold yellow]EXTRAS[/bold yellow]\n"
        "  PROB <ticker>              Probability analysis\n"
        "  EDGE <ticker>              Edge detection\n"
        "  RISK <ticker>              Risk report\n"
        "  MODEL <ticker>             Neural network\n"
        "  EXIT                       Quit",
        title="[bold cyan]COMMAND REFERENCE[/bold cyan]", border_style="cyan"))


def show_status():
    if api.connected:
        console.print(Panel(
            f"[green]✓ Connected to Kalshi API[/green]\n"
            f"  Key ID : {api.key_id[:12]}...\n"
            f"  Mode   : [green]LIVE[/green]",
            title="Status", border_style="green"))
    else:
        console.print(Panel(
            "[yellow]○ DEMO mode — using dummy data[/yellow]\n"
            "[dim]Set credentials in .env to connect live.[/dim]",
            title="Status", border_style="yellow"))


def show_search(query):
    if api.connected:
        console.print(f"[dim]  Searching Kalshi for: {query}...[/dim]")
        tickers, err = sync_search(query, limit=10)
        if err or not tickers:
            console.print(f"[red]  No results for: {query}[/red]"); return
        src = tickers
    else:
        q = query.lower()
        src = [t for t in markets
               if q in t.lower() or q in markets[t]["name"].lower()
               or q in markets[t].get("category", "").lower()]
    if not src:
        console.print(f"[red]  No results for: {query}[/red]"); return
    t = Table(title=f"Search — '{query}'", border_style="cyan")
    for c in ["Ticker", "Market", "Price", "Volume"]:
        t.add_column(c)
    for tk in src:
        d = markets[tk]
        t.add_row(tk, d["name"][:45], f"[{_cp(d['price']/100)}]{d['price']}¢[/{_cp(d['price']/100)}]", f"{d['volume']:,}")
    console.print(t)


def show_command_history():
    if not command_history:
        console.print("[dim]No commands yet.[/dim]"); return
    t = Table(title="History", border_style="dim")
    t.add_column("#"); t.add_column("Command")
    for i, c in enumerate(command_history[-15:], 1):
        t.add_row(str(i), c)
    console.print(t)


# ── Stage 2 — Trading ──────────────────────────────────────

def show_markets():
    if api.connected:
        console.print("[dim]  Fetching live markets...[/dim]")
        sync_search("", limit=10)
    t = Table(title="All Markets", border_style="yellow")
    for c in ["Ticker", "Market", "Price", "Volume", "Status"]:
        t.add_column(c)
    for tk, d in markets.items():
        t.add_row(tk, d["name"][:45], f"[{_cp(d['price']/100)}]{d['price']}¢[/{_cp(d['price']/100)}]",
                  f"{d['volume']:,}", d["status"])
    console.print(t)


def show_market(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    d = markets[ticker]
    console.print(Panel(
        f"[dim]Market[/dim]  : {d['name']}\n"
        f"[dim]Price[/dim]   : [{_cp(d['price']/100)}]{d['price']}¢[/{_cp(d['price']/100)}]\n"
        f"[dim]Volume[/dim]  : {d['volume']:,}\n"
        f"[dim]Status[/dim]  : [green]{d['status']}[/green]",
        title=ticker, border_style="yellow"))


def show_orderbook(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    d = markets[ticker]
    t = Table(title=f"{ticker} Order Book", border_style="cyan")
    for c in ["Bid", "Size", "Ask", "Size"]:
        t.add_column(c)
    for bid, ask in zip(d["bids"], d["asks"]):
        t.add_row(f"[green]{bid[0]}¢[/green]", str(bid[1]),
                  f"[red]{ask[0]}¢[/red]", str(ask[1]))
    console.print(t)


def show_spread(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    bb, ba = markets[ticker]["bids"][0][0], markets[ticker]["asks"][0][0]
    console.print(Panel(
        f"Bid    : [green]{bb}¢[/green]\nAsk    : [red]{ba}¢[/red]\nSpread : [yellow]{ba-bb}¢[/yellow]",
        title=f"{ticker} Spread", border_style="cyan"))


def show_price_history(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    hist = markets[ticker]["history"]
    t = Table(title=f"{ticker} Price History", border_style="yellow")
    for c in ["Time", "Price", ""]:
        t.add_column(c)
    mx = max(hist) if hist else 1
    for i, p in enumerate(hist):
        t.add_row(f"T-{len(hist)-i-1}", f"{p}¢", f"[cyan]{'█'*int(p/mx*25)}[/cyan]")
    console.print(t)


def show_news():
    t = Table(title="News", border_style="magenta")
    t.add_column("Ticker"); t.add_column("Headline")
    for tk, h in news:
        t.add_row(f"[cyan]{tk}[/cyan]", h)
    console.print(t)


def show_watchlist():
    t = Table(title="Watchlist", border_style="green")
    for c in ["Ticker", "Market", "Price", "24h"]:
        t.add_column(c)
    for tk in watchlist:
        if tk not in markets: continue
        d = markets[tk]
        ch = d["history"][-1] - d["history"][-2] if len(d["history"]) > 1 else 0
        cc = "green" if ch >= 0 else "red"
        t.add_row(tk, d["name"][:40], f"{d['price']}¢", f"[{cc}]{ch:+d}¢[/{cc}]")
    console.print(t)


def show_balance():
    if not api.connected:
        console.print("[yellow]  Requires live connection.[/yellow]"); return
    data, err = api.get_balance()
    if err:
        console.print(f"[red]  {err}[/red]"); return
    console.print(Panel(f"Balance : [green]${data.get('balance',0)/100:.2f}[/green]",
                        title="Account", border_style="green"))


# ── Stage 3 — Quant ────────────────────────────────────────

def show_ev(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    r = expected_value(markets[ticker])
    c = "green" if r["ev"] > 0 else "red"
    console.print(Panel(
        f"Probability    : {r['prob']*100:.1f}%\n"
        f"Ask (cost)     : ${r['cost']:.2f}\n\n"
        f"If YES wins    : +${r['win_payoff']:.2f}\n"
        f"If NO  wins    : ${r['loss_payoff']:.2f}\n\n"
        f"Expected Value : [{c}]${r['ev']:+.4f}[/{c}] per $1",
        title=f"{ticker} EV", border_style="green"))


def show_kelly(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    r = kelly_criterion(markets[ticker])
    console.print(Panel(
        f"Model Prob      : {r['model_prob']*100:.1f}%\n\n"
        f"Full Kelly      : [yellow]{r['kelly_fraction']*100:.1f}%[/yellow]\n"
        f"Half Kelly      : {r['half_kelly']*100:.1f}%\n"
        f"Quarter Kelly   : {r['quarter_kelly']*100:.1f}%",
        title=f"{ticker} Kelly", border_style="yellow"))


def show_arb():
    opps = arbitrage_check(markets)
    if not opps:
        console.print(Panel("[green]✓ No arbitrage found.[/green]",
                            title="Arb Scan", border_style="green")); return
    t = Table(title="Arbitrage", border_style="red")
    t.add_column("Ticker"); t.add_column("Profit")
    for tk, p in opps:
        t.add_row(tk, f"[green]+{p}¢[/green]")
    console.print(t)


def show_portfolio():
    rows, tc, tv = portfolio_exposure(portfolio, markets)
    t = Table(title="Portfolio", border_style="magenta")
    for c in ["Ticker", "Side", "Shares", "Entry", "Current", "P&L"]:
        t.add_column(c)
    for r in rows:
        pc = "green" if r["pnl"] >= 0 else "red"
        t.add_row(r["ticker"], r["side"], str(r["shares"]),
                  f"{r['entry']}¢", f"{r['current']}¢", f"[{pc}]${r['pnl']:+.2f}[/{pc}]")
    console.print(t)
    pc = "green" if tv - tc >= 0 else "red"
    console.print(f"  Total P&L : [{pc}]${tv-tc:+.2f}[/{pc}]")


# ── Stage 4 — Financial Engineering ────────────────────────

def show_sim(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    console.print("[dim]  Running 10,000 simulations...[/dim]")
    r = monte_carlo(markets[ticker])
    console.print(Panel(
        f"Mean    : [{_cp(r['mean'])}]{r['mean']*100:.1f}%[/{_cp(r['mean'])}]\n"
        f"Median  : {r['median']*100:.1f}%\n"
        f"Std Dev : {r['std']:.4f}\n\n"
        f"P5      : {r['p5']*100:.1f}%\n"
        f"P95     : {r['p95']*100:.1f}%\n\n"
        f"VaR 5%  : [red]{r['var']*100:.1f}%[/red]",
        title=f"{ticker} Monte Carlo", border_style="blue"))


def show_var(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    r = value_at_risk(markets[ticker])
    console.print(Panel(
        f"Mean  : {r['mean']*100:.1f}%\n"
        f"VaR   : [red]{r['var']*100:.1f}%[/red]\n"
        f"CVaR  : [red]{r['cvar']*100:.1f}%[/red]",
        title=f"{ticker} VaR", border_style="red"))


def show_correlation():
    tickers, corr = correlation_matrix(markets)
    t = Table(title="Correlation Matrix", border_style="cyan")
    t.add_column("", style="bold")
    for tk in tickers:
        t.add_column(tk)
    for i, tk in enumerate(tickers):
        row = [tk]
        for j in range(len(tickers)):
            c = corr[i, j]
            col = "green" if c > 0.5 else "red" if c < -0.5 else "yellow"
            row.append(f"[{col}]{c:+.2f}[/{col}]")
        t.add_row(*row)
    console.print(t)


def show_bayes(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    r = bayesian_update(markets[ticker])
    sc = "green" if r["shift"] > 0 else "red"
    console.print(Panel(
        f"Prior      : {r['prior']*100:.1f}%\n"
        f"Posterior   : [{_cp(r['posterior'])}]{r['posterior']*100:.1f}%[/{_cp(r['posterior'])}]\n"
        f"Std Dev    : {r['std']:.4f}\n"
        f"Shift      : [{sc}]{r['shift']*100:+.1f}%[/{sc}]\n\n"
        f"95% CI     : [{r['ci_low']*100:.1f}% — {r['ci_high']*100:.1f}%]",
        title=f"{ticker} Bayesian", border_style="magenta"))


def show_kalman(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    est = kalman_track(markets[ticker])
    t = Table(title=f"{ticker} Kalman Filter", border_style="blue")
    for c in ["Time", "Raw", "Filtered", "Var"]:
        t.add_column(c)
    for i, (raw, x, p) in enumerate(est):
        t.add_row(f"T-{len(est)-i-1}", f"{raw*100:.1f}%", f"[cyan]{x*100:.1f}%[/cyan]", f"{p:.4f}")
    console.print(t)


# ── Extras ──────────────────────────────────────────────────

def show_prob(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    r = prob_analysis(markets[ticker])
    console.print(Panel(
        f"Probability : [{_cp(r['prob'])}]{r['prob']*100:.1f}%[/{_cp(r['prob'])}]\n"
        f"E[X]        : {r['expect']:.4f}\n"
        f"Var[X]      : {r['var']:.4f}\n"
        f"σ           : {r['std']:.4f}\n\n"
        f"95% CI      : [{r['ci_low']*100:.1f}% — {r['ci_high']*100:.1f}%]",
        title=f"{ticker} Probability", border_style="cyan"))


def show_edge(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    r = edge_calc(markets[ticker])
    ec = "green" if r["edge"] > 0 else "red"
    rc = {"MINIMAL":"dim","LOW":"green","MEDIUM":"yellow","HIGH":"red"}[r["risk"]]
    console.print(Panel(
        f"Market : {r['market']*100:.1f}%\n"
        f"Model  : {r['model']*100:.1f}%\n\n"
        f"Edge   : [{ec}]{r['edge']*100:+.1f}%[/{ec}]\n"
        f"Kelly  : {r['kelly']*100:.1f}%\n"
        f"Risk   : [{rc}]{r['risk']}[/{rc}]",
        title=f"{ticker} Edge", border_style="green"))


def show_risk(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    e = edge_calc(markets[ticker])
    s = monte_carlo(markets[ticker])
    rc = {"MINIMAL":"dim","LOW":"green","MEDIUM":"yellow","HIGH":"red"}[e["risk"]]
    console.print(Panel(
        f"Market : {e['market']*100:.1f}%\n"
        f"Model  : {e['model']*100:.1f}%\n"
        f"Edge   : {e['edge']*100:+.1f}%\n"
        f"Kelly  : {e['kelly']*100:.1f}%\n\n"
        f"Risk   : [{rc}]◈ {e['risk']}[/{rc}]\n\n"
        f"VaR 5% : [red]{s['p5']*100:.1f}%[/red]",
        title=f"{ticker} Risk", border_style="red"))


def show_model(ticker):
    if not ensure_market(ticker): console.print("[red]Market not found.[/red]"); return
    d = markets[ticker]
    bid, ask = d["bids"][0][0], d["asks"][0][0]
    features = np.array([bid/100, ask/100, d["volume"]/1_000_000, (ask-bid)/100])
    nn  = PINN.predict(features)
    mkt = d["price"] / 100
    gap = nn - mkt
    gc  = "green" if gap > 0 else "red"
    console.print(Panel(
        f"Market    : {mkt*100:.1f}%\n"
        f"NN Est.   : [{_cp(nn)}]{nn*100:.1f}%[/{_cp(nn)}]\n"
        f"Gap       : [{gc}]{gap*100:+.1f}%[/{gc}]\n\n"
        f"[dim]Architecture[/dim]  : 4 → 8 → 1 (sigmoid)\n"
        f"[dim]Constraints[/dim]   : prob bounds, shrinkage, variance\n"
        f"[dim]Final loss[/dim]    : {PINN.losses[-1]:.6f}",
        title=f"{ticker} Neural Network", border_style="cyan"))
