"""
Command dispatcher.
Maps user input to display functions. No logic here.
"""

from rich.prompt import Prompt

from kalshi_terminal.data import command_history
from kalshi_terminal.display import (
    console, show_header, show_status_bar, show_help, show_status,
    show_search, show_command_history, show_markets, show_market,
    show_orderbook, show_spread, show_price_history, show_news,
    show_watchlist, show_balance, show_ev, show_kelly, show_arb,
    show_portfolio, show_sim, show_var, show_correlation, show_bayes,
    show_kalman, show_prob, show_edge, show_risk, show_model, _mode,
)

from kalshi_terminal.display import Panel


def run():
    show_header()
    console.print(Panel.fit(
        "[bold yellow]KALSHI TERMINAL[/bold yellow]\n"
        "[dim]Quantitative Prediction Market Research Platform[/dim]\n\n"
        f"Data : {_mode()}   ·   Type [cyan]HELP[/cyan] for commands",
        border_style="yellow"))
    show_status_bar()

    while True:
        try:
            raw = Prompt.ask("\n[bold cyan]KALSHI>[/bold cyan]").strip().upper()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not raw:
            continue

        command_history.append(raw)
        parts = raw.split()
        cmd   = parts[0]
        args  = parts[1:]

        try:
            # Stage 1
            if   cmd == "HELP":                show_help()
            elif cmd == "SEARCH" and args:     show_search(" ".join(args))
            elif cmd == "WATCHLIST":            show_watchlist()
            elif cmd == "HISTORY":             show_command_history()
            elif cmd == "STATUS":              show_status()
            elif cmd == "CLEAR":               show_header()
            # Stage 2
            elif cmd == "MARKETS":             show_markets()
            elif cmd == "MARKET"    and args:  show_market(args[0])
            elif cmd == "BOOK"      and args:  show_orderbook(args[0])
            elif cmd == "SPREAD"    and args:  show_spread(args[0])
            elif cmd == "PRICEHIST" and args:  show_price_history(args[0])
            elif cmd == "NEWS":                show_news()
            elif cmd == "BALANCE":             show_balance()
            # Stage 3
            elif cmd == "EV"        and args:  show_ev(args[0])
            elif cmd == "KELLY"     and args:  show_kelly(args[0])
            elif cmd == "ARB":                 show_arb()
            elif cmd == "PORTFOLIO":           show_portfolio()
            # Stage 4
            elif cmd == "SIM"       and args:  show_sim(args[0])
            elif cmd == "VAR"       and args:  show_var(args[0])
            elif cmd == "CORR":                show_correlation()
            elif cmd == "BAYES"     and args:  show_bayes(args[0])
            elif cmd == "KALMAN"    and args:  show_kalman(args[0])
            # Extras
            elif cmd == "PROB"      and args:  show_prob(args[0])
            elif cmd == "EDGE"      and args:  show_edge(args[0])
            elif cmd == "RISK"      and args:  show_risk(args[0])
            elif cmd == "MODEL"     and args:  show_model(args[0])

            elif cmd == "EXIT":
                console.print("[dim]Goodbye.[/dim]")
                break
            else:
                console.print("[red]Unknown command. Type HELP.[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        show_status_bar()
