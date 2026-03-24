"""CLI command: hl selftest — read-only connectivity & config check."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from hltrader.config import settings

app = typer.Typer(name="selftest", help="Verify connectivity and config (read-only, no trades)")
console = Console()


@app.command(name="run")
def run() -> None:
    """Run a read-only self-test against the Hyperliquid API.

    Checks: config loaded, wallet derived, API reachable, metadata fetched,
    positions/orders readable, mid-prices fetched.  No orders are placed.
    """
    api_url = settings.effective_api_url
    is_testnet = settings.HL_TESTNET
    is_dry_run = settings.HL_DRY_RUN

    console.print("[bold]Self-test starting...[/bold]\n")

    results: list[tuple[str, bool, str]] = []

    # 1. Config
    has_key = bool(settings.HL_PRIVATE_KEY)
    results.append(("Private key set", has_key, "OK" if has_key else "HL_PRIVATE_KEY is empty"))

    net_label = f"testnet ({api_url})" if is_testnet else f"mainnet ({api_url})"
    results.append(("Network", True, net_label))
    results.append(("Dry-run mode", True, "ON" if is_dry_run else "OFF"))

    # 2. Wallet
    address = ""
    try:
        from hltrader.client import get_account
        acct = get_account()
        address = acct.address
        results.append(("Wallet address", True, address))
    except Exception as exc:
        results.append(("Wallet address", False, str(exc)))

    # 3. API connectivity — meta
    try:
        from hltrader.client import get_info
        info = get_info()
        meta = info.meta()
        n_coins = len(meta["universe"])
        results.append(("Fetch metadata", True, f"{n_coins} perp coins available"))
    except Exception as exc:
        results.append(("Fetch metadata", False, str(exc)))

    # 4. All mids (pricing)
    try:
        mids = info.all_mids()
        btc_mid = mids.get("BTC", mids.get("BTC", "?"))
        eth_mid = mids.get("ETH", "?")
        results.append(("Fetch mid-prices", True, f"BTC={btc_mid}  ETH={eth_mid}"))
    except Exception as exc:
        results.append(("Fetch mid-prices", False, str(exc)))

    # 5. User state (positions)
    if address:
        try:
            user_state = info.user_state(address)
            acct_val = user_state["marginSummary"]["accountValue"]
            n_pos = sum(
                1 for p in user_state["assetPositions"]
                if float(p["position"]["szi"]) != 0
            )
            results.append(("Account state", True, f"value={acct_val}  open_positions={n_pos}"))
        except Exception as exc:
            results.append(("Account state", False, str(exc)))

        # 6. Frontend open orders (trigger orders)
        try:
            orders = info.frontend_open_orders(address)
            n_triggers = sum(1 for o in orders if o.get("isTrigger", False))
            results.append(("Trigger orders", True, f"{n_triggers} active trigger order(s)"))
        except Exception as exc:
            results.append(("Trigger orders", False, str(exc)))

    # Display results
    console.print()
    table = Table(title="Self-Test Results")
    table.add_column("Check", style="cyan")
    table.add_column("Pass", justify="center")
    table.add_column("Details")

    all_pass = True
    for check, passed, detail in results:
        mark = "[green]Y[/green]" if passed else "[red]N[/red]"
        if not passed:
            all_pass = False
        table.add_row(check, mark, detail)

    console.print(table)
    console.print()

    if all_pass:
        console.print("[bold green]All checks passed.[/bold green] Your config is ready.")
    else:
        console.print("[bold yellow]Some checks failed.[/bold yellow] Review the table above.")

    if is_dry_run:
        console.print(
            "\n[dim]Dry-run mode is ON — all commands will log orders "
            "instead of sending them to the exchange.[/dim]"
        )
    if is_testnet:
        console.print(
            "\n[dim]Testnet mode is ON — using Hyperliquid testnet "
            "(separate wallet/funds from mainnet).[/dim]"
        )
