"""One-time Google Calendar OAuth, so the agent can act as you.

    python -m agent.connect

Prints a Composio Connect link, waits while you grant access in your browser,
then confirms. You authorise on Google's own page -- no credential passes
through this code.

Composio retired `connected_accounts.initiate` for their managed OAuth in favour
of `connected_accounts.link`, and `toolkits.authorize` still calls the old one,
so it fails with ComposioLegacyConnectedAccountsEndpointRetiredError. This walks
the current path explicitly: find (or create) the auth config, then link.

The user id MUST match the one tools.execute() sends at run time, which is why
both read COMPOSIO_USER_ID with the same default. A connection made under a
different id looks identical in the dashboard and fails at execute with a
"no connected account" error -- a miserable thing to debug on demo day.
"""
from __future__ import annotations

import argparse
import os

TOOLKIT = "googlecalendar"
DEFAULT_USER = os.environ.get("COMPOSIO_USER_ID", "speller")


def _items(page):
    return getattr(page, "items", page) or []


def auth_config_for(composio, toolkit: str) -> str:
    """Reuse this toolkit's auth config, creating a Composio-managed one if new."""
    for config in _items(composio.auth_configs.list()):
        slug = getattr(getattr(config, "toolkit", None), "slug", None)
        if slug == toolkit:
            print(f"auth config: {config.id} (existing)")
            return config.id
    created = composio.auth_configs.create(
        toolkit=toolkit, options={"type": "use_composio_managed_auth"},
    )
    print(f"auth config: {created.id} (created)")
    return created.id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--toolkit", default=TOOLKIT, help=f"toolkit slug (default: {TOOLKIT})")
    parser.add_argument("--user", default=DEFAULT_USER,
                        help=f"Composio user id (default: {DEFAULT_USER}; must match at run time)")
    parser.add_argument("--auth-config", default=os.environ.get("COMPOSIO_AUTH_CONFIG_ID"),
                        help="force a specific auth config id (ac_...)")
    args = parser.parse_args()

    if not os.environ.get("COMPOSIO_API_KEY"):
        raise SystemExit("COMPOSIO_API_KEY is not set; add it to .env")

    from composio import Composio

    composio = Composio()
    auth_config = args.auth_config or auth_config_for(composio, args.toolkit)

    request = composio.connected_accounts.link(args.user, auth_config)
    print(f"\nOpen this and grant calendar access as yourself:\n\n  {request.redirect_url}\n")
    print("Waiting for you to finish in the browser (Ctrl+C to give up)...")
    account = request.wait_for_connection()
    print(f"\nConnected. account={getattr(account, 'id', account)} user_id={args.user!r}")
    print("Now check the argument names with:  python -m agent.tools --schema")


if __name__ == "__main__":
    main()
