from __future__ import annotations

from typing import Any


def search_bindingdb(query: str, limit: int = 5) -> dict[str, Any]:
    """Placeholder for BindingDB integration.

    BindingDB endpoint and licensing expectations should be confirmed before
    adding network calls. This intentionally fails gracefully.
    """
    return {
        "ok": False,
        "tool": "bindingdb_search",
        "implemented": False,
        "input": {"query": query, "limit": limit},
        "error": "BindingDB API wrapper is not implemented yet.",
    }


def smoke_test() -> dict[str, Any]:
    return search_bindingdb("EGFR", limit=5)
