from __future__ import annotations

from typing import Any


BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


def _request_json(url: str, params: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
    try:
        import requests
    except Exception as exc:
        return {"ok": False, "error": f"requests unavailable: {exc}"}

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return {"ok": True, "data": response.json(), "url": response.url}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": url}


def search_targets(query: str, limit: int = 5, timeout: float = 15.0) -> dict[str, Any]:
    """Search ChEMBL target records."""
    payload = _request_json(
        f"{BASE_URL}/target/search.json",
        params={"q": query, "limit": limit},
        timeout=timeout,
    )
    if not payload["ok"]:
        return {
            "ok": False,
            "tool": "chembl_target_search",
            "input": {"query": query, "limit": limit},
            "error": payload["error"],
            "url": payload.get("url"),
        }

    targets = payload["data"].get("targets", [])
    return {
        "ok": True,
        "tool": "chembl_target_search",
        "input": {"query": query, "limit": limit},
        "url": payload.get("url"),
        "targets": targets[:limit],
    }


def smoke_test() -> dict[str, Any]:
    return search_targets("EGFR", limit=5)
