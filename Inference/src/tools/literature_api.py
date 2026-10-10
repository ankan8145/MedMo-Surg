from __future__ import annotations

from typing import Any


EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def search_europe_pmc(query: str, page_size: int = 5, timeout: float = 15.0) -> dict[str, Any]:
    """Search Europe PMC literature records."""
    try:
        import requests
    except Exception as exc:
        return {
            "ok": False,
            "tool": "europe_pmc_search",
            "input": {"query": query, "page_size": page_size},
            "error": f"requests unavailable: {exc}",
        }

    params = {"query": query, "format": "json", "pageSize": page_size}
    try:
        response = requests.get(EUROPE_PMC_SEARCH_URL, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return {
            "ok": False,
            "tool": "europe_pmc_search",
            "input": {"query": query, "page_size": page_size},
            "error": str(exc),
            "url": EUROPE_PMC_SEARCH_URL,
        }

    results = data.get("resultList", {}).get("result", [])
    compact_results = [
        {
            "id": item.get("id"),
            "source": item.get("source"),
            "title": item.get("title"),
            "journal": item.get("journalTitle"),
            "year": item.get("pubYear"),
            "doi": item.get("doi"),
        }
        for item in results[:page_size]
    ]
    return {
        "ok": True,
        "tool": "europe_pmc_search",
        "input": {"query": query, "page_size": page_size},
        "url": response.url,
        "results": compact_results,
    }


def smoke_test() -> dict[str, Any]:
    return search_europe_pmc("EGFR inhibitor", page_size=5)
