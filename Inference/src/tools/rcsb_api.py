from __future__ import annotations

from typing import Any


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"


def _post_json(url: str, payload: dict[str, Any], timeout: float = 20.0) -> dict[str, Any]:
    try:
        import requests
    except Exception as exc:
        return {"ok": False, "error": f"requests unavailable: {exc}"}

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return {"ok": True, "data": response.json(), "url": response.url}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": url}


def _get_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    try:
        import requests
    except Exception as exc:
        return {"ok": False, "error": f"requests unavailable: {exc}"}

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return {"ok": True, "data": response.json(), "url": response.url}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": url}


def search_by_uniprot(accession: str, rows: int = 10, timeout: float = 20.0) -> dict[str, Any]:
    """Search RCSB polymer entities associated with a UniProt accession."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": accession,
            },
        },
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "return_all_hits": False,
        },
        "return_type": "polymer_entity",
    }
    payload = _post_json(SEARCH_URL, query, timeout=timeout)
    if not payload["ok"]:
        return {
            "ok": False,
            "tool": "rcsb_uniprot_search",
            "input": {"accession": accession, "rows": rows},
            "error": payload["error"],
            "url": payload.get("url"),
        }

    result_set = payload["data"].get("result_set", [])
    return {
        "ok": True,
        "tool": "rcsb_uniprot_search",
        "input": {"accession": accession, "rows": rows},
        "url": payload.get("url"),
        "result_set": result_set,
    }


def get_entry(pdb_id: str, timeout: float = 15.0) -> dict[str, Any]:
    payload = _get_json(f"{DATA_URL}/{pdb_id}", timeout=timeout)
    if not payload["ok"]:
        return {
            "ok": False,
            "tool": "rcsb_entry",
            "input": {"pdb_id": pdb_id},
            "error": payload["error"],
            "url": payload.get("url"),
        }
    data = payload["data"]
    return {
        "ok": True,
        "tool": "rcsb_entry",
        "input": {"pdb_id": pdb_id},
        "url": payload.get("url"),
        "entry": {
            "entry_id": data.get("rcsb_id"),
            "title": data.get("struct", {}).get("title"),
            "experimental_methods": [item.get("method") for item in data.get("exptl", [])],
        },
    }


def smoke_test() -> dict[str, Any]:
    return search_by_uniprot("P00533", rows=5)
