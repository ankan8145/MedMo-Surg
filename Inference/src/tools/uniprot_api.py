from __future__ import annotations

from typing import Any


BASE_URL = "https://rest.uniprot.org/uniprotkb"


def _request_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
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


def get_entry(accession: str, timeout: float = 15.0) -> dict[str, Any]:
    """Fetch a UniProtKB entry by accession."""
    payload = _request_json(f"{BASE_URL}/{accession}.json", timeout=timeout)
    if not payload["ok"]:
        return {
            "ok": False,
            "tool": "uniprot_entry",
            "input": {"accession": accession},
            "error": payload["error"],
            "url": payload.get("url"),
        }

    data = payload["data"]
    protein_description = data.get("proteinDescription", {})
    recommended_name = protein_description.get("recommendedName", {}).get("fullName", {}).get("value")
    genes = [gene.get("geneName", {}).get("value") for gene in data.get("genes", []) if gene.get("geneName")]
    entry = {
        "primary_accession": data.get("primaryAccession"),
        "entry_name": data.get("uniProtkbId"),
        "protein_name": recommended_name,
        "genes": genes,
        "organism": data.get("organism", {}).get("scientificName"),
        "sequence_length": data.get("sequence", {}).get("length"),
    }
    return {
        "ok": True,
        "tool": "uniprot_entry",
        "input": {"accession": accession},
        "url": payload.get("url"),
        "entry": entry,
    }


def smoke_test() -> dict[str, Any]:
    return get_entry("P00533")
