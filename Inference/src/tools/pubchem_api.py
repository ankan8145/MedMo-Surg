from __future__ import annotations

from typing import Any


BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DEFAULT_PROPERTIES = [
    "MolecularFormula",
    "MolecularWeight",
    "CanonicalSMILES",
    "IsomericSMILES",
    "IUPACName",
    "InChIKey",
    "XLogP",
    "TPSA",
]


def _request_json(url: str, params: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
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


def lookup_compound(
    name: str | None = None,
    cid: str | int | None = None,
    smiles: str | None = None,
    properties: list[str] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Lookup PubChem compound properties by name, CID, or SMILES."""
    selected = properties or DEFAULT_PROPERTIES
    property_text = ",".join(selected)
    query: dict[str, Any] = {"name": name, "cid": cid, "smiles": smiles}

    if cid is not None:
        url = f"{BASE_URL}/compound/cid/{cid}/property/{property_text}/JSON"
        params = None
    elif name is not None:
        url = f"{BASE_URL}/compound/name/{name}/property/{property_text}/JSON"
        params = None
    elif smiles is not None:
        url = f"{BASE_URL}/compound/smiles/property/{property_text}/JSON"
        params = {"smiles": smiles}
    else:
        return {"ok": False, "tool": "pubchem_lookup", "input": query, "error": "Provide name, cid, or smiles"}

    payload = _request_json(url, params=params, timeout=timeout)
    if not payload["ok"]:
        return {"ok": False, "tool": "pubchem_lookup", "input": query, "error": payload["error"], "url": payload.get("url")}

    compounds = payload["data"].get("PropertyTable", {}).get("Properties", [])
    return {
        "ok": True,
        "tool": "pubchem_lookup",
        "input": query,
        "url": payload.get("url"),
        "compounds": compounds,
    }


def smoke_test() -> dict[str, Any]:
    return lookup_compound(name="aspirin")
