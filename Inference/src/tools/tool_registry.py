from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from src.tools import bindingdb_api, chembl_api, literature_api, pubchem_api, rcsb_api, rdkit_tools, uniprot_api


ToolFunction = Callable[..., dict[str, Any]]


TOOLS: dict[str, dict[str, Any]] = {
    "rdkit_descriptors": {
        "implemented": True,
        "module": "src.tools.rdkit_tools",
        "function": "calculate_descriptors",
        "input_schema": {"smiles": "str"},
        "output_schema": {
            "ok": "bool",
            "valid_smiles": "bool|null",
            "descriptors": "dict|null",
            "error": "str|null",
        },
        "used_during_inference": False,
        "callable": rdkit_tools.calculate_descriptors,
    },
    "pubchem_lookup": {
        "implemented": True,
        "module": "src.tools.pubchem_api",
        "function": "lookup_compound",
        "input_schema": {"name": "str|null", "cid": "str|int|null", "smiles": "str|null"},
        "output_schema": {"ok": "bool", "compounds": "list[dict]", "error": "str|null"},
        "used_during_inference": False,
        "callable": pubchem_api.lookup_compound,
    },
    "chembl_target_search": {
        "implemented": True,
        "module": "src.tools.chembl_api",
        "function": "search_targets",
        "input_schema": {"query": "str", "limit": "int"},
        "output_schema": {"ok": "bool", "targets": "list[dict]", "error": "str|null"},
        "used_during_inference": False,
        "callable": chembl_api.search_targets,
    },
    "uniprot_entry": {
        "implemented": True,
        "module": "src.tools.uniprot_api",
        "function": "get_entry",
        "input_schema": {"accession": "str"},
        "output_schema": {"ok": "bool", "entry": "dict|null", "error": "str|null"},
        "used_during_inference": False,
        "callable": uniprot_api.get_entry,
    },
    "rcsb_uniprot_search": {
        "implemented": True,
        "module": "src.tools.rcsb_api",
        "function": "search_by_uniprot",
        "input_schema": {"accession": "str", "rows": "int"},
        "output_schema": {"ok": "bool", "result_set": "list[dict]", "error": "str|null"},
        "used_during_inference": False,
        "callable": rcsb_api.search_by_uniprot,
    },
    "europe_pmc_search": {
        "implemented": True,
        "module": "src.tools.literature_api",
        "function": "search_europe_pmc",
        "input_schema": {"query": "str", "page_size": "int"},
        "output_schema": {"ok": "bool", "results": "list[dict]", "error": "str|null"},
        "used_during_inference": False,
        "callable": literature_api.search_europe_pmc,
    },
    "bindingdb_search": {
        "implemented": False,
        "module": "src.tools.bindingdb_api",
        "function": "search_bindingdb",
        "input_schema": {"query": "str", "limit": "int"},
        "output_schema": {"ok": "bool", "implemented": "False", "error": "str"},
        "used_during_inference": False,
        "callable": bindingdb_api.search_bindingdb,
    },
}


def list_tools() -> list[dict[str, Any]]:
    listed: list[dict[str, Any]] = []
    for name, metadata in TOOLS.items():
        listed.append({key: value for key, value in metadata.items() if key != "callable"} | {"name": name})
    return listed


def run_smoke_tests() -> dict[str, Any]:
    tests = {
        "rdkit_descriptors_aspirin": rdkit_tools.smoke_test,
        "pubchem_aspirin": pubchem_api.smoke_test,
        "chembl_egfr": chembl_api.smoke_test,
        "uniprot_p00533": uniprot_api.smoke_test,
        "rcsb_uniprot_p00533": rcsb_api.smoke_test,
        "europe_pmc_egfr_inhibitor": literature_api.smoke_test,
        "bindingdb_placeholder": bindingdb_api.smoke_test,
    }
    results: dict[str, Any] = {}
    for name, test_func in tests.items():
        try:
            results[name] = test_func()
        except Exception as exc:
            results[name] = {"ok": False, "error": f"Unhandled tool test error: {exc}"}
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List and smoke-test optional drug-discovery tools.")
    parser.add_argument("--list", action="store_true", help="List registered tools and schemas")
    parser.add_argument("--test", action="store_true", help="Run smoke tests for registered tools")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list:
        print(json.dumps({"tools": list_tools()}, indent=2, sort_keys=True))
    if args.test:
        print(json.dumps({"tests": run_smoke_tests()}, indent=2, sort_keys=True))
    if not args.list and not args.test:
        build_parser().print_help()


if __name__ == "__main__":
    main()
