from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

from src.prompts import SUPPORTED_TASKS
from src.utils import read_jsonl, setup_logging, write_jsonl


COLUMN_ALIASES = {
    "smiles": ["smiles", "canonical_smiles", "molecule_smiles", "compound_smiles"],
    "target_name": ["target_name", "target", "protein_name", "gene_name"],
    "target_sequence": ["target_sequence", "sequence", "protein_sequence"],
    "label": ["label", "activity_label", "active", "class"],
    "pIC50": ["pIC50", "pic50", "pchembl_value", "pchembl"],
}

TARGET_AWARE_TASKS = {"target_activity_prediction", "pic50_prediction", "molecule_optimization"}
MOLECULE_TASKS = {
    "target_activity_prediction",
    "pic50_prediction",
    "molecule_understanding",
    "molecule_optimization",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return read_jsonl(path)
    raise ValueError(f"Unsupported input extension '{path.suffix}'. Use CSV or JSONL.")


def detect_columns(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    if not rows:
        return {field: None for field in COLUMN_ALIASES}

    columns: dict[str, str] = {}
    for row in rows:
        for column in row:
            columns.setdefault(column.lower(), column)

    mapping: dict[str, str | None] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        detected = None
        for alias in aliases:
            if alias.lower() in columns:
                detected = columns[alias.lower()]
                break
        mapping[canonical] = detected
    return mapping


def value_from_row(row: dict[str, Any], column: str | None) -> Any:
    if column is None:
        return ""
    value = row.get(column, "")
    return "" if value is None else value


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "active", "act", "positive"}:
        return "active"
    if text in {"0", "false", "no", "inactive", "inact", "negative"}:
        return "inactive"
    if text in {"uncertain", "unknown", "ambiguous"}:
        return "uncertain"
    return text


def is_missing(value: Any) -> bool:
    return str(value).strip() == ""


def convert_row(
    row: dict[str, Any],
    row_index: int,
    task: str,
    mapping: dict[str, str | None],
) -> dict[str, Any] | None:
    smiles = str(value_from_row(row, mapping["smiles"])).strip()
    target_name = str(value_from_row(row, mapping["target_name"])).strip()
    target_sequence = str(value_from_row(row, mapping["target_sequence"])).strip()
    label = normalize_label(value_from_row(row, mapping["label"]))
    pic50 = str(value_from_row(row, mapping["pIC50"])).strip()

    if task in MOLECULE_TASKS and is_missing(smiles):
        return None
    if task in TARGET_AWARE_TASKS and is_missing(target_name) and is_missing(target_sequence):
        return None
    if task == "target_activity_prediction" and is_missing(label):
        return None
    if task == "pic50_prediction" and is_missing(pic50):
        return None

    sample: dict[str, Any] = {
        "id": str(row.get("id") or row.get("sample_id") or f"{task}_{row_index:06d}"),
        "task": task,
    }
    if smiles:
        sample["smiles"] = smiles
    if target_name:
        sample["target_name"] = target_name
    if target_sequence:
        sample["target_sequence"] = target_sequence
    if task == "target_activity_prediction":
        sample["label"] = label
    if task == "pic50_prediction":
        try:
            sample["pIC50"] = float(pic50)
        except ValueError:
            sample["pIC50"] = pic50
    if task == "molecule_understanding":
        sample["question"] = "Summarize functional groups, drug-likeness, possible risks, and key limitations."
    if task == "molecule_optimization":
        sample["optimization_goal"] = str(
            row.get("optimization_goal")
            or row.get("goal")
            or "Suggest a conservative modification for the target while preserving drug-like properties."
        )
    return sample


def prepare_dataset(input_path: Path, output_path: Path, task: str, limit: int | None = None) -> dict[str, Any]:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}")

    rows = read_rows(input_path)
    mapping = detect_columns(rows)
    output_rows: list[dict[str, Any]] = []
    skipped = 0

    for index, row in enumerate(rows, start=1):
        converted = convert_row(row, index, task, mapping)
        if converted is None:
            skipped += 1
            continue
        output_rows.append(converted)
        if limit is not None and len(output_rows) >= limit:
            break

    write_jsonl(output_path, output_rows)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "task": task,
        "rows_read": len(rows),
        "rows_written": len(output_rows),
        "rows_skipped": skipped,
        "column_mapping": mapping,
    }
    logging.info("Rows read: %d", len(rows))
    logging.info("Rows written: %d", len(output_rows))
    logging.info("Rows skipped: %d", skipped)
    logging.info("Detected column mapping: %s", mapping)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark CSV/JSONL files for MedMO inference.")
    parser.add_argument("--input", required=True, type=Path, help="Raw CSV or JSONL input")
    parser.add_argument("--output", required=True, type=Path, help="Prepared JSONL output")
    parser.add_argument("--task", required=True, choices=sorted(SUPPORTED_TASKS), help="Task name")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of converted rows")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    prepare_dataset(args.input, args.output, args.task, args.limit)


if __name__ == "__main__":
    main()
