from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import shutil
from pathlib import Path
from typing import Any

import requests

from src.utils import setup_logging, write_json, write_jsonl


DEEPCHEM_BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
DEEPCHEM_HIV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
TDC_DATAVERSE = {
    "davis": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/5219748",
        "delimiter": "\t",
    },
    "kiba": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/5255037",
        "delimiter": "\t",
    },
    "bindingdb_ic50": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/4291560",
        "delimiter": ",",
    },
}
SUPPORTED_DATASETS = {"davis", "kiba", "bindingdb_ic50", "bace", "hiv", "all"}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_delimited(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def download_file(url: str, path: Path, timeout: float = 60.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def stable_test_split(rows: list[dict[str, Any]], test_size: float = 0.2) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(42).shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * test_size))
    return shuffled[:test_count]


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[:limit]


def to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ic50_nm_to_pic50(ic50_nm: float) -> float:
    return 9.0 - math.log10(ic50_nm)


def activity_from_pic50(pic50: float) -> str:
    if pic50 >= 6.0:
        return "active"
    if pic50 < 5.0:
        return "inactive"
    return "uncertain"


def activity_from_ic50_nm(ic50_nm: float) -> str:
    if ic50_nm <= 1000:
        return "active"
    if ic50_nm > 10000:
        return "inactive"
    return "uncertain"


def write_dataset(path: Path, rows: list[dict[str, Any]]) -> int:
    write_jsonl(path, rows)
    return len(rows)


def download_deepchem_csv(name: str, url: str, raw_dir: Path) -> Path:
    output = raw_dir / f"{name}.csv"
    if output.exists() and output.stat().st_size > 0:
        logging.info("Using existing raw file: %s", output)
        return output
    logging.info("Downloading %s from %s", name, url)
    download_file(url, output)
    return output


def process_bace(output_dir: Path, activity_limit: int | None, pic50_limit: int | None) -> dict[str, Any]:
    raw_path = download_deepchem_csv("bace", DEEPCHEM_BACE_URL, output_dir / "raw")
    rows = stable_test_split(read_csv(raw_path))

    activity_rows: list[dict[str, Any]] = []
    pic50_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        smiles = str(row.get("mol", "")).strip()
        pic50 = to_float(row.get("pIC50"))
        klass = str(row.get("Class", "")).strip()
        cid = str(row.get("CID", "")).strip() or f"bace_{index:06d}"
        if not smiles:
            continue
        if klass in {"0", "1"}:
            activity_rows.append(
                {
                    "id": f"bace_{cid}",
                    "task": "target_activity_prediction",
                    "smiles": smiles,
                    "target_name": "BACE1",
                    "target_sequence": "",
                    "label": "active" if klass == "1" else "inactive",
                }
            )
        if pic50 is not None:
            pic50_rows.append(
                {
                    "id": f"bace_{cid}",
                    "task": "pic50_prediction",
                    "smiles": smiles,
                    "target_name": "BACE1",
                    "target_sequence": "",
                    "pIC50": pic50,
                }
            )

    processed = output_dir / "processed"
    activity_path = processed / "bace_activity_test.jsonl"
    pic50_path = processed / "bace_pic50_test.jsonl"
    write_dataset(activity_path, limit_rows(activity_rows, activity_limit))
    write_dataset(pic50_path, limit_rows(pic50_rows, pic50_limit))
    return {
        "dataset": "bace",
        "status": "success",
        "raw_file": str(raw_path),
        "activity_file": str(activity_path),
        "activity_rows": min(len(activity_rows), activity_limit or len(activity_rows)),
        "pic50_file": str(pic50_path),
        "pic50_rows": min(len(pic50_rows), pic50_limit or len(pic50_rows)),
        "notes": "DeepChem MoleculeNet BACE; pIC50 is negative log IC50; Class 1=active, 0=inactive.",
    }


def process_hiv(output_dir: Path, activity_limit: int | None) -> dict[str, Any]:
    raw_path = download_deepchem_csv("hiv", DEEPCHEM_HIV_URL, output_dir / "raw")
    rows = stable_test_split(read_csv(raw_path))

    activity_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        smiles = str(row.get("smiles", "")).strip()
        label_value = str(row.get("HIV_active", "")).strip()
        if not smiles or label_value not in {"0", "1"}:
            continue
        activity_rows.append(
            {
                "id": f"hiv_{index:06d}",
                "task": "target_activity_prediction",
                "smiles": smiles,
                "target_name": "HIV replication inhibition",
                "target_sequence": "",
                "label": "active" if label_value == "1" else "inactive",
            }
        )

    processed = output_dir / "processed"
    activity_path = processed / "hiv_activity_test.jsonl"
    write_dataset(activity_path, limit_rows(activity_rows, activity_limit))
    return {
        "dataset": "hiv",
        "status": "success",
        "raw_file": str(raw_path),
        "activity_file": str(activity_path),
        "activity_rows": min(len(activity_rows), activity_limit or len(activity_rows)),
        "notes": "DeepChem MoleculeNet HIV; HIV_active 1=active, 0=inactive.",
    }


def dataframe_to_records(frame: Any) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def get_tdc_test_records(name: str) -> list[dict[str, Any]]:
    try:
        from tdc.multi_pred import DTI  # type: ignore

        data = DTI(name=name)
        split = data.get_split(method="random", seed=42, frac=[0.8, 0.0, 0.2])
        return dataframe_to_records(split["test"])
    except ImportError as exc:
        raise RuntimeError(
            "PyTDC is not installed. Falling back to direct Dataverse is handled by caller."
        ) from exc


def get_dataverse_test_records(dataset_key: str, output_dir: Path) -> list[dict[str, Any]]:
    metadata = TDC_DATAVERSE[dataset_key]
    raw_path = output_dir / "raw" / f"{dataset_key}_tdc_dataverse.csv"
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        logging.info("Downloading %s direct Dataverse fallback from %s", dataset_key, metadata["url"])
        download_file(str(metadata["url"]), raw_path)
    records = read_delimited(raw_path, str(metadata["delimiter"]))
    return stable_test_split(records)


def get_tdc_or_dataverse_test_records(dataset_key: str, tdc_name: str, output_dir: Path) -> list[dict[str, Any]]:
    try:
        return get_tdc_test_records(tdc_name)
    except Exception as tdc_exc:
        logging.warning("%s via PyTDC failed: %s", dataset_key, tdc_exc)
        try:
            return get_dataverse_test_records(dataset_key, output_dir)
        except Exception as dataverse_exc:
            raise RuntimeError(
                f"PyTDC failed ({tdc_exc}); direct Dataverse fallback failed ({dataverse_exc})"
            ) from dataverse_exc


def find_first(row: dict[str, Any], keys: list[str]) -> Any:
    lowered = {str(key).lower(): key for key in row}
    for key in keys:
        actual = lowered.get(key.lower())
        if actual is not None:
            return row.get(actual)
    return ""


def process_tdc_affinity(
    dataset_key: str,
    tdc_name: str,
    output_dir: Path,
    pic50_limit: int | None,
) -> dict[str, Any]:
    records = get_tdc_or_dataverse_test_records(dataset_key, tdc_name, output_dir)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(records, start=1):
        smiles = str(find_first(row, ["Drug", "X1", "SMILES", "smiles"])).strip()
        target_sequence = str(
            find_first(row, ["Target", "X2", "Target Sequence", "target_sequence"])
        ).strip()
        target_name = str(find_first(row, ["Target_ID", "ID2", "Target Name", "target_name"])).strip()
        y_value = to_float(find_first(row, ["Y", "Label", "pIC50", "pKd"]))
        if not smiles or y_value is None:
            continue
        if dataset_key == "davis":
            if y_value <= 0:
                continue
            label_value = ic50_nm_to_pic50(y_value)
            label_note = "TDC DAVIS Y is raw Kd-like affinity in nM; converted to 9 - log10(Kd_nM)."
        elif dataset_key == "kiba":
            label_value = y_value
            label_note = "TDC KIBA Y is a dimensionless KIBA score; stored in pIC50 field for schema compatibility, not log-transformed."
        else:
            label_value = y_value
            label_note = f"TDC {tdc_name}; Y stored as provided by TDC."
        rows.append(
            {
                "id": f"{dataset_key}_{index:06d}",
                "task": "pic50_prediction",
                "smiles": smiles,
                "target_name": target_name or dataset_key.upper(),
                "target_sequence": target_sequence,
                "pIC50": label_value,
            }
        )

    output_path = output_dir / "processed" / f"{dataset_key}_pic50_test.jsonl"
    write_dataset(output_path, limit_rows(rows, pic50_limit))
    return {
        "dataset": dataset_key,
        "status": "success",
        "pic50_file": str(output_path),
        "pic50_rows": min(len(rows), pic50_limit or len(rows)),
        "notes": label_note if rows else f"TDC {tdc_name}; no convertible rows found.",
    }


def process_tdc_bindingdb(
    output_dir: Path,
    activity_limit: int | None,
    pic50_limit: int | None,
) -> dict[str, Any]:
    records = get_tdc_or_dataverse_test_records("bindingdb_ic50", "BindingDB_IC50", output_dir)
    activity_rows: list[dict[str, Any]] = []
    pic50_rows: list[dict[str, Any]] = []
    for index, row in enumerate(records, start=1):
        smiles = str(find_first(row, ["Drug", "X1", "SMILES", "smiles"])).strip()
        target_sequence = str(
            find_first(row, ["Target", "X2", "Target Sequence", "target_sequence"])
        ).strip()
        target_name = str(find_first(row, ["Target_ID", "ID2", "Target Name", "target_name"])).strip()
        y_value = to_float(find_first(row, ["Y", "Label", "IC50", "ic50"]))
        if not smiles or y_value is None:
            continue
        pic50 = ic50_nm_to_pic50(y_value) if y_value > 20 else y_value
        label = activity_from_ic50_nm(y_value) if y_value > 20 else activity_from_pic50(pic50)
        base = {
            "smiles": smiles,
            "target_name": target_name or "BindingDB target",
            "target_sequence": target_sequence,
        }
        activity_rows.append(
            {
                "id": f"bindingdb_ic50_{index:06d}",
                "task": "target_activity_prediction",
                **base,
                "label": label,
            }
        )
        pic50_rows.append(
            {
                "id": f"bindingdb_ic50_{index:06d}",
                "task": "pic50_prediction",
                **base,
                "pIC50": pic50,
            }
        )

    processed = output_dir / "processed"
    activity_path = processed / "bindingdb_ic50_activity_test.jsonl"
    pic50_path = processed / "bindingdb_ic50_pic50_test.jsonl"
    write_dataset(activity_path, limit_rows(activity_rows, activity_limit))
    write_dataset(pic50_path, limit_rows(pic50_rows, pic50_limit))
    return {
        "dataset": "bindingdb_ic50",
        "status": "success",
        "activity_file": str(activity_path),
        "activity_rows": min(len(activity_rows), activity_limit or len(activity_rows)),
        "pic50_file": str(pic50_path),
        "pic50_rows": min(len(pic50_rows), pic50_limit or len(pic50_rows)),
        "notes": "TDC BindingDB_IC50; Y converted as IC50 nM when scale appears nM, otherwise treated as pIC50.",
    }


def dataset_failed(name: str, exc: Exception) -> dict[str, Any]:
    logging.warning("Dataset %s failed: %s", name, exc)
    return {"dataset": name, "status": "failed", "error": str(exc)}


def choose_first_success(results: list[dict[str, Any]], key: str) -> str | None:
    for result in results:
        if result.get("status") == "success" and result.get(key):
            path = Path(str(result[key]))
            if path.exists() and path.stat().st_size > 0:
                return str(path)
    return None


def create_default_outputs(output_dir: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    activity_source = choose_first_success(
        [result for result in results if result.get("dataset") == "bindingdb_ic50"], "activity_file"
    )
    if activity_source is None:
        activity_source = choose_first_success(
            [result for result in results if result.get("dataset") in {"bace", "hiv"}], "activity_file"
        )

    pic50_source = choose_first_success(
        [result for result in results if result.get("dataset") == "davis"], "pic50_file"
    )
    if pic50_source is None:
        pic50_source = choose_first_success(
            [result for result in results if result.get("dataset") in {"kiba", "bindingdb_ic50", "bace"}],
            "pic50_file",
        )

    defaults: dict[str, Any] = {}
    if activity_source:
        destination = output_dir / "test_activity.jsonl"
        shutil.copyfile(activity_source, destination)
        defaults["test_activity"] = {"source": activity_source, "path": str(destination)}
    if pic50_source:
        destination = output_dir / "test_pic50.jsonl"
        shutil.copyfile(pic50_source, destination)
        defaults["test_pic50"] = {"source": pic50_source, "path": str(destination)}
    return defaults


def ensure_placeholder_files(output_dir: Path) -> None:
    processed = output_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for filename in [
        "davis_pic50_test.jsonl",
        "kiba_pic50_test.jsonl",
        "bindingdb_ic50_activity_test.jsonl",
        "bindingdb_ic50_pic50_test.jsonl",
        "bace_activity_test.jsonl",
        "hiv_activity_test.jsonl",
    ]:
        path = processed / filename
        if not path.exists():
            path.touch()


def run_downloads(
    dataset: str,
    output_dir: Path,
    limit: int | None,
    activity_limit: int | None,
    pic50_limit: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)
    if limit is not None:
        activity_limit = activity_limit or limit
        pic50_limit = pic50_limit or limit

    selected = ["davis", "kiba", "bindingdb_ic50", "bace", "hiv"] if dataset == "all" else [dataset]
    results: list[dict[str, Any]] = []
    for item in selected:
        try:
            if item == "davis":
                results.append(process_tdc_affinity("davis", "DAVIS", output_dir, pic50_limit))
            elif item == "kiba":
                results.append(process_tdc_affinity("kiba", "KIBA", output_dir, pic50_limit))
            elif item == "bindingdb_ic50":
                results.append(process_tdc_bindingdb(output_dir, activity_limit, pic50_limit))
            elif item == "bace":
                results.append(process_bace(output_dir, activity_limit, pic50_limit))
            elif item == "hiv":
                results.append(process_hiv(output_dir, activity_limit))
            else:
                raise ValueError(f"Unsupported dataset '{item}'")
        except Exception as exc:
            results.append(dataset_failed(item, exc))

    defaults = create_default_outputs(output_dir, results)
    ensure_placeholder_files(output_dir)
    summary = {"results": results, "default_outputs": defaults}
    write_json(output_dir / "processed" / "download_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert real drug-discovery benchmark datasets.")
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--activity-limit", type=int, default=None)
    parser.add_argument("--pic50-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    summary = run_downloads(args.dataset, args.output_dir, args.limit, args.activity_limit, args.pic50_limit)
    print("Download summary:")
    for result in summary["results"]:
        print(result)
    print("Default outputs:", summary["default_outputs"])


if __name__ == "__main__":
    main()
