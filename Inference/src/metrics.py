"""Evaluate JSONL prediction files for local inference tests."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTIVITY_LABELS = ("active", "inactive", "uncertain")
REFUSAL_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "cannot assist",
    "can't assist",
    "unable to",
    "as an ai",
    "traceback",
    "exception",
    "error:",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL line") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def rate(count: int, total: int) -> float | None:
    return count / total if total else None


def is_refusal_or_error(raw_output: Any) -> bool:
    if raw_output is None:
        return True
    text = str(raw_output).strip()
    if not text:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def parsed_dict(row: dict[str, Any]) -> dict[str, Any]:
    parsed = row.get("parsed_output")
    return parsed if isinstance(parsed, dict) else {}


def normalize_activity(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in ACTIVITY_LABELS else None


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_success_rate(rows: Iterable[dict[str, Any]]) -> float | None:
    rows_list = list(rows)
    return rate(sum(bool(row.get("parse_success")) for row in rows_list), len(rows_list))


def evaluate_activity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    invalid_prediction_count = 0
    invalid_label_count = 0

    for row in rows:
        label = normalize_activity(row.get("label"))
        prediction = normalize_activity(parsed_dict(row).get("prediction"))
        if label is None:
            invalid_label_count += 1
            continue
        if prediction is None:
            invalid_prediction_count += 1
            prediction = "invalid_prediction"
        y_true.append(label)
        y_pred.append(prediction)
        confusion[label][prediction] += 1

    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    labels_with_invalid = list(ACTIVITY_LABELS) + ["invalid_prediction"]
    per_class: dict[str, dict[str, Any]] = {}
    for label in ACTIVITY_LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in ACTIVITY_LABELS if other != label)
        false_negative = sum(count for pred, count in confusion[label].items() if pred != label)
        support = sum(confusion[label].values())
        precision = rate(true_positive, true_positive + false_positive)
        recall = rate(true_positive, true_positive + false_negative)
        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "support": support,
            "predicted_count": sum(confusion[true][label] for true in ACTIVITY_LABELS),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    valid_f1 = [values["f1"] for values in per_class.values() if values["f1"] is not None]
    macro_f1_fallback = sum(valid_f1) / len(ACTIVITY_LABELS) if valid_f1 else None
    valid_precision = [
        values["precision"] for values in per_class.values() if values["precision"] is not None
    ]
    valid_recall = [values["recall"] for values in per_class.values() if values["recall"] is not None]
    result: dict[str, Any] = {
        "samples": len(rows),
        "parse_success_rate": parse_success_rate(rows),
        "evaluable_samples": len(y_true),
        "invalid_label_count": invalid_label_count,
        "invalid_prediction_count": invalid_prediction_count,
        "accuracy": rate(correct, len(y_true)),
        "macro_f1": macro_f1_fallback,
        "macro_precision": sum(valid_precision) / len(ACTIVITY_LABELS) if valid_precision else None,
        "macro_recall": sum(valid_recall) / len(ACTIVITY_LABELS) if valid_recall else None,
        "per_class": per_class,
        "per_class_counts": {
            label: {
                "true_count": sum(confusion[label].values()),
                "predicted_count": sum(confusion[true][label] for true in ACTIVITY_LABELS),
            }
            for label in ACTIVITY_LABELS
        },
        "confusion_counts": {
            label: {pred: confusion[label][pred] for pred in labels_with_invalid}
            for label in ACTIVITY_LABELS
        },
    }

    if y_true:
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score  # type: ignore

            result["macro_f1"] = float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=list(ACTIVITY_LABELS),
                    average="macro",
                    zero_division=0,
                )
            )
            result["macro_precision"] = float(
                precision_score(
                    y_true,
                    y_pred,
                    labels=list(ACTIVITY_LABELS),
                    average="macro",
                    zero_division=0,
                )
            )
            result["macro_recall"] = float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=list(ACTIVITY_LABELS),
                    average="macro",
                    zero_division=0,
                )
            )
        except Exception:
            pass

    return result


def pearson_correlation(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) < 2:
        return None
    try:
        from scipy.stats import pearsonr  # type: ignore

        correlation = float(pearsonr(actual, predicted)[0])
        return correlation if math.isfinite(correlation) else None
    except Exception:
        pass

    try:
        import numpy as np  # type: ignore

        correlation = float(np.corrcoef(actual, predicted)[0, 1])
        return correlation if math.isfinite(correlation) else None
    except Exception:
        return None


def spearman_correlation(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) < 2:
        return None
    try:
        from scipy.stats import spearmanr  # type: ignore

        correlation = float(spearmanr(actual, predicted)[0])
        return correlation if math.isfinite(correlation) else None
    except Exception:
        return None


def evaluate_pic50(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual: list[float] = []
    predicted: list[float] = []
    non_numeric_prediction_count = 0
    non_numeric_label_count = 0

    for row in rows:
        label = to_float(row.get("pIC50"))
        prediction = to_float(parsed_dict(row).get("predicted_pIC50"))
        if label is None:
            non_numeric_label_count += 1
            continue
        if prediction is None:
            non_numeric_prediction_count += 1
            continue
        actual.append(label)
        predicted.append(prediction)

    errors = [pred - label for label, pred in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors) if errors else None
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None

    return {
        "samples": len(rows),
        "parse_success_rate": parse_success_rate(rows),
        "evaluable_samples": len(actual),
        "numeric_prediction_count": len(predicted),
        "non_numeric_prediction_count": non_numeric_prediction_count,
        "non_numeric_label_count": non_numeric_label_count,
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson_correlation(actual, predicted),
        "spearman": spearman_correlation(actual, predicted),
    }


def smiles_is_valid(smiles: Any) -> bool | None:
    if smiles is None:
        return None
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return None

    try:
        return Chem.MolFromSmiles(str(smiles)) is not None
    except Exception:
        return False


def evaluate_molecule_optimization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(rows),
        "parse_success_rate": parse_success_rate(rows),
        "valid_smiles": None,
    }

    try:
        from rdkit import Chem  # noqa: F401  # type: ignore
    except Exception:
        return result

    per_sample: list[dict[str, Any]] = []
    counts = Counter()
    for row in rows:
        smiles = parsed_dict(row).get("edited_smiles")
        valid = smiles_is_valid(smiles)
        if valid is None:
            counts["missing"] += 1
        elif valid:
            counts["valid"] += 1
        else:
            counts["invalid"] += 1
        per_sample.append(
            {
                "id": row.get("id"),
                "edited_smiles": smiles,
                "valid": valid,
            }
        )

    result["valid_smiles"] = {
        "valid": counts["valid"],
        "invalid": counts["invalid"],
        "missing": counts["missing"],
        "per_sample": per_sample,
    }
    return result


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task", "unknown"))].append(row)

    metrics: dict[str, Any] = {
        "samples": len(rows),
        "parse_success_rate": parse_success_rate(rows),
        "refusal_or_error_count": sum(is_refusal_or_error(row.get("raw_output")) for row in rows),
        "tasks": {
            task: {
                "samples": len(task_rows),
                "parse_success_rate": parse_success_rate(task_rows),
                "refusal_or_error_count": sum(
                    is_refusal_or_error(row.get("raw_output")) for row in task_rows
                ),
            }
            for task, task_rows in sorted(by_task.items())
        },
    }

    if "target_activity_prediction" in by_task:
        metrics["tasks"]["target_activity_prediction"].update(
            evaluate_activity(by_task["target_activity_prediction"])
        )
    if "pic50_prediction" in by_task:
        metrics["tasks"]["pic50_prediction"].update(evaluate_pic50(by_task["pic50_prediction"]))
    if "molecule_optimization" in by_task:
        metrics["tasks"]["molecule_optimization"].update(
            evaluate_molecule_optimization(by_task["molecule_optimization"])
        )

    return metrics


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate JSONL model predictions.")
    parser.add_argument("--predictions", required=True, type=Path, help="Path to predictions JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Path to output metrics JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(args.predictions)
    metrics = evaluate(rows)
    write_metrics(metrics, args.output)
    print(f"Wrote metrics for {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
