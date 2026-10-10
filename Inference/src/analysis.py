from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.metrics import normalize_activity, parsed_dict, to_float
from src.utils import read_jsonl, write_json, write_jsonl


def analyze_predictions(predictions_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_parse: list[dict[str, Any]] = []
    wrong_predictions: list[dict[str, Any]] = []
    uncertain_predictions: list[dict[str, Any]] = []
    large_error_predictions: list[dict[str, Any]] = []

    for row in rows:
        parsed = parsed_dict(row)
        if not row.get("parse_success"):
            failed_parse.append(row)

        if row.get("task") == "target_activity_prediction":
            label = normalize_activity(row.get("label"))
            prediction = normalize_activity(parsed.get("prediction"))
            if prediction == "uncertain":
                uncertain_predictions.append(row)
            if label is not None and prediction is not None and prediction != label:
                wrong_predictions.append(row)
            elif label is not None and prediction is None:
                wrong_predictions.append(row)

        if row.get("task") == "pic50_prediction":
            label_value = to_float(row.get("pIC50"))
            prediction_value = to_float(parsed.get("predicted_pIC50"))
            if label_value is not None and prediction_value is not None:
                abs_error = abs(prediction_value - label_value)
                if abs_error > 1.0:
                    enriched = dict(row)
                    enriched["absolute_error"] = abs_error
                    large_error_predictions.append(enriched)

    write_jsonl(output_dir / "failed_parse.jsonl", failed_parse)
    write_jsonl(output_dir / "wrong_predictions.jsonl", wrong_predictions)
    write_jsonl(output_dir / "uncertain_predictions.jsonl", uncertain_predictions)
    write_jsonl(output_dir / "large_error_predictions.jsonl", large_error_predictions)

    summary = {
        "predictions": str(predictions_path),
        "samples": len(rows),
        "failed_parse_count": len(failed_parse),
        "wrong_prediction_count": len(wrong_predictions),
        "uncertain_prediction_count": len(uncertain_predictions),
        "large_error_prediction_count": len(large_error_predictions),
        "outputs": {
            "failed_parse": str(output_dir / "failed_parse.jsonl"),
            "wrong_predictions": str(output_dir / "wrong_predictions.jsonl"),
            "uncertain_predictions": str(output_dir / "uncertain_predictions.jsonl"),
            "large_error_predictions": str(output_dir / "large_error_predictions.jsonl"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prediction JSONL outputs.")
    parser.add_argument("--predictions", required=True, type=Path, help="Predictions JSONL file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for analysis files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_predictions(args.predictions, args.output_dir)
    print(f"Wrote analysis for {summary['samples']} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
