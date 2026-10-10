from __future__ import annotations

import json
from typing import Any


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced JSON object found in text."""
    if not text:
        return None

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_model_output(raw_output: str) -> dict[str, Any]:
    parsed = extract_first_json_object(raw_output)
    if parsed is None:
        return {
            "parse_success": False,
            "parsed_output": None,
            "raw_output": raw_output,
        }
    return {
        "parse_success": True,
        "parsed_output": parsed,
        "raw_output": raw_output,
    }
