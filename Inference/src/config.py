from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODEL_PATH = "checkpoints/Qwen3.5-9B"


@dataclass
class InferenceConfig:
    model_path: str = DEFAULT_MODEL_PATH
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"


def load_config(path: str | Path) -> InferenceConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    model = data.get("model", {})
    generation = data.get("generation", {})

    return InferenceConfig(
        model_path=str(model.get("model_path", DEFAULT_MODEL_PATH)),
        trust_remote_code=bool(model.get("trust_remote_code", True)),
        torch_dtype=str(model.get("torch_dtype", "bfloat16")),
        device_map=str(model.get("device_map", "auto")),
        max_new_tokens=int(generation.get("max_new_tokens", 512)),
        temperature=float(generation.get("temperature", 0.0)),
        top_p=float(generation.get("top_p", 1.0)),
        do_sample=bool(generation.get("do_sample", False)),
    )
