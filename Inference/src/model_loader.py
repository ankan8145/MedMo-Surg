from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import InferenceConfig


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype '{name}'")


def load_model_and_tokenizer(config: InferenceConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=_torch_dtype(config.torch_dtype),
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()
    return model, tokenizer
