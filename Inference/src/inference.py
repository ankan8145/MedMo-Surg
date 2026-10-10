from __future__ import annotations

import argparse
import logging
from typing import Any

from src.config import load_config
from src.parser import parse_model_output
from src.prompts import build_prompt
from src.utils import read_jsonl, setup_logging, write_jsonl


def format_chat(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return system_prompt + "\n\n" + user_prompt


def generate_one(model: Any, tokenizer: Any, prompt_text: str, config: Any) -> str:
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    input_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_inference(
    config_path: str,
    input_path: str,
    output_path: str,
    model_path: str | None = None,
    max_new_tokens: int | None = None,
) -> None:
    from tqdm import tqdm

    from src.model_loader import load_model_and_tokenizer

    config = load_config(config_path)
    if model_path is not None:
        config.model_path = model_path
    if max_new_tokens is not None:
        config.max_new_tokens = max_new_tokens
    samples = read_jsonl(input_path)
    logging.info("Loaded %d samples from %s", len(samples), input_path)
    logging.info("Loading model from %s", config.model_path)
    model, tokenizer = load_model_and_tokenizer(config)

    rows: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="Generating"):
        system_prompt, user_prompt = build_prompt(sample)
        prompt_text = format_chat(tokenizer, system_prompt, user_prompt)
        raw_output = generate_one(model, tokenizer, prompt_text, config)
        parsed = parse_model_output(raw_output)

        row = {
            "id": sample.get("id"),
            "task": sample.get("task"),
            "prompt": prompt_text,
            "raw_output": parsed["raw_output"],
            "parsed_output": parsed["parsed_output"],
            "parse_success": parsed["parse_success"],
        }
        if "label" in sample:
            row["label"] = sample["label"]
        if "pIC50" in sample:
            row["pIC50"] = sample["pIC50"]
        rows.append(row)

    write_jsonl(output_path, rows)
    logging.info("Saved %d predictions to %s", len(rows), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen inference on JSONL tasks.")
    parser.add_argument("--config", required=True, help="Path to inference_config.yaml")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local checkpoint path. Overrides model.model_path from the YAML config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional generation limit. Overrides generation.max_new_tokens from the YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_inference(
        args.config,
        args.input,
        args.output,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
