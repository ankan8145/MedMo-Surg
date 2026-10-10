from __future__ import annotations

import json
from typing import Any


SUPPORTED_TASKS = {
    "molecule_understanding",
    "target_activity_prediction",
    "pic50_prediction",
    "molecule_optimization",
    "drug_reasoning",
}

EVALUATION_ONLY_FIELDS = {
    "activity_label",
    "activity_value",
    "answer",
    "answers",
    "expected_answer",
    "expected_output",
    "gold",
    "gold_answer",
    "ground_truth",
    "label",
    "pIC50",
    "reference",
}


SYSTEM_PROMPT = (
    "You are a careful medical and molecular reasoning assistant. "
    "Return only one valid JSON object matching the requested schema. "
    "Do not include markdown, prose outside JSON, or unsupported claims. "
    "Keep the JSON compact so the full object fits in the generation budget."
)


SCHEMAS: dict[str, dict[str, Any]] = {
    "target_activity_prediction": {
        "prediction": "active/inactive/uncertain",
        "confidence": "low/medium/high",
        "reason": "...",
        "limitations": "...",
    },
    "pic50_prediction": {
        "predicted_pIC50": 0.0,
        "confidence": "low/medium/high",
        "reason": "...",
        "limitations": "...",
    },
    "molecule_understanding": {
        "validity_assumption": "...",
        "functional_groups": [],
        "drug_likeness": "...",
        "possible_risks": [],
        "short_explanation": "...",
    },
    "molecule_optimization": {
        "edited_smiles": "...",
        "modification_reason": "...",
        "expected_benefit": "...",
        "risks": "...",
        "limitations": "...",
    },
    "drug_reasoning": {
        "answer": "...",
        "evidence_summary": "...",
        "risks": [],
        "limitations": "...",
    },
}


def build_prompt(sample: dict[str, Any]) -> tuple[str, str]:
    task = str(sample.get("task", "")).strip()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}")

    task_instructions = {
        "molecule_understanding": (
            "Analyze the molecule at a high level for drug-discovery screening. "
            "Do not claim experimental validation unless provided."
        ),
        "target_activity_prediction": (
            "Predict whether the molecule is active, inactive, or uncertain against the target. "
            "Use the target name, target sequence if present, and SMILES."
        ),
        "pic50_prediction": (
            "Estimate pIC50 for the molecule-target pair. "
            "Return a numeric predicted_pIC50 value."
        ),
        "molecule_optimization": (
            "Suggest one conservative SMILES edit intended to improve the molecule for the stated target or goal. "
            "Do not invent synthesis guarantees."
        ),
        "drug_reasoning": (
            "Answer the biomedical drug-discovery question using cautious reasoning. "
            "Surface uncertainty and risks."
        ),
    }

    schema = json.dumps(SCHEMAS[task], indent=2)
    prompt_sample = {key: value for key, value in sample.items() if key not in EVALUATION_ONLY_FIELDS}
    sample_text = json.dumps(prompt_sample, ensure_ascii=False, indent=2)
    user_prompt = (
        f"Task: {task}\n"
        f"Instruction: {task_instructions[task]}\n\n"
        f"Input JSON:\n{sample_text}\n\n"
        f"Return exactly one compact JSON object with this schema:\n{schema}\n\n"
        "Constraints: keep string values to one short sentence, use at most three array items, "
        "and close the JSON object completely."
    )
    return SYSTEM_PROMPT, user_prompt
