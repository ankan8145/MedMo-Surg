# MedMO Qwen Inference

This is a small inference and evaluation workspace for testing the local fine-tuned Qwen checkpoint on medical, molecular, and drug-discovery reasoning tasks.

Checkpoint path:

```bash
checkpoints/Qwen3.5-9B
```

The code only performs inference, JSON parsing, prediction saving, and lightweight evaluation. It does not train or modify the checkpoint.

## Setup

Install the required runtime packages in the Python environment you plan to use for GPU inference:

```bash
pip install -r requirements.txt
```

Optional evaluation packages such as `scikit-learn`, `scipy`, `numpy`, and `rdkit` are used only if installed.

## Input JSONL

Inputs are JSONL files with one JSON object per line. Common fields:

- `id`: stable sample id
- `task`: supported task name
- `smiles`: molecule SMILES when applicable
- `target_name`: target name when applicable
- `target_sequence`: optional target sequence
- `question`: biomedical reasoning question when applicable
- `label`: optional activity label for evaluation
- `pIC50`: optional numeric pIC50 label for evaluation

Supported tasks:

- `molecule_understanding`
- `target_activity_prediction`
- `pic50_prediction`
- `molecule_optimization`
- `drug_reasoning`

See `data/sample_inputs.jsonl` for examples.

## Prepare Benchmark Data

Download programmatic real benchmark datasets and create default test files:

```bash
bash scripts/download_real_datasets.sh
```

This creates:

- `data/raw/`
- `data/processed/`
- `data/test_activity.jsonl`
- `data/test_pic50.jsonl`

The downloader tries TDC DAVIS, KIBA, and BindingDB_IC50 first through `tdc` if installed, then through official TDC Dataverse file IDs. If those fail, it continues to MoleculeNet BACE/HIV from DeepChem-hosted CSVs. In the current environment, `PyTDC` is optional because installing it may alter the working Transformers stack.

Convert a raw CSV or JSONL file into the inference JSONL schema:

```bash
python -m src.dataset_prepare \
  --input raw_file.csv \
  --output data/test_activity.jsonl \
  --task target_activity_prediction \
  --limit 1000
```

Equivalent shell wrapper:

```bash
bash scripts/prepare_dataset.sh \
  --input raw_file.csv \
  --output data/test_activity.jsonl \
  --task target_activity_prediction \
  --limit 1000
```

Common input columns are detected automatically:

- SMILES: `smiles`, `canonical_smiles`, `molecule_smiles`, `compound_smiles`
- Target name: `target_name`, `target`, `protein_name`, `gene_name`
- Protein sequence: `target_sequence`, `sequence`, `protein_sequence`
- Activity label: `label`, `activity_label`, `active`, `class`
- pIC50: `pIC50`, `pic50`, `pchembl_value`, `pchembl`

Essential fields:

- All molecule tasks need `smiles`.
- Target-aware tasks need `target_name` or `target_sequence`.
- `target_activity_prediction` needs `label`.
- `pic50_prediction` needs `pIC50`.

Rows missing essential fields are skipped and logged.

## Held-Out Evaluation

Activity classification expects `data/test_activity.jsonl`:

```bash
bash scripts/run_activity_eval.sh --max-new-tokens 256
```

Outputs:

- `outputs/predictions_activity.jsonl`
- `outputs/metrics_activity.json`
- `outputs/analysis_activity/`

pIC50 regression expects `data/test_pic50.jsonl`:

```bash
bash scripts/run_pic50_eval.sh --max-new-tokens 256
```

Outputs:

- `outputs/predictions_pic50.jsonl`
- `outputs/metrics_pic50.json`
- `outputs/analysis_pic50/`

Inspect analysis files:

```bash
ls outputs/analysis_activity
head -n 5 outputs/analysis_activity/failed_parse.jsonl
head -n 5 outputs/analysis_activity/wrong_predictions.jsonl
head -n 5 outputs/analysis_activity/uncertain_predictions.jsonl
cat outputs/analysis_activity/summary.json
```

For pIC50, inspect `outputs/analysis_pic50/large_error_predictions.jsonl` for rows with absolute error greater than 1.0.

## Run Inference

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate corl
export PYTHONNOUSERSITE=1

cd Inference

python -m src.inference \
  --config configs/inference_config.yaml \
  --input data/sample_inputs.jsonl \
  --output outputs/predictions.jsonl \
  --model-path checkpoints/Qwen3.5-9B
```

Or:

```bash
bash scripts/run_inference.sh \
  --model-path checkpoints/Qwen3.5-9B
```

For a shorter smoke test, add `--max-new-tokens 128`.

Each output line contains the sample id, task, rendered prompt, raw model output, parsed JSON output, parse status, and any available labels.

## Run Evaluation

```bash
cd Inference

python -m src.metrics \
  --predictions outputs/predictions.jsonl \
  --output outputs/metrics.json
```

Or:

```bash
bash scripts/eval_outputs.sh
```

Evaluation writes aggregate metrics to `outputs/metrics.json`.

## Optional Tools

The `src.tools` package contains separate wrappers for RDKit descriptors and selected public biomedical APIs. These are not used by baseline model inference.

Current model inference is text-only local generation. External tools are not called inside generation yet.

List registered tools:

```bash
python -m src.tools.tool_registry --list
```

Run tool smoke tests:

```bash
bash scripts/test_tools.sh
```

External API tests require outbound HTTPS access from the server. If the network or proxy blocks a service, the tool returns a structured error instead of crashing.

## Outputs

- `outputs/predictions.jsonl`: one prediction JSON object per input row
- `outputs/metrics.json`: parse rates and task-specific metrics

## Troubleshooting

CUDA memory:

- Reduce `generation.max_new_tokens` in `configs/inference_config.yaml`.
- Keep `model.device_map: auto`.
- Use `bfloat16` or `float16` when supported by your GPU.
- Make sure no unrelated GPU jobs are holding memory.

Tokenizer and chat template:

- The tokenizer and model are both loaded from the same local checkpoint.
- If the tokenizer has a chat template, inference uses it automatically with Qwen thinking disabled for JSON-only output.
- If outputs look malformed, inspect the saved `prompt` field in `outputs/predictions.jsonl`.

Python packages:

- If `transformers` imports packages from `~/.local`, run with `export PYTHONNOUSERSITE=1`.
- Install runtime dependencies inside the active conda environment with `python -m pip install -r requirements.txt`.

JSON parsing:

- Prompts request strict JSON, but model text can still include extra prose.
- `src.parser` extracts the first valid JSON object from each raw output.
- Failed parses are saved with `parse_success: false`, `parsed_output: null`, and the original `raw_output`.
