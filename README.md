# MedMO-surge

HEICO-FOCUS fine-tuning of the MedMO 9B vision-language model (supervised fine-tuning on frame- and segment-level VQA), built on a fork of [TRL](https://github.com/huggingface/trl).

## Checkpoint

Hugging Face: [`ankanmbz/Medmo-Surg`](https://huggingface.co/ankanmbz/Medmo-Surg) (public, image-text-to-text, base model `MBZUAI/MedMO-8B-Next`).

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained("ankanmbz/Medmo-Surg", torch_dtype="auto")
processor = AutoProcessor.from_pretrained("ankanmbz/Medmo-Surg")
```

## Layout

```
trl/
├── configs/heico_focus_sft.yaml       # SFT hyperparameters
├── scripts/train_heico_focus_sft.sh   # launcher (smoke | train)
└── trl/                               # TRL library (scripts/sft.py is the entry point)
Inference/                             # separate inference/eval workspace (see Inference/README.md)
```

## Data

| Dataset | Train | Test |
|---|---|---|
| `heico_focus_frame_vqa` | `annotations/frame/train.csv` | `annotations/frame/test.csv` |
| `heico_focus_segment_vqa` | `annotations/segment/train.csv` | `annotations/segment/test.csv` |

Merged: 16,000 train / 8,000 test rows. Test rows are never added to training. The dataset is not included in this repo; place it under `trl/data/heico_focus/` (or set `DATASET_DIR`).

## Setup

```bash
cd trl
pip install -e ".[deepspeed,peft]"
```

Put the starting MedMO 9B SFT checkpoint at `trl/checkpoints/MedMO-9B-SFT` (or set `MODEL_PATH`).

## Train

```bash
cd trl
scripts/train_heico_focus_sft.sh smoke   # single optimizer step sanity check
scripts/train_heico_focus_sft.sh train   # 1 epoch + end-of-epoch test evaluation
```

Overridable env vars: `MODEL_PATH`, `DATASET_DIR`, `OUTPUT_DIR`, `CONFIG`, `ACCELERATE_CONFIG`, `NUM_PROCESSES`, `MAIN_PROCESS_PORT`. Outputs go to `trl/outputs/`.

Default setup: ZeRO-3, bf16, 8 GPUs, 1 epoch.

## Notes

The dataset registration code (`heico_focus_*`) and the original launcher were not in the source folder, so this launcher feeds the dataset through TRL's stock `sft.py` via `--dataset_name`. You may need to adapt the data loading to your CSV columns.
