# CALA

This repository contains research code for Conflict-Aware Logit Adapters (CALA), a lightweight training-time defense against distillation attacks on reasoning models.

The code in this repository is a partial research implementation of CALA, adapted from the Antidistillation Sampling codebase.

## What this code contains

- `finetune.py`: train a logits adapter for the teacher model
- `new_gentraces.py`: generate CALA-perturbed reasoning traces
- `distill.py`: train a student model on generated traces
- `student_eval.py`: evaluate student performance
- `logit_adapter.py`: define the adapter used by the teacher model
- `utils.py`: dataset loading and helper utilities

## Installation

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Run `uv sync` to install dependencies

## Quick Start
The experiments in this repository were developed and tested on a single H100 GPU (1xH100) setup.

Activate the virtual environment:
```bash
source .venv/bin/activate
```

Train the teacher logits adapter:

```bash
bash script/train/train_teacher_lora.sh
```

Run the GSM8K evaluation flow:

```bash
bash script/eval/gsm8k/run_eval_0.sh
```

## Pipeline flow

### Stage 1: Train teacher logits adapter
- Run `script/train/train_teacher_lora.sh`
- Uses `finetune.py` to fit a small adapter on the teacher model
- Prepares the teacher for CALA-style trace generation

### Stage 2: Generate CALA traces
- Run `new_gentraces.py`
- Wraps the teacher model with `LogitAdapter`
- Generates traces with `delta` controlling the adapter perturbation and `tau` controlling sampling temperature
- Optional `answer_force=true` improves final answers

### Stage 3: Distill the student
- Use `distill.py` to train the student model on the generated poisoned traces
- Student training is supervised fine-tuning on completion-only targets

### Stage 4: Evaluate the student
- Use `student_eval.py` to measure student accuracy on downstream data
- Compare student performance against the defended teacher traces

### Stage 5: Evaluate the teacher
- Re-run `new_gentraces.py` for teacher evaluation
- Verifies that teacher quality remains stable under CALA settings

## Important files and config

- `gen_config.yaml`: Hydra config for `new_gentraces.py`
- `train_config.yaml`: Hydra config for `distill.py`
- `acc_config.yaml`: Accelerate config for distributed launches
- `configs/`: DeepSpeed and distributed configuration files
- `script/train/`: training scripts
- `script/eval/`: evaluation scripts
