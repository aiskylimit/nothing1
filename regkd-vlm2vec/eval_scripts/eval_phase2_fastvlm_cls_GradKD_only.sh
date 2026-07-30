#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
if [[ ! -d "$REPO_ROOT/src/data" ]]; then
    echo "Missing $REPO_ROOT/src/data. This checkout is incomplete; sync or commit the src/data package before evaluation." >&2
    exit 1
fi

NAME_CHECKPOINT=gvendi_phase2_cls_fastvlm_gradkd_only

python eval_mmeb.py \
    --model_name ./training/${NAME_CHECKPOINT}/checkpoint-final\
    --encode_output_path ./MMEB-evaloutputs/${NAME_CHECKPOINT}/ \
    --lora --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split test \
    --per_device_eval_batch_size 32 \
    --image_dir "VLM_Embed/eval_images" \
    --tgt_prefix_mod
