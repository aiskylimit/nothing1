source .venv/bin/activate
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

python scripts/extract-doge-model-checkpoint.py \
  --model_dir="outputs/qwen7b-doge-coef0.001-temp2-head_proj0-epoch1-lr5e-5/checkpoint-20"
