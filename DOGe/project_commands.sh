uv sync
source .venv/bin/activate

RAW="TwXoDncyxhVSoTFklwVXpsPaXzVipMJavD"
export HF_TOKEN="hf_${RAW}"
hf auth login --token "hf_${RAW}"

mkdir -p data/r1-qwen-7b

wget -c \
  "https://huggingface.co/datasets/VoCuc/anti-data/resolve/main/distillation_data.jsonl?download=true" \
  -O data/r1-qwen-7b/distillation_data.jsonl

bash train-doge.sh
ls outputs -R