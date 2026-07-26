uv sync
source .venv/bin/activate

RAW="TwXoDncyxhVSoTFklwVXpsPaXzVipMJavD"
export HF_TOKEN="hf_${RAW}"
hf auth login --token "hf_${RAW}"

# hf download VoCuc/anti-data \
#   --repo-type dataset \
#   --local-dir .
# unzip -o anti_data.zip


# bash ./script/train/train_teacher_lora.sh &
# bash ./script/train/train_no_virtual_ascent.sh &
# bash ./script/train/train_no_loss_base.sh &
# bash ./script/train/train_no_conflict.sh &

bash ./script/eval/gsm8k/run_eval_0.sh &
bash ./script/eval/gsm8k/run_eval_1.sh &
bash ./script/eval/gsm8k/run_eval_2.sh &
bash ./script/eval/gsm8k/run_eval_3.sh &

wait