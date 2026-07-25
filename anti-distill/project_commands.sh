uv sync
source .venv/bin/activate

hf download VoCuc/anti-data \
  --repo-type dataset \
  --local-dir .
unzip -o anti_data.zip


bash nothing1/anti-distill/script/train/train_teacher_lora.sh &
bash nothing1/anti-distill/script/train/train_no_virtual_ascent.sh &
bash nothing1/anti-distill/script/train/train_no_loss_base.sh &
bash nothing1/anti-distill/script/train/train_no_conflict.sh &

wait