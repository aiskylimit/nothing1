#! /bin/bash

BASE_PATH=${1-"."}
MASTER_PORT=${2-2012}
NUM_GPUS=${3-1}

# Train the cross-tokenizer velocity field first.
bash ${BASE_PATH}/scripts/gpt2/train_velocity_field_qwen_0.1B_1.8B.sh ${BASE_PATH} ${MASTER_PORT} ${NUM_GPUS}

# Then run DistiLLM + CONTRA + SRA alignment.
bash ${BASE_PATH}/scripts/gpt2/distillm/contra_qwen_0.1B_1.8B.sh ${BASE_PATH} ${MASTER_PORT} ${NUM_GPUS}

echo "Model: ${BASE_PATH}/results/gpt2/train/contra/distillm/qwen1.5-1.8b_0.1B"
