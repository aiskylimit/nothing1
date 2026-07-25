#! /bin/bash

GPUS=(1 5)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=.

STUDENT_CKPT_NAME="Qwen"
STUDENT_CKPT="Qwen/Qwen2.5-1.5B"
TEACHER_CKPT_NAME="DeepSeek"
TEACHER_CKPT="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# data
DATA_DIR="${BASE_PATH}/processed_data/MetaMathQA-50k/qwen/"
# hp
BATCH_SIZE=8
LR=0.0003
GRAD_ACC=1
EVAL_BATCH_SIZE=64
EPOCHS=5
# length
MAX_LENGTH=640
# runtime
SAVE_PATH="${BASE_PATH}/results/finetune_no_virtual_ascent"
# seed
SEED=42


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${TEACHER_CKPT}"
OPTS+=" --student-model-path ${STUDENT_CKPT}"
OPTS+=" --ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --student-ckpt-name ${STUDENT_CKPT_NAME}"
OPTS+=" --student-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-ratio 0.05"
OPTS+=" --lr-decay-style wrmup_cosine"
# OPTS+=" --lr-decay-style constant"
# OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.03"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio 1.0"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 128"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 20"
OPTS+=" --mid-log-num -1"

OPTS+=" --save ${SAVE_PATH}"
# lora
OPTS+=" --do-train"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# type
OPTS+=" --type fkl"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"


OPTS+=" --temperature 0.7"
OPTS+=" --delta-lamda 0.1"
OPTS+=" --student-lr 0.001"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_no_virtual_ascent.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}

bash ./script/eval/gsm8k/run_eval_1.sh