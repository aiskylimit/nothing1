#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=${1-"."}
CKPT_NAME="gpt2-base"
CKPT=""
TEACHER_CKPT_NAME="qwen1.5-1.8b"
TEACHER_CKPT=""
TEACHER_TOKENIZER=""
VELOCITY_FIELD_CKPT="${BASE_PATH}/results/gpt2/train/velocity_field/qwen1.5-1.8b_0.1B/velocity_field.pth"
PROJECTOR_CKPT="${BASE_PATH}/results/gpt2/train/velocity_field/qwen1.5-1.8b_0.1B/projector.pth"
DATA_DIR="hf://dvtiendat/contra-data/dolly"
BATCH_SIZE=16
LR=0.0005
GRAD_ACC=1
EVAL_BATCH_SIZE=16
MAX_LENGTH=512
SAVE_PATH="${BASE_PATH}/results/gpt2/train/contra/distillm/qwen1.5-1.8b_0.1B"
SEED=10

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --model-type gpt2"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-model-type qwen"
OPTS+=" --teacher-tokenizer-path ${TEACHER_TOKENIZER}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --gradient-checkpointing"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --gt-data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 20"
OPTS+=" --kd-ratio 1.0"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
OPTS+=" --type srkl-contra"
OPTS+=" --skew-alpha 0.1"
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
OPTS+=" --cross-tokenizer-sra"
OPTS+=" --sra-loss-weight 1.0"
OPTS+=" --velocity-n-layers 4"
OPTS+=" --velocity-d-model 1200"
OPTS+=" --d-teacher 2048"
OPTS+=" --d-student 768"
OPTS+=" --num-distill-layers 6"
OPTS+=" --num-teacher-layers 24"
OPTS+=" --num-student-layers 12"
OPTS+=" --hidden-loss-weights 1 1 2 2 3 3"
OPTS+=" --geom-loss-weight 50"
OPTS+=" --teacher-device 0"
OPTS+=" --student-device 0"
OPTS+=" --velocity-field-path ${VELOCITY_FIELD_CKPT}"
OPTS+=" --projector-path ${PROJECTOR_CKPT}"
OPTS+=" --velocity-epochs 1"
OPTS+=" --velocity-update-interval 1"

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_cross_tokenizer.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
