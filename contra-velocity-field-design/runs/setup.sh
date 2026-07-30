BASE_PATH=${1-"."}
MAX_LENGTH=512
export HF_HOME="/mnt/data/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1

bash ${BASE_PATH}/install.sh

python3 tools/get_openwebtext.py

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/openwebtext \
    --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/gpt2/${MAX_LENGTH}/ \
    --model-path gpt2 \
    --max-length 512 \
    --train-num 22870 \
    --data-process-workers 32 \
    --dev-num 1000 \

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir hf://dvtiendat/contra-data/dolly \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path gpt2 \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type gpt2

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/openwebtext \
    --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/openllama2/512/ \
    --model-path openlm-research/open_llama_3b_v2 \
    --max-length 512 \
    --train-num 22870 \
    --data-process-workers 32 \
    --dev-num 1000 \

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir hf://dvtiendat/contra-data/dolly \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path openlm-research/open_llama_3b_v2 \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type openllama2

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/openwebtext \
    --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/llama2/512/ \
    --model-path meta-llama/Llama-2-7b-hf \
    --max-length 512 \
    --train-num 22870 \
    --data-process-workers 32 \
    --dev-num 1000 \

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir hf://dvtiendat/contra-data/dolly \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path meta-llama/Llama-2-7b-hf \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type llama2
