#!/usr/bin/env bash
set -euo pipefail

BASE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MASTER_PORT=${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")}
GPUS_PER_NODE=1
CUDA_VISIBLE_DEVICES=2
HF_HOME=${HF_HOME:-"${BASE_PATH}/.cache/huggingface"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"no-v-theta"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}

TRAIN_CKPT="contra/distillm/0.1B_1.5B"
RESULT_DIR="${BASE_PATH}/results/gpt2/train/${TRAIN_CKPT}"
LOG_DIR="${RESULT_DIR}/run_logs"
EVAL_DIR="${BASE_PATH}/results/gpt2/eval_main"
PROJECTOR_CKPT="${BASE_PATH}/results/gpt2/train/velocity_field/projector.pth"

export BASE_PATH
export CUDA_VISIBLE_DEVICES
export HF_HOME
export HF_HUB_DISABLE_TELEMETRY=1
export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}"

conda_base=$(conda info --base)
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
if conda tos --help >/dev/null 2>&1; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
fi
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${CONDA_ENV_NAME}"

mkdir -p "${LOG_DIR}"

bash "${BASE_PATH}/install.sh" 2>&1 | tee "${LOG_DIR}/setup.log"

if [[ ! -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  conda install -y -c nvidia -c conda-forge cuda-toolkit=13.0
fi
export CUDA_HOME="${CONDA_PREFIX}"

python3 "${BASE_PATH}/tools/get_openwebtext.py" 2>&1 | tee "${LOG_DIR}/get_openwebtext.log"

bash "${BASE_PATH}/scripts/gpt2/tools/process_data_dolly.sh" \
  "${BASE_PATH}" 2>&1 | tee "${LOG_DIR}/process_dolly.log"

bash "${BASE_PATH}/scripts/gpt2/tools/process_data_pretrain.sh" \
  "${BASE_PATH}" 2>&1 | tee "${LOG_DIR}/process_pretrain.log"

# hf download bachthetrollface/velocity-field-gpt2 \
#   --local-dir "${BASE_PATH}/results/gpt2/train/velocity_field/"
# if [[ ! -f "${PROJECTOR_CKPT}" ]]; then
#   echo "Missing projector checkpoint: ${PROJECTOR_CKPT}" >&2
#   echo "The new target does not use v_theta, but GPT-2 0.1B -> 1.5B still needs this projector for hidden-size alignment." >&2
#   exit 1
# fi

# bash "${BASE_PATH}/scripts/gpt2/distillm/contra_0.1B_1.5B.sh" \
#   "${BASE_PATH}" \
#   "${MASTER_PORT}" \
#   "${GPUS_PER_NODE}" 2>&1 | tee "${LOG_DIR}/train.log"

# bash "${BASE_PATH}/scripts/gpt2/eval/run_eval.sh" \
#   "${CUDA_VISIBLE_DEVICES}" \
#   "${TRAIN_CKPT}" 2>&1 | tee "${LOG_DIR}/eval.log"

bash "${BASE_PATH}/runs/gpt2/distillm_contra.sh" \
  "${BASE_PATH}" \
  "${MASTER_PORT}" 2>&1 | tee "${LOG_DIR}/eval.log"

# mkdir -p "${RESULT_DIR}/eval_main"
# cp -R "${EVAL_DIR}/." "${RESULT_DIR}/eval_main/"