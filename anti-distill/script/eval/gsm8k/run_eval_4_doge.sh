export CUDA_VISIBLE_DEVICES=4

teacher="../DOGe/outputs/qwen7b-doge-coef0.001-temp2-head_proj0-epoch1-lr5e-5/checkpoint-49"

pairs=(
    "0.0 0.9"
    "0.0 1.16"
    "0.0 1.26"
    "0.0 1.5"
)

for pair in "${pairs[@]}"; do
    read delta tau <<< "$pair"

    echo "=========================================================================="
    echo " START PIPELINE WITH DELTA = ${delta} AND TAU = ${tau}"
    echo "=========================================================================="

    echo ">>> [1/4] Eval Teacher with delta = ${delta} and tau = ${tau}..."
    accelerate launch --config_file acc_config_4.yaml --main_process_port 0 student_eval.py  \
    hydra.run.dir=experiments_gsm8k_doge/metadata/eval/teacher_tau${tau}_delta${delta} \
    is_teacher=true exp_dir=experiments_gsm8k_doge answer_force=true tau=${tau} \
    data_split=gsm8k_test batch_size=128 trace_name=eval_teacher_lora_tau${tau}_delta${delta} seed=42 \
    tokenizer=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    teacher=${teacher} delta=${delta} max_samples=2880 

    echo ">>> [2/4] Gen Training Traces with delta = ${delta} and tau = ${tau}..."
    accelerate launch --config_file acc_config_4.yaml --main_process_port 0 student_eval.py \
    trace_name=teacher_lora_tau${tau}_delta${delta} \
    trace_path=./experiments_gsm8k_doge/traces_gsm8k/teacher_lora_tau${tau}_delta${delta} \
    data_split=gsm8k_train \
    teacher=${teacher} \
    tokenizer=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    batch_size=128 \
    max_length=1024 \
    max_prompt_length=512  \
    answer_force=true \
    tau=${tau} seed=42 delta=${delta}

    echo ">>> [3/4] Training Student model with delta = ${delta} and tau = ${tau}..."
    accelerate launch --config_file acc_config_4.yaml --main_process_port 0 distill.py \
    hydra.run.dir=experiments_gsm8k_doge/metadata/distill/lora_teacher \
    student=google/gemma-2b-it \
    tokenizer=google/gemma-2b-it \
    exp_dir=experiments_gsm8k_doge \
    train_traces=experiments_gsm8k_doge/traces_gsm8k/teacher_lora_tau${tau}_delta${delta} \
    holdout_traces=traces_holdout \
    model_name=student_manua_tau${tau}_delta${delta} max_length=1025

    echo ">>> [4/4] Eval Student model with delta = ${delta} and tau = ${tau}..."
    accelerate launch --config_file acc_config_4.yaml --main_process_port 0 student_eval.py  \
    hydra.run.dir=experiments_gsm8k_doge/metadata/eval/student_gsm8k \
    teacher=experiments_gsm8k_doge/models/student_manua_tau${tau}_delta${delta}/final \
    is_teacher=false exp_dir=experiments_gsm8k_doge answer_force=true \
    data_split=gsm8k_test batch_size=128 max_samples=2880 \
    trace_name=eval_student_gsm8k_tau${tau}_delta${delta} seed=42

done

echo "🎉 DONE!"
