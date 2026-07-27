export CUDA_VISIBLE_DEVICES=2

gen="new_gentraces.py"
teacher="./results/DeepSeek-R1-Distill-Qwen-7B/3841"

pairs=(
    # "0.25 0.5"
    "0.75 0.75"
    "0.8 0.5"
    "0.85 0.5"
)

for pair in "${pairs[@]}"; do
    read delta tau <<< "$pair"

    echo "=========================================================================="
    echo " START PIPELINE WITH DELTA = ${delta} AND TAU = ${tau}"
    echo "=========================================================================="

    # echo ">>> [1/4] Eval Teacher with delta = ${delta} and tau = ${tau}..."
    # accelerate launch --config_file acc_config_2.yaml --main_process_port 0 "$gen"  \
    # hydra.run.dir=experiments_gpqa_cala/metadata/eval/lora_teacher_tau${tau}_delta${delta} \
    # is_teacher=true exp_dir=experiments_gpqa_cala answer_force=true tau=${tau} \
    # data_split=gpqa_test batch_size=256 trace_name=eval_teacher_lora_tau${tau}_delta${delta} seed=42 \
    # tokenizer=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    # teacher=${teacher} delta=${delta} max_samples=2880 max_length=1024 max_prompt_length=256

    # echo ">>> [2/4] Gen Training Traces with delta = ${delta} and tau = ${tau}..."
    # python "$gen" \
    # trace_name=teacher_lora_tau${tau}_delta${delta} \
    # trace_path=experiments_gpqa_cala/traces_mmlu/teacher_lora_tau${tau}_delta${delta} \
    # data_split=mmlu_train \
    # teacher=${teacher} \
    # tokenizer=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    # batch_size=256 \
    # max_length=1024 max_prompt_length=256  \
    # answer_force=true \
    # tau=${tau} seed=42 delta=${delta} max_samples=2880 

    echo ">>> [3/4] Training Student model with delta = ${delta} and tau = ${tau}..."
    accelerate launch --config_file acc_config_2.yaml --main_process_port 0 distill.py \
    hydra.run.dir=experiments_gpqa_cala/metadata/distill/lora_teacher \
    student=meta-llama/Llama-3.2-3B \
    tokenizer=meta-llama/Llama-3.2-3B-Instruct \
    exp_dir=experiments_gpqa_cala \
    train_traces=experiments_gpqa_cala/traces_mmlu/teacher_lora_tau${tau}_delta${delta} \
    holdout_traces=traces_holdout \
    model_name=student_manua_tau${tau}_delta${delta}

    echo ">>> [4/4] Eval Student model with delta = ${delta} and tau = ${tau}..."
    accelerate launch --config_file acc_config_2.yaml --main_process_port 0 student_eval.py  \
    hydra.run.dir=experiments_gpqa_cala/metadata/eval/student_mmlu \
    teacher=experiments_gpqa_cala/models/student_manua_tau${tau}_delta${delta}/final \
    is_teacher=false exp_dir=experiments_gpqa_cala answer_force=true \
    data_split=gpqa_test batch_size=256 max_samples=2880 \
    trace_name=eval_student_mmlu_tau${tau}_delta${delta} seed=42 max_length=1024 max_prompt_length=256

done

echo "🎉 DONE!"