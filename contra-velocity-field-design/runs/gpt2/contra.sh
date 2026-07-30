BASE_PATH=${1-"."}
MASTER_PORT=${2-2040}

# load openwebtext dataset
# python3 tools/get_openwebtext.py
# bash ${BASE_PATH}/scripts/gpt2/tools/process_data_pretrain.sh ${BASE_PATH} 2012 1

# bash ${BASE_PATH}/scripts/gpt2/tools/process_data_dolly.sh ${BASE_PATH} 2012 1


# stage 1
bash ${BASE_PATH}/scripts/gpt2/train_velocity_field.sh ${BASE_PATH} ${MASTER_PORT} 1

# stage 2
bash ${BASE_PATH}/scripts/gpt2/contra_0.1B_1.5B.sh ${BASE_PATH} ${MASTER_PORT} 1

# evaluation
# bash ${BASE_PATH}/scripts/gpt2/eval/run_eval.sh 0 contra_0.1B_1.5B_final2
ckpt="contra/0.1B_1.5B"

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        bash ${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh ${BASE_PATH} ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 32
    done
done