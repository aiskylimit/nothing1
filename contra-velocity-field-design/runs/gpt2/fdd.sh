BASE_PATH=${1-"."}
MASTER_PORT=${2-2040}

# # FDD
# bash ${BASE_PATH}/scripts/gpt2/fdd/fdd_base.sh ${BASE_PATH} 2012 1

# MASTER_PORT=2040
# DEVICE=0
# for benchmark in dolly self_inst vicuna sinst uinst
# do
#     for seed in 10 20 30 40 50
#     do
#         CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${MASTER_PORT} 1 fdd/base --seed $seed --eval-batch-size 64
#     done
# done

# FDD + Contra
# stage 1
# bash ${BASE_PATH}/scripts/gpt2/train_velocity_field.sh ${BASE_PATH} 2012 1
# stage 2
bash ${BASE_PATH}/scripts/gpt2/fdd/contra_base.sh ${BASE_PATH} ${MASTER_PORT} 1

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        bash ${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh ${BASE_PATH} ${MASTER_PORT} 1 contra/fdd/base --seed $seed --eval-batch-size 32
    done
done
