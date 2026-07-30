NAME_CHECKPOINT=gvendi_phase2_cls_fastvlm_K80

python eval_mmeb.py \
    --model_name ./training/${NAME_CHECKPOINT}/checkpoint-final\
    --encode_output_path ./MMEB-evaloutputs/${NAME_CHECKPOINT}/ \
    --lora --lora_r 64 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split test \
    --per_device_eval_batch_size 32 \
    --image_dir "VLM_Embed/eval_images" \
    --tgt_prefix_mod
