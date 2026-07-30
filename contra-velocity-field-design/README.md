# CONTRA-KD: Continuous Trajectory Alignment for Knowledge Distillation

Official PyTorch implementation of our paper, **CONTRA-KD: Continuous Trajectory Alignment for Knowledge Distillation**

## Environment Setup
Run the following to install dependencies:
```bash
bash install.sh
```

## Data
### Resources
+ We use the instruction-following data for training and evaluation provided by DSKD (Zhang et al., 2024), which can be found in their [GitHub repository](https://github.com/songmzhang/DSKD).
+ The pretraining plain-text corpus can be download from the HuggingFace datasets [repository](https://huggingface.co/datasets/Skylion007/openwebtext).


### Data Processing
Get plain-text corpus. For convenience, we use a smaller subset of the full OpenWebText dataset which can be found in [this repository](https://huggingface.co/datasets/Elriggs/openwebtext-100k). The dataset's name in `tools/get_openwebtext.py` can be modified to use the complete dataset:
```bash
python3 tools/get_openwebtext.py
```

Tokenize the data and store them in binary files. Change `gpt2` to `openllama2` and `llama2` to preprocess for OpenLLaMA2 and LLaMA2. (See [below](#command-arguments) for providing arguments)
```bash
# Process Dolly Train / Validation Data
bash scripts/gpt2/tools/process_data_dolly.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}

# Process OpenWebText Train Data
bash scripts/gpt2/tools/process_data_pretrain.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

## Run Scripts
We provide scripts for the training pipeline, including data preparation, teacher SFT and student warm-up, training, and evaluation in the `runs` directory. Comment/Uncomment lines as needed to include/exclude steps in the pipeline, such as one-time steps like data preparation, teacher SFT and student warm-up.

We also provide commands to run each step individually. Here, we present examples for GPT2; check out scripts in `runs/` and `scripts/` for paths to scripts for OpenLLaMA2 and LLaMA2.
```bash
bash runs/gpt2/distillm_contra.sh   # Change to any training script in the `runs` directory
# Example
bash runs/gpt2/distillm2_contra.sh
bash runs/gpt2/distillm_contra_qwen.sh
bash runs/gpt2/fdd.sh
bash runs/openllama2/distillm_contra.sh
bash runs/llama2/distillm_contra.sh
```

### Command Arguments:
+ `BASE_PATH`: Path to top-level directory (i.e., the `contra-kd` directory). If you are running commands from this directory, simply pass `.` for `BASE_PATH`.
+ `MASTER_PORT`: Port number to be used. You can pass any number here as it does not affect training process; we mainly use 2012 in our experiments. Note that if you run multiple training processes concurrently, use different values for `MASTER_PORT` for each process.
+ `GPU_NUM`: Number of GPUs to be used. For simplicity, we use 1 GPU and pass `1` here; you can pass your desired number of GPUs to be used for baselines, however code for CONTRA-KD does not support multiple GPUs yet.

### Model Checkpoints
By default, we use model checkpoint paths on HuggingFace for base models (GPT2, OpenLLaMA2, LLaMA2). 

For fine-tuned teacher and warm-up student checkpoints used in training, it is assumed that these checkpoints are trained using our scripts (can be found in `<model-type>/sft` and `<model-type>/init`) and saved locally in the `results/` directory. Thus, we use local paths as provided in save paths for teacher SFT and student warm-up training scripts. These paths can be modified in training scripts in `scripts/` in case of using existing checkpoints (e.g. on HuggingFace).

For LLaMA2 models, make sure you log in using a HuggingFace token (`hf auth login`) of an account with access to the model checkpoints (`meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-hf`).

### Baselines
The final checkpoints are selected by the **ROUGE-L** scores.

#### Teacher SFT
```bash
bash scripts/gpt2/sft/sft_xlarge.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```
#### SFT Baseline
```bash
bash scripts/gpt2/sft/sft_base.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```
#### Student Warm-up
```bash
bash scripts/gpt2/init/init_base.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```
#### KD Baseline
```bash
bash scripts/gpt2/kd/kd_base.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```
#### DistiLLM Baseline
```bash
bash scripts/gpt2/distillm/train_0.1B_1.5B.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```
#### FDD Baseline
```bash
bash ${BASE_PATH}/scripts/gpt2/fdd/fdd_base.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```
#### DistiLLM2 Baseline
```bash
# Generate initial TGOs and SGOs for KD using DistiLLM2
bash ${BASE_PATH}/scripts/gpt2/distillm2/generate_data.sh ${BASE_PATH}
# Run DistiLLM2
bash ${BASE_PATH}/scripts/gpt2/distillm2/distillm2_v2_base.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

### CONTRA-KD
#### Train Velocity Field
Note that the trained velocity field and projector on a training dataset can be reused for training processes on the same dataset; here, they can be used for both FDD + CONTRA-KD and DistiLLM + CONTRA-KD.
```bash
bash ${BASE_PATH}/scripts/gpt2/train_velocity_field.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```
#### DistiLLM + CONTRA-KD
```bash
bash ${BASE_PATH}/scripts/gpt2/distillm/contra_0.1B_1.5B.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```
#### Cross-tokenizer DistiLLM + CONTRA-KD + SRA
This repo now includes a narrow cross-tokenizer path for teacher `Qwen1.5-1.8B` and student `GPT-2 120M`. Unlike the same-tokenizer GPT-2 scripts, this path uses raw Dolly jsonl data in `data/dolly` so the student and teacher can be tokenized separately and aligned with span-level SRA tensors.
```bash
# Train the cross-tokenizer velocity field
bash ${BASE_PATH}/scripts/gpt2/train_velocity_field_qwen_0.1B_1.8B.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}

# Run DistiLLM + CONTRA + SRA alignment
bash ${BASE_PATH}/scripts/gpt2/distillm/contra_qwen_0.1B_1.8B.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}

# Or run the combined pipeline
bash ${BASE_PATH}/runs/gpt2/distillm_contra_qwen.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

#### FDD + CONTRA-KD
```bash
bash ${BASE_PATH}/scripts/gpt2/fdd/contra_base.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```
#### DistiLLM2 + CONTRA-KD
Since DistiLLM2 use TGOs and SGOs, we train a different velocity field using this data.
```bash
# Generate initial TGOs and SGOs (can reuse generated data from baseline)
bash ${BASE_PATH}/scripts/gpt2/distillm2/generate_data.sh ${BASE_PATH}

# Train Velocity Field
bash ${BASE_PATH}/scripts/gpt2/distillm2/train_velocity_field_distillm2.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}

# Run DistiLLM2 + CONTRA-KD
bash ${BASE_PATH}/scripts/gpt2/distillm2/contra_0.1B_1.5B.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

## Run Evaluation
Provide the path to trained checkpoints to `CKPT_PATH`. Note that evaluation uses local paths; download checkpoints to local directory if needed (e.g. for checkpoints on HuggingFace, download using `hf download`).

Find paths to checkpoints specified in training scripts in `scripts/` directory.
+ All are saved in the `results/` directory and mostly follow this pattern: `results/<model-type>/train/<baseline-method>/<student-and-teacher-size>/`.
+ CONTRA-KD checkpoints are instead saved in `results/<model-type>/train/contra/<baseline-method>/<student-and-teacher-size>`.
```bash
bash scripts/gpt2/eval/run_eval.sh ${GPU_IDX} ${CKPT_PATH}
bash scripts/openllama2/eval/run_eval.sh ${GPU_IDX} ${CKPT_PATH}
bash scripts/llama2/eval/run_eval.sh ${GPU_IDX} ${CKPT_PATH}
```

## Acknowledgement
Our code is based on the code of ICML2024 [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/pdf/2402.03898).
