#1 +120
#doge
#v2

#2 -f-/home/ubuntu/aiskylimit_nothing1/anti-distill/results_yaml/ +a

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt-get install -y cuda-toolkit-13-0
# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc
# bash install_miniconda.sh

# cd gpu_burn
# make CUDAPATH=/usr/local/cuda-13.0
# ./gpu_burn 36000000000

# kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
# sleep 10
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH



cd ./anti-distill
# rm -rf results_yaml
# bash ./collect_results.sh
# ls experiments_gsm8k_7 -R
# ls experiments_gsm8k -R
# ls experiments_gpqa_cala -R
# ls experiments_gpqa_tau -R
bash ./project_commands.sh


# cd ./DOGe
# source .venv/bin/activate
# export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
# bash ./extract-model.sh
# bash ./project_commands.sh
