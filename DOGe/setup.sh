conda create -y -n doge python=3.10
conda activate doge
pip install -r requirements.txt

export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH


mkdir -p data
#huggingface-cli download --repo-type dataset ANONYMOUS/doge-exps --local-dir data/doge-exps --token $HF_TOKEN
