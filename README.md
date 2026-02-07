conda create -n d3tom python=3.13 -y
conda activate d3tom
pip install -e .[train]

huggingface下载: jacklishufan/lavida-llada-v1.0-instruct

--local-dir: ./lavida-ckpts