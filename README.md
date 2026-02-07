# D³ToM: Decider-Guided Dynamic Token Merging for Accelerating Diffusion MLLMs

[![Paper](https://img.shields.io/badge/arXiv-2511.12280-b31b1b.svg)](https://arxiv.org/abs/2511.12280)


Official PyTorch implementation of **"D³ToM: Decider-Guided Dynamic Token Merging for Accelerating Diffusion MLLMs"** (AAAI 2026).

## 📋 Overview

D³ToM (Decider-Guided Dynamic Token Merge) introduces a training-free acceleration framework for Diffusion MLLMs. We utilizes decider tokens to guide the dynamic merging of redundant visual tokens across different denoising steps, effectively shortening the sequence length to achieve faster inference while maintaining competitive performance.

## 🛠️ Installation

### Setup Environment

```bash
# Create a new conda environment
conda create -n d3tom python=3.13 -y
conda activate d3tom

# Install the package with training dependencies
pip install -e .[train]
```

## 📦 Model Weights

Download the pre-trained model weights from Huggingface:

```bash
# Using huggingface-cli (recommended)
hf download jacklishufan/lavida-llada-v1.0-instruct --local-dir ./lavida-ckpts

# Or manually download from:
# https://huggingface.co/jacklishufan/lavida-llada-v1.0-instruct
```

The model weights should be saved to `./lavida-ckpts` directory.

## 🚀 Quick Start

### Demo Script

Run the provided demo script:

```bash
# Basic usage with default settings
python demo.py

# Custom image and prompt
python demo.py \
    --image images/village.png \
    --prompt "What can you see in this image?" \
    --temperature 0.3

# With custom token merging settings
python demo.py \
    --image images/temple.png \
    --prompt "Describe this scene." \
    --merge-layer 3 \
    --merge-ratio 0.8 \
    --merge-adaptive 1 \
```

### Configuration Parameters

#### Generation Parameters

- `--image`: Path to the input image (default: `images/dog.png`)
- `--prompt`: Text prompt for the model (default: `"Describe the image in detail."`)
- `--temperature`: Sampling temperature (default: `0.3`)
- `--pretrained`: Path to pre-trained model checkpoint (default: `lavida-ckpts/lavida-llada-hd`)

#### Token Merging Parameters

Token merging helps reduce computational costs during inference:

- `--merge-layer`: Layer index for token merging (default: `3`)
- `--merge-ratio`: Ratio of tokens to merge (default: `0.8`)
- `--merge-adaptive`: Enable adaptive merging (default: `1`)


## 📄 Citation

If you find this work useful in your research, please cite:

```bibtex
@misc{chang2025d3tomdeciderguideddynamictoken,
      title={D$^{3}$ToM: Decider-Guided Dynamic Token Merging for Accelerating Diffusion MLLMs}, 
      author={Shuochen Chang and Xiaofeng Zhang and Qingyang Liu and Li Niu},
      year={2025},
      eprint={2511.12280},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.12280}, 
}
```

## 🙏 Acknowledgements

This work builds upon several excellent open-source projects:

- [LaViDA](https://github.com/jacklishufan/LaViDa) - A Large Diffusion Language Model for Multimodal Understanding, NeurIPS 2025 Spotlight
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) - The Evaluation Suite of Large Multimodal Models
