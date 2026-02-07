# D3ToM: Discrete Diffusion Decoding for Multimodal Large Language Models

[![Paper](https://img.shields.io/badge/arXiv-2511.12280-b31b1b.svg)](https://arxiv.org/abs/2511.12280)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **"D3ToM: Discrete Diffusion Decoding for Multimodal Large Language Models"** (AAAI 2026).

## 📋 Overview

D3ToM (Discrete Diffusion Decoding for Theory of Mind) introduces **LLaDA** (Large Language Discrete Diffusion Autoregressive) model, a novel framework that combines discrete diffusion models with multimodal large language models (MLLMs). This approach enables more effective and flexible token generation for vision-language tasks.

### Key Features

- **Discrete Diffusion Decoding**: Novel token generation mechanism based on discrete diffusion processes
- **Multimodal Understanding**: Seamless integration with vision encoders for image understanding
- **LLaDA Architecture**: Custom transformer-based model with masked diffusion training
- **Flexible Generation**: Support for various decoding schedules (uniform, shift, cosine, logit-normal)
- **Token Merging**: Adaptive token merging for efficient inference
- **Multiple Vision Encoders**: Compatible with SigLIP and other vision transformers

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- conda or pip package manager

### Setup Environment

```bash
# Create a new conda environment
conda create -n d3tom python=3.13 -y
conda activate d3tom

# Install the package with training dependencies
pip install -e .[train]
```

### Dependencies

The main dependencies include:
- PyTorch 2.6.0
- Transformers 4.50.3
- torchvision 0.21.0
- DeepSpeed 0.16.4
- Accelerate 1.6.0
- PEFT 0.4.0
- timm (for vision models)
- And more (see `pyproject.toml` for complete list)

## 📦 Model Weights

Download the pre-trained model weights from Hugging Face:

```bash
# Using huggingface-cli (recommended)
huggingface-cli download jacklishufan/lavida-llada-v1.0-instruct --local-dir ./lavida-ckpts

# Or manually download from:
# https://huggingface.co/jacklishufan/lavida-llada-v1.0-instruct
```

The model weights should be saved to `./lavida-ckpts` directory.

## 🚀 Quick Start

### Basic Usage

```python
from src import lavida

# Run inference on an image
answer, elapsed_time = lavida.run(
    image_path="images/dog.png",
    prompt="Describe the image in detail.",
    temperature=0.3,
    pretrained="lavida-ckpts/lavida-llada-hd"
)

print(f"Answer: {answer}")
print(f"Inference time: {elapsed_time:.4f}s")
```

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
    --merge-window 0.2
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
- `--merge-window`: Window size for merging (default: `0.2`)

These parameters can also be set via environment variables:
```bash
export LLADA_MERGE_LAYER=3
export LLADA_MERGE_RATIO=0.8
export LLADA_MERGE_RATIO_ADAPTIVE=1
```

## 🏗️ Architecture

### LLaDA Model

The LLaDA (Large Language Discrete Diffusion Autoregressive) model consists of:

1. **Vision Encoder**: SigLIP-SO400M for image feature extraction
2. **Visual Projector**: MLP projection layer (2-layer with GELU)
3. **LLaDA Transformer**: Custom transformer with discrete diffusion capability
4. **Diffusion Generation**: Novel discrete diffusion-based token generation

### Key Components

- **Masked Diffusion Training**: Trains the model to predict masked tokens with diffusion process
- **Flexible Noise Schedules**: Support for uniform, shift, cosine, and logit-normal schedules
- **Adaptive Attention**: Efficient attention mechanisms with optional FlexAttention
- **Token Merging**: Reduces computational cost by merging similar visual tokens

## 📊 Example Images

The repository includes sample images in the `images/` directory:
- `dog.png` - Dog image example
- `village.png` - Village scene
- `temple.png` - Temple architecture
- `port.png` - Harbor/port scene
- `firework.png` - Fireworks display

## 🔬 Training (Advanced)

The repository includes training scripts in `llava/train/` for advanced users:

- `train.py`: Main training script
- `train_mem.py`: Memory-efficient training
- `train_dpo.py`: DPO (Direct Preference Optimization) training

For training details, please refer to the paper and training scripts.

## 📄 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{d3tom2026,
  title={D3ToM: Discrete Diffusion Decoding for Multimodal Large Language Models},
  author={[Authors]},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026},
  url={https://arxiv.org/abs/2511.12280}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 BCMI

## 🙏 Acknowledgements

This work builds upon several excellent open-source projects:

- [LLaVA](https://github.com/haotian-liu/LLaVA) - Visual instruction tuning framework
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face transformers library
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Deep learning optimization library
- [SigLIP](https://github.com/google-research/big_vision) - Vision encoder

## 📧 Contact

For questions and discussions, please open an issue on GitHub or contact the authors.

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐!