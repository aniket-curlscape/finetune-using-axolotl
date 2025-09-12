# Finetune LLMs using Axolotl

Slides to the workshop [link](https://docs.google.com/presentation/d/1lIVR0L439pkEWvrm3ulcVQT7fn__f_Bu7sZFVaQfZNA/edit?usp=sharing)

This repository provides a comprehensive toolkit for fine-tuning Large Language Models (LLMs) using the [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) framework. The project focuses on PII (Personally Identifiable Information) masking tasks, demonstrating how to fine-tune models to automatically detect and mask sensitive information in text.

## What This Repo Does

This repository contains everything you need to fine-tune language models for PII masking tasks, including:

- **PII Detection & Masking**: Fine-tune models to automatically replace usernames with `[USERNAME]`, emails with `[EMAIL]`, phone numbers with `[PHONE]`, addresses with `[ADDRESS]`, and timestamps with `[TIME]`
- **Multiple Training Environments**: Support for both cloud-based (Modal) and local training setups
- **Data Processing**: Helper tools for downloading and preparing datasets
- **Baseline Evaluation**: Notebooks to test model performance before and after fine-tuning
- **Production Ready**: Includes inference scripts and VLLM integration for production deployment

## Table of Contents

### 1. [Finetune LLMs using Axolotl on Modal](./modal/)
Cloud-based training using Modal's infrastructure for scalable GPU training. Perfect for users who want to leverage cloud resources without managing their own hardware.

**Key Features:**
- Automated Modal deployment with Axolotl Docker image
- GPU-optimized training with A10 instances (but configurable to other GPUs)
- Persistent volume storage for checkpoints
- Easy configuration management

### 2. [Finetune LLMs using Axolotl on VMs/Local Environment](./local_training/)
Local training setup using Docker for users who prefer to train on their own hardware or VMs.

**Key Features:**
- Docker-based training environment
- GPU support with NVIDIA drivers
- TensorBoard integration for monitoring
- VLLM inference support
- Step-by-step training instructions

### 3. [Helper Notebooks](./helpers/)
Data processing and evaluation utilities to help with dataset preparation and model testing.

**Key Features:**
- Dataset downloading and exploration
- Baseline model evaluation
- Performance comparison tools
- PII masking accuracy testing

### 4. [Configuration Files](./configs/)
Centralized Axolotl configuration files for different training scenarios and model setups.

**Key Features:**
- Pre-configured settings for Llama-3-8B
- LoRA adapter configurations
- Optimized hyperparameters for PII masking tasks
- Easy customization for different datasets

## Quick Start

### Option 1: Modal Training (Recommended for Cloud)
```bash
# Install Modal CLI
pip install -r requirements.txt

# Use vLLM and run inference server for initial benchmarking
modal deploy ./src/modal_base_inference.py

# Run training on Modal
modal run src/modal_train.py --config configs/config-llama-8b-100.yaml

# Run inference on Modal
modal run --quiet -m src.modal_infer --prompt "Let us meet at 9am"
```

### Option 2: Local Training
```bash
# Start Axolotl Docker container
docker run --privileged --gpus '"all"' --shm-size 10g --rm -it \
  --name axolotl --ipc=host \
  --mount type=bind,src="${PWD}",target=/workspace/project \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -p 6006:6006 \
  axolotlai/axolotl:main-latest

# Preprocess data
axolotl preprocess configs/config.yaml

# Train model
axolotl train configs/config.yaml --debug
```

## Dataset

The repository is configured to work with the `aniket-curlscape/pii-masking-english-1k` dataset, which contains examples of text with PII that need to be masked. The model learns to:

- Replace usernames with `[USERNAME]`
- Replace email addresses with `[EMAIL]`
- Replace phone numbers with `[PHONE]`
- Replace addresses with `[ADDRESS]`
- Replace timestamps with `[TIME]`

## Configuration

The main configuration is stored in `configs/config.yaml` and includes:

- **Base Model**: `NousResearch/Meta-Llama-3-8B`
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Sequence Length**: 4096 tokens
- **Batch Size**: 2 (with gradient accumulation)
- **Learning Rate**: 0.0002
- **Epochs**: 1

## Monitoring

- **TensorBoard**: Access training metrics at `http://localhost:6006`
- **Wandb**: Optional integration for experiment tracking
- **Checkpoints**: Automatically saved in `./outputs/` directory

## Requirements

- Python 3.8+
- Docker with GPU support (for local training)
- NVIDIA drivers (for local training)
- Hugging Face token (for model access)
- Modal account (for cloud training)

## License

This project is open source and available under the MIT License.
