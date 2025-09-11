# Axolotl Training Instructions

## Prerequisites
- Docker with GPU support
- NVIDIA drivers installed
- Hugging Face token (for model access)

## Quick Start

### 1. Start Axolotl Docker Container
```bash
docker run --privileged --gpus '"all"' --shm-size 10g --rm -it \
  --name axolotl --ipc=host \
  --mount type=bind,src="${PWD}",target=/workspace/project \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -p 6006:6006 \
  axolotlai/axolotl:main-latest
```

### 2. Preprocess Data
```bash
axolotl preprocess config.yaml
```

### 3. Train Model
```bash
axolotl train config.yaml --debug
```

### 4. Run Inference
```bash
axolotl inference config.yaml --lora-model-dir="./outputs/pii/checkpoint-119"
```

## Alternative: VLLM Inference

For production inference with better performance, use VLLM:

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model NousResearch/Meta-Llama-3-8B-Instruct -tp 2
```

## Important Notes

- **GPU Memory**: Ensure you have sufficient GPU memory (recommended 24GB+ for 7B models)
- **Checkpoints**: Training checkpoints are saved in `./outputs/` directory
- **Monitoring**: Access TensorBoard at `http://localhost:6006` during training
- **Debug Mode**: Use `--debug` flag for training to see detailed logs
- **LoRA Models**: Inference uses LoRA adapters from the specified checkpoint directory

## Troubleshooting

- If you encounter CUDA out of memory errors, reduce batch size in config
- For permission issues, ensure Docker has proper GPU access
- Check that your config.yaml file is properly formatted
