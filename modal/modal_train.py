import json
from typing import Any

import aiohttp
import modal

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

AXOLOTL_REGISTRY_SHA = (
    "9578c47333bdcc9ad7318e54506b9adaf283161092ae780353d506f7a656590a"
)

axolotl_image = (
    modal.Image.from_registry(f"winglian/axolotl@sha256:{AXOLOTL_REGISTRY_SHA}")
    .pip_install(
        "huggingface_hub==0.23.2",
        "hf-transfer==0.1.5",
        "wandb==0.16.3",
        "fastapi==0.110.0",
        "pydantic==2.6.3",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
            AXOLOTL_NCCL_TIMEOUT="60",
        )
    )
    .entrypoint([])
)

MODEL_NAME = "NousResearch/Meta-Llama-3-8B"

app = modal.App("training", image=axolotl_image)

runs_volume = modal.Volume.from_name("example-runs-vol", create_if_missing=True)


@app.function(gpu="A10", volumes={"/runs": runs_volume}, timeout=24 * HOURS)
def run_axolotl(run_folder: str, timeout=24 * HOURS):
    import torch
    import subprocess
    cmd = f"accelerate launch --num_processes {torch.cuda.device_count()} --num_machines 1 --mixed_precision no --dynamo_backend no -m axolotl.cli.train ./config.yml  --debug"
    subprocess.call(cmd.split(), cwd=run_folder)


@app.function(volumes={"/runs": runs_volume}, timeout=24 * HOURS)
def launch(config_raw: str, timeout=24 * HOURS):
    import os
    from datetime import datetime
    import secrets

    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"axo-{time_string}-{secrets.token_hex(2)}"
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)

    with open(f"{run_folder}/config.yml", "w") as f:
        f.write(config_raw)
    runs_volume.commit()

    handle = run_axolotl.spawn(run_folder)
    return run_name, handle


@app.local_entrypoint()
def main(config: str, timeout=24 * HOURS):
    with open(config, "r") as cfg:
        run_name, handle = launch.remote(cfg.read())
    handle.get(timeout=24 * HOURS)
    print(f"Run complete. Tag: {run_name}")



