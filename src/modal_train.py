import modal
from pathlib import PurePosixPath
from typing import Union
from pathlib import Path

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

runs_volume = modal.Volume.from_name("runs-vol", create_if_missing=True)
pretrained_volume = modal.Volume.from_name(
    "pretrained-vol", create_if_missing=True
)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}

@app.function(image=axolotl_image, gpu="A10", volumes=VOLUME_CONFIG, timeout=24 * HOURS)
def run_axolotl(run_folder: str, output_dir, timeout=24 * HOURS):
    import torch
    cmd = f"accelerate launch --num_processes {torch.cuda.device_count()} --num_machines 1 --mixed_precision no --dynamo_backend no -m axolotl.cli.train ./config.yml  --debug"
    run_cmd(cmd, run_folder)

    merge_handle = merge.spawn(run_folder, output_dir)
    with open(f"{run_folder}/logs.txt", "a") as f:
        f.write(f"<br>merge: https://modal.com/logs/call/{merge_handle.object_id}\n")
        print(f"Beginning merge {merge_handle.object_id}.")
    return merge_handle


@app.function(
    image=axolotl_image,
    gpu='a10',
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
)
def merge(run_folder: str, output_dir: str):
    import shutil
    import torch
    import subprocess

    output_path = Path(run_folder) / output_dir
    shutil.rmtree(output_path / "merged", ignore_errors=True)

    with open(f"{run_folder}/config.yml"):
        print(f"Merge from {output_path}")

    MERGE_CMD = [
        "accelerate",
        "launch",
        "--num_processes",
        str(torch.cuda.device_count()),
        "--num_machines",
        "1",
        "--mixed_precision",
        "no",
        "--dynamo_backend",
        "no",
        "-m",
        "axolotl.cli.merge_lora",
        "./config.yml",
        f"--lora_model_dir={output_path}",
    ]
    run_cmd(MERGE_CMD, run_folder)

    VOLUME_CONFIG["/runs"].commit()

@app.function(image=axolotl_image, volumes=VOLUME_CONFIG, timeout=24 * HOURS)
def launch(config_raw: str, timeout=24 * HOURS):
    import os
    from datetime import datetime
    import secrets
    from huggingface_hub import snapshot_download
    import yaml

    config = yaml.safe_load(config_raw)
    model_name = config["base_model"]

    try:
        snapshot_download(model_name, local_files_only=True)
        print(f"Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} ...")
        snapshot_download(model_name)

        print("Committing /pretrained directory (no progress bar) ...")
        VOLUME_CONFIG["/pretrained"].commit()

    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"axo-{time_string}-{secrets.token_hex(2)}"
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)

    with open(f"{run_folder}/config.yml", "w") as f:
        f.write(config_raw)
    runs_volume.commit()

    handle = run_axolotl.spawn(run_folder, config["output_dir"])
    return run_name, handle


@app.local_entrypoint()
def main(config: str, timeout=24 * HOURS):
    with open(config, "r") as cfg:
        run_name, handle = launch.remote(cfg.read())
    merge_handle = handle.get(timeout=24 * HOURS)
    if merge_handle is not None:
        merge_handle.get(timeout=24 * HOURS)
    print(f"Run complete. Tag: {run_name}")


def run_cmd(cmd, run_folder: str):
    """Run a command inside a folder, ensuring Modal volumes are reloaded/committed."""
    import subprocess

    # Ensure volumes contain latest files.
    VOLUME_CONFIG["/pretrained"].reload()
    VOLUME_CONFIG["/runs"].reload()

    # Support both list and string commands; propagate errors.
    if isinstance(cmd, list):
        exit_code = subprocess.call(cmd, cwd=run_folder)
    else:
        exit_code = subprocess.call(str(cmd).split(), cwd=run_folder)
    if exit_code:
        exit(exit_code)

    # Commit writes to volume.
    VOLUME_CONFIG["/runs"].commit()



