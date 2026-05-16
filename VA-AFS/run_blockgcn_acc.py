import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


def detect_device():
    try:
        import torch
    except ImportError:
        return ["cpu"]

    if torch.cuda.is_available():
        return ["0"]

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return ["mps"]

    return ["cpu"]


def resolve_path(path: str, base_dir: Path):
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def resolve_existing_path(path: str, base_dirs: list[Path]):
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    resolved_candidates = [(base_dir / candidate).resolve() for base_dir in base_dirs]
    for resolved in resolved_candidates:
        if resolved.exists():
            return resolved

    return resolved_candidates[0]


def write_eval_config(
    source_config: Path,
    output_config: Path,
    data_path: Path,
    test_batch_size: int | None,
    num_worker: int | None,
):
    with source_config.open("r") as f:
        config = yaml.safe_load(f)

    config.setdefault("test_feeder_args", {})
    config["test_feeder_args"]["data_path"] = str(data_path)
    config["test_feeder_args"]["split"] = "test"

    if test_batch_size is not None:
        config["test_batch_size"] = test_batch_size

    if num_worker is not None:
        config["num_worker"] = num_worker

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def make_blockgcn_env(blockgcn_dir: Path):
    env = os.environ.copy()
    torchlight_path = str(blockgcn_dir / "torchlight")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        torchlight_path
        if not current_pythonpath
        else f"{torchlight_path}{os.pathsep}{current_pythonpath}"
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_accuracy(output: str):
    match = re.search(r"Accuracy:\s+([0-9.]+)", output)
    if not match:
        return None
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(
        description="Run BlockGCN test phase and report accuracy for a given .npz."
    )
    parser.add_argument(
        "--blockgcn_dir",
        default="../BlockGCN",
        help="Path to the BlockGCN repository.",
    )
    parser.add_argument(
        "--config",
        default="config/nturgbd-cross-subject/default.yaml",
        help="BlockGCN config path, relative to --blockgcn_dir unless absolute.",
    )
    parser.add_argument("--data_npz", required=True, help="NTU .npz to evaluate")
    parser.add_argument("--weights", required=True, help="BlockGCN checkpoint .pt")
    parser.add_argument(
        "--work_dir",
        default="outputs/blockgcn_acc",
        help="Directory for BlockGCN logs/scores. Relative paths are under VA-AFS.",
    )
    parser.add_argument("--model", default="model.BlockGCN.Model")
    parser.add_argument(
        "--device",
        nargs="+",
        default=None,
        help="cpu, mps, or CUDA device ids such as 0 1. Defaults to auto detection.",
    )
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_worker", type=int, default=None)
    parser.add_argument("--save_score", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the BlockGCN command without running it.",
    )
    args = parser.parse_args()
    device = args.device or detect_device()

    va_afs_dir = Path(__file__).resolve().parent
    blockgcn_dir = resolve_path(args.blockgcn_dir, va_afs_dir)
    config_path = resolve_path(args.config, blockgcn_dir)
    data_path = resolve_path(args.data_npz, va_afs_dir)
    weights_path = resolve_existing_path(args.weights, [va_afs_dir, blockgcn_dir])
    work_dir = resolve_path(args.work_dir, va_afs_dir)

    if not blockgcn_dir.exists():
        raise FileNotFoundError(f"BlockGCN directory not found: {blockgcn_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data npz not found: {data_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    generated_config = (
        va_afs_dir
        / "outputs"
        / "blockgcn_configs"
        / f"{data_path.stem}_{config_path.stem}_eval.yaml"
    )
    write_eval_config(
        source_config=config_path,
        output_config=generated_config,
        data_path=data_path,
        test_batch_size=args.test_batch_size,
        num_worker=args.num_worker,
    )

    command = [
        sys.executable,
        "-u",
        "main.py",
        "--phase",
        "test",
        "--config",
        str(generated_config),
        "--model",
        args.model,
        "--weights",
        str(weights_path),
        "--work-dir",
        str(work_dir),
        "--save-score",
        str(args.save_score),
        "--device",
        *[str(device_id) for device_id in device],
    ]

    print("BlockGCN command:", flush=True)
    print(" ".join(command), flush=True)

    if args.dry_run:
        return

    process = subprocess.Popen(
        command,
        cwd=blockgcn_dir,
        env=make_blockgcn_env(blockgcn_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    output_parts = []
    for chunk in iter(lambda: process.stdout.read(1), ""):
        print(chunk, end="", flush=True)
        output_parts.append(chunk)
    returncode = process.wait()
    output = "".join(output_parts)

    accuracy = parse_accuracy(output)
    if accuracy is not None:
        print(f"Parsed Top-1 accuracy: {accuracy:.4f}", flush=True)

    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
