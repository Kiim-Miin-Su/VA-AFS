import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from run_blockgcn_acc import detect_device, resolve_path


def write_train_config(
    source_config: Path,
    output_config: Path,
    data_path: Path,
    batch_size: int | None,
    test_batch_size: int | None,
    num_worker: int | None,
    num_epoch: int | None,
    save_epoch: int | None,
):
    with source_config.open("r") as f:
        config = yaml.safe_load(f)

    config.setdefault("train_feeder_args", {})
    config.setdefault("test_feeder_args", {})
    config["train_feeder_args"]["data_path"] = str(data_path)
    config["train_feeder_args"]["split"] = "train"
    config["test_feeder_args"]["data_path"] = str(data_path)
    config["test_feeder_args"]["split"] = "test"

    if batch_size is not None:
        config["batch_size"] = batch_size
    if test_batch_size is not None:
        config["test_batch_size"] = test_batch_size
    if num_worker is not None:
        config["num_worker"] = num_worker
    if num_epoch is not None:
        config["num_epoch"] = num_epoch
    if save_epoch is not None:
        config["save_epoch"] = save_epoch

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


def parse_best_epoch(log_path: Path):
    if not log_path.exists():
        return None

    matches = re.findall(r"Epoch number:\s+(\d+)", log_path.read_text())
    if not matches:
        return None

    return int(matches[-1])


def keep_best_checkpoint_only(work_dir: Path):
    best_epoch = parse_best_epoch(work_dir / "log.txt")
    checkpoints = sorted(work_dir.glob("runs-*.pt"))

    if best_epoch is None:
        print(f"Could not find best epoch in {work_dir / 'log.txt'}")
        return None

    best_matches = sorted(work_dir.glob(f"runs-{best_epoch}-*.pt"))
    if not best_matches:
        print(f"Could not find checkpoint for best epoch {best_epoch} in {work_dir}")
        return None

    best_checkpoint = best_matches[-1]
    for checkpoint in checkpoints:
        if checkpoint != best_checkpoint:
            checkpoint.unlink()

    print(f"Kept best checkpoint only: {best_checkpoint}")
    return best_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Train BlockGCN on a given NTU .npz, usually a small subset."
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
    parser.add_argument("--data_npz", required=True, help="NTU .npz to train on")
    parser.add_argument(
        "--work_dir",
        default="outputs/blockgcn_train",
        help="Directory for BlockGCN training logs and weights. Relative paths are under VA-AFS.",
    )
    parser.add_argument("--model", default="model.BlockGCN.Model")
    parser.add_argument(
        "--device",
        nargs="+",
        default=None,
        help="cpu, mps, or CUDA device ids such as 0 1. Defaults to auto detection.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument(
        "--save_epoch",
        type=int,
        default=0,
        help=(
            "Start saving checkpoints after this epoch. Default 0 is useful for "
            "short subset runs; BlockGCN's original default 10 would save no "
            "checkpoint for 1-5 epoch smoke tests."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the BlockGCN command without running it.",
    )
    parser.add_argument(
        "--keep_best_only",
        action="store_true",
        help="After training, delete non-best checkpoint .pt files in --work_dir.",
    )
    args = parser.parse_args()
    device = args.device or detect_device()

    va_afs_dir = Path(__file__).resolve().parent
    blockgcn_dir = resolve_path(args.blockgcn_dir, va_afs_dir)
    config_path = resolve_path(args.config, blockgcn_dir)
    data_path = resolve_path(args.data_npz, va_afs_dir)
    work_dir = resolve_path(args.work_dir, va_afs_dir)

    if not blockgcn_dir.exists():
        raise FileNotFoundError(f"BlockGCN directory not found: {blockgcn_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data npz not found: {data_path}")

    generated_config = (
        va_afs_dir
        / "outputs"
        / "blockgcn_configs"
        / f"{data_path.stem}_{config_path.stem}_train.yaml"
    )
    write_train_config(
        source_config=config_path,
        output_config=generated_config,
        data_path=data_path,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_worker=args.num_worker,
        num_epoch=args.num_epoch,
        save_epoch=args.save_epoch,
    )

    command = [
        sys.executable,
        "-u",
        "main.py",
        "--phase",
        "train",
        "--config",
        str(generated_config),
        "--model",
        args.model,
        "--work-dir",
        str(work_dir),
        "--device",
        *[str(device_id) for device_id in device],
    ]

    print("BlockGCN command:", flush=True)
    print(" ".join(command), flush=True)

    if args.dry_run:
        return

    result = subprocess.run(
        command,
        cwd=blockgcn_dir,
        env=make_blockgcn_env(blockgcn_dir),
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    if args.keep_best_only:
        keep_best_checkpoint_only(work_dir)


if __name__ == "__main__":
    main()
