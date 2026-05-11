import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print()
    print(f"[cwd] {cwd}")
    print("[cmd] " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def run_preprocess_scripts(subset_dir: Path, dry_run: bool) -> None:
    for script_name in (
        "get_raw_skes_data.py",
        "get_raw_denoised_data.py",
        "seq_transformation.py",
    ):
        run([sys.executable, script_name], cwd=subset_dir, dry_run=dry_run)


def find_checkpoint(work_dir: Path) -> Path:
    checkpoints = sorted(work_dir.glob("runs-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {work_dir}")
    if len(checkpoints) > 1:
        print(f"warning: multiple checkpoints found; using {checkpoints[-1]}")
    return checkpoints[-1]


def has_vaafs_metadata(npz_path: Path, splits: tuple[str, ...]) -> bool:
    if not npz_path.exists():
        return False
    try:
        with np.load(npz_path) as data:
            return all(
                f"{split}_original_counts" in data and f"{split}_selected_counts" in data
                for split in splits
            )
    except Exception as exc:
        print(f"warning: could not read existing VA-AFS npz; regenerating: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Colab presentation pipeline: NTU subset preprocessing, "
            "BlockGCN training, VA-AFS test split reduction, and accuracy comparison."
        )
    )
    parser.add_argument("--sample_size", type=int, default=20000)
    parser.add_argument(
        "--full_data",
        action="store_true",
        help=(
            "Use the full NTU60 data directory at BlockGCN/data/ntu instead of "
            "creating a sampled subset."
        ),
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sampling_strategy",
        choices=["balanced", "random", "balanced_fallback_random"],
        default="balanced_fallback_random",
    )
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_worker", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--k_max", type=int, default=13)
    parser.add_argument("--window_size", type=int, default=13)
    parser.add_argument(
        "--device",
        nargs="+",
        default=None,
        help="Optional device override, for example: --device 0 or --device cpu.",
    )
    parser.add_argument("--force_subset", action="store_true")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_vaafs", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    va_afs_dir = Path(__file__).resolve().parent
    project_root = va_afs_dir.parent
    blockgcn_dir = project_root / "BlockGCN"
    subset_name = "ntu_full" if args.full_data else f"ntu_subset_{args.sample_size}"
    subset_dir = blockgcn_dir / "data" / ("ntu" if args.full_data else subset_name)
    original_npz = subset_dir / "NTU60_CS.npz"
    train_work_dir = va_afs_dir / "outputs" / "blockgcn_train" / (
        f"{subset_name}_original_e{args.num_epoch}"
    )
    vaafs_npz = va_afs_dir / "outputs" / "blockgcn_npz" / (
        f"NTU60_CS_{subset_name}_vaafs_test_only_"
        f"tau{args.tau}_k{args.k_max}_w{args.window_size}.npz"
    )

    if args.full_data:
        if not subset_dir.exists():
            raise FileNotFoundError(f"Full NTU60 data directory not found: {subset_dir}")
        print(f"use full NTU60 data directory: {subset_dir}")
    elif args.force_subset or not subset_dir.exists():
        run(
            [
                sys.executable,
                "prepare_ntu_subset.py",
                "--sample_size",
                str(args.sample_size),
                "--test_ratio",
                str(args.test_ratio),
                "--seed",
                str(args.seed),
                "--sampling_strategy",
                args.sampling_strategy,
                "--output_dir",
                str(subset_dir),
                "--force",
            ],
            cwd=va_afs_dir,
            dry_run=args.dry_run,
        )
    else:
        print(f"skip subset creation: {subset_dir}")

    if args.force_preprocess or not original_npz.exists():
        run_preprocess_scripts(subset_dir=subset_dir, dry_run=args.dry_run)
    else:
        print(f"skip preprocessing: {original_npz}")

    checkpoint = None
    if not args.skip_train:
        existing_checkpoints = sorted(train_work_dir.glob("runs-*.pt"))
        if args.force_train or not existing_checkpoints:
            train_command = [
                sys.executable,
                "run_blockgcn_train.py",
                "--data_npz",
                str(original_npz),
                "--work_dir",
                str(train_work_dir),
                "--num_epoch",
                str(args.num_epoch),
                "--batch_size",
                str(args.batch_size),
                "--test_batch_size",
                str(args.test_batch_size),
                "--num_worker",
                str(args.num_worker),
                "--save_epoch",
                "0",
                "--keep_best_only",
            ]
            if args.device:
                train_command.extend(["--device", *args.device])
            run(train_command, cwd=va_afs_dir, dry_run=args.dry_run)
        else:
            print(f"skip training: checkpoint already exists in {train_work_dir}")

        if not args.dry_run:
            checkpoint = find_checkpoint(train_work_dir)

    if args.force_vaafs or not has_vaafs_metadata(vaafs_npz, ("test",)):
        run(
            [
                sys.executable,
                "apply_va_afs_to_blockgcn_npz.py",
                "--input_npz",
                str(original_npz),
                "--output_npz",
                str(vaafs_npz),
                "--tau",
                str(args.tau),
                "--k_max",
                str(args.k_max),
                "--window_size",
                str(args.window_size),
                "--splits",
                "test",
            ],
            cwd=va_afs_dir,
            dry_run=args.dry_run,
        )
    else:
        print(f"skip VA-AFS npz: {vaafs_npz}")

    if args.skip_eval or args.skip_train:
        print("pipeline complete.")
        return

    if args.dry_run:
        print("dry run complete.")
        return

    assert checkpoint is not None
    for label, data_npz in (("original", original_npz), ("vaafs", vaafs_npz)):
        eval_work_dir = va_afs_dir / "outputs" / "blockgcn_acc" / (
            f"{subset_name}_{label}_e{args.num_epoch}"
        )
        eval_command = [
            sys.executable,
            "run_blockgcn_acc.py",
            "--data_npz",
            str(data_npz),
            "--weights",
            str(checkpoint),
            "--work_dir",
            str(eval_work_dir),
            "--test_batch_size",
            str(args.test_batch_size),
            "--num_worker",
            str(args.num_worker),
            "--save_score",
        ]
        if args.device:
            eval_command.extend(["--device", *args.device])
        run(eval_command, cwd=va_afs_dir, dry_run=False)

    print("pipeline complete.")
    print(f"checkpoint : {checkpoint}")
    print(f"original npz: {original_npz}")
    print(f"VA-AFS npz : {vaafs_npz}")


if __name__ == "__main__":
    main()
