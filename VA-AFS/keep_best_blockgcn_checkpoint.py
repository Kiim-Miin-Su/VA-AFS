import argparse
import re
from pathlib import Path


def parse_best_epoch(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    matches = re.findall(r"Epoch number:\s+(\d+)", log_path.read_text())
    if not matches:
        raise ValueError(f"Could not find best epoch in {log_path}")

    return int(matches[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Delete non-best BlockGCN checkpoint .pt files in a work_dir."
    )
    parser.add_argument("work_dir", help="BlockGCN training output directory")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be kept/deleted without deleting files.",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    best_epoch = parse_best_epoch(work_dir / "log.txt")
    checkpoints = sorted(work_dir.glob("runs-*.pt"))
    best_matches = sorted(work_dir.glob(f"runs-{best_epoch}-*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {work_dir}")
    if not best_matches:
        raise FileNotFoundError(
            f"No checkpoint found for best epoch {best_epoch} in {work_dir}"
        )

    best_checkpoint = best_matches[-1]
    delete_targets = [path for path in checkpoints if path != best_checkpoint]

    print(f"Best epoch: {best_epoch}")
    print(f"Keep: {best_checkpoint}")
    print(f"Delete count: {len(delete_targets)}")

    if args.dry_run:
        for path in delete_targets:
            print(f"Would delete: {path}")
        return

    for path in delete_targets:
        path.unlink()


if __name__ == "__main__":
    main()
