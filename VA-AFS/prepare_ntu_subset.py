import argparse
import shutil
from pathlib import Path

import numpy as np


CS_TRAIN_IDS = {
    1,
    2,
    4,
    5,
    8,
    9,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    25,
    27,
    28,
    31,
    34,
    35,
    38,
}
CS_TEST_IDS = {
    3,
    6,
    7,
    10,
    11,
    12,
    20,
    21,
    22,
    23,
    24,
    26,
    29,
    30,
    32,
    33,
    36,
    37,
    39,
    40,
}

STAT_FILES = [
    "skes_available_name.txt",
    "label.txt",
    "performer.txt",
    "camera.txt",
    "replication.txt",
    "setup.txt",
]

PREPROCESS_SCRIPTS = [
    "get_raw_skes_data.py",
    "get_raw_denoised_data.py",
    "seq_transformation.py",
]


def load_lines(path: Path):
    return path.read_text().splitlines()


def select_indices_per_class(labels, performers, samples_per_class_per_split):
    selected = []

    for label in sorted(set(labels)):
        label_indices = np.where(labels == label)[0]
        train_indices = [
            int(idx) for idx in label_indices if int(performers[idx]) in CS_TRAIN_IDS
        ]
        test_indices = [
            int(idx) for idx in label_indices if int(performers[idx]) in CS_TEST_IDS
        ]
        selected.extend(train_indices[:samples_per_class_per_split])
        selected.extend(test_indices[:samples_per_class_per_split])

    return np.array(sorted(set(selected)), dtype=np.int64)


def balanced_sample_from_pool(labels, pool_indices, target_size, rng):
    if target_size <= 0 or len(pool_indices) == 0:
        return np.array([], dtype=np.int64)

    if target_size > len(pool_indices):
        raise ValueError(
            f"Requested {target_size} samples from a pool of size {len(pool_indices)}."
        )

    class_to_indices = {}
    for idx in pool_indices:
        class_to_indices.setdefault(int(labels[idx]), []).append(int(idx))

    classes = sorted(class_to_indices)
    if not classes:
        return np.array([], dtype=np.int64)

    selected = []
    remaining = int(target_size)

    # Repeatedly give each remaining class one sample at a time so counts stay balanced.
    while remaining > 0:
        available_classes = [
            label for label in classes if len(class_to_indices[label]) > 0
        ]
        if not available_classes:
            break

        round_labels = available_classes.copy()
        rng.shuffle(round_labels)
        round_take = min(remaining, len(round_labels))

        for label in round_labels[:round_take]:
            candidates = class_to_indices[label]
            pick_pos = int(rng.integers(len(candidates)))
            selected.append(candidates.pop(pick_pos))
            remaining -= 1
            if remaining == 0:
                break

    if len(selected) != target_size:
        raise ValueError(
            f"Could only draw {len(selected)} balanced samples out of {target_size}."
        )

    return np.array(sorted(selected), dtype=np.int64)


def random_sample_from_pool(pool_indices, target_size, rng):
    if target_size <= 0 or len(pool_indices) == 0:
        return np.array([], dtype=np.int64)

    if target_size > len(pool_indices):
        raise ValueError(
            f"Requested {target_size} samples from a pool of size {len(pool_indices)}."
        )

    return np.array(sorted(rng.choice(pool_indices, size=target_size, replace=False)), dtype=np.int64)


def can_balance_pool_exactly(labels, pool_indices, target_size):
    if target_size <= 0:
        return True

    class_counts = {}
    for idx in pool_indices:
        class_counts[int(labels[idx])] = class_counts.get(int(labels[idx]), 0) + 1

    num_classes = len(class_counts)
    if num_classes == 0:
        return False

    base, remainder = divmod(target_size, num_classes)
    required_counts = [base + (1 if i < remainder else 0) for i in range(num_classes)]
    available_counts = sorted(class_counts.values())

    return all(available >= required for available, required in zip(available_counts, required_counts))


def select_indices_random(labels, performers, sample_size, test_ratio, seed, sampling_strategy):
    rng = np.random.default_rng(seed)
    train_pool = np.array(
        [idx for idx, performer in enumerate(performers) if performer in CS_TRAIN_IDS],
        dtype=np.int64,
    )
    test_pool = np.array(
        [idx for idx, performer in enumerate(performers) if performer in CS_TEST_IDS],
        dtype=np.int64,
    )

    if sample_size < 2:
        raise ValueError("--sample_size must be at least 2 so train/test are non-empty.")

    test_size = int(round(sample_size * test_ratio))
    test_size = min(max(test_size, 1), len(test_pool))
    train_size = sample_size - test_size
    train_size = min(max(train_size, 1), len(train_pool))

    if train_size + test_size < sample_size:
        raise ValueError(
            f"Requested {sample_size} samples, but only "
            f"{train_size + test_size} are available."
        )

    if sampling_strategy == "random":
        train_indices = random_sample_from_pool(train_pool, train_size, rng)
        test_indices = random_sample_from_pool(test_pool, test_size, rng)
        strategy_used = "random"
    elif sampling_strategy == "balanced":
        train_indices = balanced_sample_from_pool(
            labels=labels,
            pool_indices=train_pool,
            target_size=train_size,
            rng=rng,
        )
        test_indices = balanced_sample_from_pool(
            labels=labels,
            pool_indices=test_pool,
            target_size=test_size,
            rng=rng,
        )
        strategy_used = "balanced"
    elif sampling_strategy == "balanced_fallback_random":
        can_balance_train = can_balance_pool_exactly(labels, train_pool, train_size)
        can_balance_test = can_balance_pool_exactly(labels, test_pool, test_size)

        if can_balance_train and can_balance_test:
            train_indices = balanced_sample_from_pool(
                labels=labels,
                pool_indices=train_pool,
                target_size=train_size,
                rng=rng,
            )
            test_indices = balanced_sample_from_pool(
                labels=labels,
                pool_indices=test_pool,
                target_size=test_size,
                rng=rng,
            )
            strategy_used = "balanced"
        else:
            train_indices = random_sample_from_pool(train_pool, train_size, rng)
            test_indices = random_sample_from_pool(test_pool, test_size, rng)
            strategy_used = "random_fallback"
    else:
        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")

    indices = np.array(sorted(np.concatenate([train_indices, test_indices])), dtype=np.int64)
    return indices, strategy_used


def write_subset_statistics(source_stat_dir: Path, output_stat_dir: Path, indices):
    output_stat_dir.mkdir(parents=True, exist_ok=True)

    for filename in STAT_FILES:
        lines = load_lines(source_stat_dir / filename)
        subset_lines = [lines[int(idx)] for idx in indices]
        (output_stat_dir / filename).write_text("\n".join(subset_lines) + "\n")

    missing_path = source_stat_dir / "samples_with_missing_skeletons.txt"
    if missing_path.exists():
        selected_names = set(load_lines(output_stat_dir / "skes_available_name.txt"))
        missing_names = [
            line for line in load_lines(missing_path) if line.strip() in selected_names
        ]
        (output_stat_dir / "samples_with_missing_skeletons.txt").write_text(
            "\n".join(missing_names) + ("\n" if missing_names else "")
        )


def copy_preprocess_scripts(source_ntu_dir: Path, output_dir: Path):
    for filename in PREPROCESS_SCRIPTS:
        shutil.copy2(source_ntu_dir / filename, output_dir / filename)

    raw_skes_script = output_dir / "get_raw_skes_data.py"
    raw_skes_text = raw_skes_script.read_text()
    raw_skes_text = raw_skes_text.replace(
        "frames_cnt = np.zeros(num_files, dtype=_)",
        "frames_cnt = np.zeros(num_files, dtype=np.int_)",
    )
    raw_skes_script.write_text(raw_skes_text)

    seq_transform_script = output_dir / "seq_transformation.py"
    seq_transform_text = seq_transform_script.read_text()
    seq_transform_text = seq_transform_text.replace(
        "skes_name = np.loadtxt(skes_name_file, dtype=np.string_)",
        "skes_name = np.loadtxt(skes_name_file, dtype=str)",
    )
    if "try:\n    import h5py\nexcept ImportError:" not in seq_transform_text:
        seq_transform_text = seq_transform_text.replace(
            "import h5py\n",
            "try:\n    import h5py\nexcept ImportError:\n    h5py = None\n",
        )
    if (
        "try:\n    from sklearn.model_selection import train_test_split\n"
        "except ImportError:"
        not in seq_transform_text
    ):
        seq_transform_text = seq_transform_text.replace(
            "from sklearn.model_selection import train_test_split\n",
            (
                "try:\n"
                "    from sklearn.model_selection import train_test_split\n"
                "except ImportError:\n"
                "    train_test_split = None\n"
            ),
        )
    seq_transform_script.write_text(seq_transform_text)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a small BlockGCN NTU preprocessing directory by subsetting "
            "the NTU60 statistics files. Run the copied preprocessing scripts "
            "inside the generated directory to create a small NTU60_CS.npz."
        )
    )
    parser.add_argument(
        "--blockgcn_dir",
        default="../BlockGCN",
        help="Path to the BlockGCN repository.",
    )
    parser.add_argument(
        "--output_dir",
        default="../BlockGCN/data/ntu_subset",
        help="Generated subset preprocessing directory.",
    )
    parser.add_argument(
        "--samples_per_class_per_split",
        type=int,
        default=None,
        help=(
            "Number of samples to keep for each action class in each CS split. "
            "For example, 1 gives up to 120 samples total. Ignored when "
            "--sample_size is provided."
        ),
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=120,
        help=(
            "Total number of NTU60 samples to draw from CS train/test pools. "
            "Default 120."
        ),
    )
    parser.add_argument(
        "--sampling_strategy",
        choices=["balanced", "random", "balanced_fallback_random"],
        default="balanced_fallback_random",
        help=(
            "Sampling policy for --sample_size. "
            "'balanced' minimizes class imbalance within each CS split, "
            "'random' ignores class labels, and "
            "'balanced_fallback_random' tries exact per-class balancing first "
            "and falls back to random sampling when that is not possible."
        ),
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Approximate fraction of sampled data to draw from the CS test pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used with --sample_size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    args = parser.parse_args()

    va_afs_dir = Path(__file__).resolve().parent
    blockgcn_dir = (va_afs_dir / args.blockgcn_dir).resolve()
    source_ntu_dir = blockgcn_dir / "data" / "ntu"
    source_stat_dir = source_ntu_dir / "statistics"
    output_dir = (va_afs_dir / args.output_dir).resolve()

    if not source_stat_dir.exists():
        raise FileNotFoundError(f"Statistics directory not found: {source_stat_dir}")

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Pass --force to overwrite it."
            )
        shutil.rmtree(output_dir)

    labels = np.loadtxt(source_stat_dir / "label.txt", dtype=np.int64)
    performers = np.loadtxt(source_stat_dir / "performer.txt", dtype=np.int64)
    names = np.loadtxt(source_stat_dir / "skes_available_name.txt", dtype=str)

    if args.samples_per_class_per_split is not None:
        indices = select_indices_per_class(
            labels=labels,
            performers=performers,
            samples_per_class_per_split=args.samples_per_class_per_split,
        )
        selection_mode = (
            f"per-class, {args.samples_per_class_per_split} per CS split"
        )
    else:
        indices, strategy_used = select_indices_random(
            labels=labels,
            performers=performers,
            sample_size=args.sample_size,
            test_ratio=args.test_ratio,
            seed=args.seed,
            sampling_strategy=args.sampling_strategy,
        )
        selection_mode = (
            f"{strategy_used} sample_size={args.sample_size}, "
            f"test_ratio={args.test_ratio}, seed={args.seed}, "
            f"requested_strategy={args.sampling_strategy}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_subset_statistics(source_stat_dir, output_dir / "statistics", indices)
    copy_preprocess_scripts(source_ntu_dir, output_dir)

    selected_names = names[indices]
    train_count = sum(int(performers[idx]) in CS_TRAIN_IDS for idx in indices)
    test_count = sum(int(performers[idx]) in CS_TEST_IDS for idx in indices)

    print(f"Created NTU subset directory: {output_dir}")
    print(f"Selection mode: {selection_mode}")
    print(f"Selected samples: {len(indices)}")
    print(f"CS train samples: {train_count}")
    print(f"CS test samples: {test_count}")
    print(f"First sample: {selected_names[0] if len(selected_names) else 'none'}")
    print()
    print("Next steps:")
    print(f"cd {output_dir}")
    print("python get_raw_skes_data.py")
    print("python get_raw_denoised_data.py")
    print("python seq_transformation.py")
    print()
    print("Expected output:")
    print(output_dir / "NTU60_CS.npz")


if __name__ == "__main__":
    main()
