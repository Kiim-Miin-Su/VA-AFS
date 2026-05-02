import argparse
from pathlib import Path

import numpy as np

from constants import K_MAX, TAU, WINDOW_SIZE
from va_afs_threshold import va_afs_threshold_sampling


def valid_frame_count(sequence: np.ndarray):
    frame_has_data = np.any(sequence != 0, axis=1)
    return int(np.sum(frame_has_data))


def ntu_sequence_to_xy(sequence: np.ndarray):
    """
    BlockGCN NTU sequence shape: (T, 150) = T x (M=2, V=25, C=3).
    VA-AFS uses xy only and flattens people/joints into V=50.
    """
    T = sequence.shape[0]
    skeleton = sequence.reshape(T, 2, 25, 3)
    return skeleton[:, :, :, :2].reshape(T, 50, 2)


def apply_va_afs_to_sequence(
    sequence: np.ndarray,
    tau: float,
    k_max: int,
    window_size: int,
):
    valid_count = valid_frame_count(sequence)
    reduced = np.zeros_like(sequence)

    if valid_count == 0:
        return reduced, np.array([], dtype=np.int64)

    valid_sequence = sequence[:valid_count]
    skeleton_xy = ntu_sequence_to_xy(valid_sequence)
    _, metadata = va_afs_threshold_sampling(
        skeleton_xy=skeleton_xy,
        window_size=window_size,
        tau=tau,
        k_max=k_max,
    )

    selected_indices = metadata["selected_indices"]
    selected_sequence = valid_sequence[selected_indices]
    reduced[: len(selected_sequence)] = selected_sequence

    return reduced, selected_indices


def apply_va_afs_to_split(
    data: np.ndarray,
    tau: float,
    k_max: int,
    window_size: int,
):
    reduced = np.zeros_like(data)
    selected_counts = np.zeros(len(data), dtype=np.int32)
    original_counts = np.zeros(len(data), dtype=np.int32)

    for idx, sequence in enumerate(data):
        reduced_sequence, selected_indices = apply_va_afs_to_sequence(
            sequence=sequence,
            tau=tau,
            k_max=k_max,
            window_size=window_size,
        )
        reduced[idx] = reduced_sequence
        selected_counts[idx] = len(selected_indices)
        original_counts[idx] = valid_frame_count(sequence)

        if (idx + 1) % 100 == 0 or idx + 1 == len(data):
            ratio = selected_counts[: idx + 1].sum() / max(
                original_counts[: idx + 1].sum(), 1
            )
            print(f"Processed {idx + 1}/{len(data)} samples, ratio={ratio:.3f}")

    return reduced, original_counts, selected_counts


def main():
    parser = argparse.ArgumentParser(
        description="Apply VA-AFS frame selection to a BlockGCN NTU .npz file."
    )
    parser.add_argument("--input_npz", required=True, help="Original NTU .npz path")
    parser.add_argument(
        "--output_npz",
        default=None,
        help="Output .npz path. Defaults to outputs/blockgcn_npz/<input>_vaafs_*.npz",
    )
    parser.add_argument("--tau", type=float, default=TAU[-1])
    parser.add_argument("--k_max", type=int, default=K_MAX[-1])
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE[-1])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Splits to reduce. Unlisted splits are copied unchanged.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_npz)
    if args.output_npz:
        output_path = Path(args.output_npz)
    else:
        output_path = (
            Path("outputs")
            / "blockgcn_npz"
            / (
                f"{input_path.stem}_vaafs_tau{args.tau}_"
                f"k{args.k_max}_w{args.window_size}.npz"
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    npz = np.load(input_path)
    output = {}
    metadata = {}

    for split in ("train", "test"):
        x_key = f"x_{split}"
        y_key = f"y_{split}"

        if x_key not in npz or y_key not in npz:
            continue

        output[y_key] = npz[y_key]

        if split in args.splits:
            print(f"Applying VA-AFS to {x_key}: {npz[x_key].shape}")
            reduced, original_counts, selected_counts = apply_va_afs_to_split(
                data=npz[x_key],
                tau=args.tau,
                k_max=args.k_max,
                window_size=args.window_size,
            )
            output[x_key] = reduced
            metadata[f"{split}_original_counts"] = original_counts
            metadata[f"{split}_selected_counts"] = selected_counts
            metadata[f"{split}_processed_frame_ratio"] = np.array(
                selected_counts.sum() / max(original_counts.sum(), 1),
                dtype=np.float32,
            )
        else:
            print(f"Copying {x_key} unchanged: {npz[x_key].shape}")
            output[x_key] = npz[x_key]

    output.update(metadata)
    output["vaafs_tau"] = np.array(args.tau, dtype=np.float32)
    output["vaafs_k_max"] = np.array(args.k_max, dtype=np.int32)
    output["vaafs_window_size"] = np.array(args.window_size, dtype=np.int32)

    np.savez_compressed(output_path, **output)

    print(f"Saved VA-AFS BlockGCN npz: {output_path}")
    for split in ("train", "test"):
        ratio_key = f"{split}_processed_frame_ratio"
        if ratio_key in metadata:
            print(f"{split} processed frame ratio: {metadata[ratio_key]:.3f}")


if __name__ == "__main__":
    main()
