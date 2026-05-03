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


def make_count_summary(split: str, original_counts: np.ndarray, selected_counts: np.ndarray):
    valid_mask = original_counts > 0
    ratios = np.divide(
        selected_counts,
        original_counts,
        out=np.zeros_like(selected_counts, dtype=np.float64),
        where=valid_mask,
    )
    valid_ratios = ratios[valid_mask]
    valid_original = original_counts[valid_mask]
    valid_selected = selected_counts[valid_mask]
    total_original = int(valid_original.sum())
    total_selected = int(valid_selected.sum())
    total_ratio = total_selected / max(total_original, 1)
    reduction = 1.0 - total_ratio

    lines = [
        "",
        f"[{split}] VA-AFS frame selection summary",
        f"  samples                 : {len(original_counts)}",
        f"  non-empty samples       : {int(valid_mask.sum())}",
        f"  original frames total   : {total_original}",
        f"  selected frames total   : {total_selected}",
        f"  processed frame ratio   : {total_ratio:.3f} ({total_ratio * 100:.1f}%)",
        f"  frame reduction         : {reduction:.3f} ({reduction * 100:.1f}%)",
    ]

    if len(valid_ratios) == 0:
        lines.append("  no non-empty samples")
        return lines

    def stats_line(name: str, values: np.ndarray):
        percentiles = np.percentile(values, [0, 25, 50, 75, 100])
        mean = float(np.mean(values))
        return (
            f"  {name:<24}: "
            f"mean={mean:.3f}, "
            f"min={percentiles[0]:.3f}, "
            f"p25={percentiles[1]:.3f}, "
            f"median={percentiles[2]:.3f}, "
            f"p75={percentiles[3]:.3f}, "
            f"max={percentiles[4]:.3f}"
        )

    lines.extend(
        [
            stats_line("original frame count", valid_original.astype(np.float64)),
            stats_line("selected frame count", valid_selected.astype(np.float64)),
            stats_line("per-sample ratio", valid_ratios),
            "  ratio histogram         :",
        ]
    )

    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001])
    hist, _ = np.histogram(valid_ratios, bins=bins)
    max_count = max(int(hist.max()), 1)
    for left, right, count in zip(bins[:-1], bins[1:], hist):
        bar = "#" * max(1, int(round((count / max_count) * 24))) if count else ""
        right_label = 1.0 if right > 1.0 else right
        lines.append(f"    {left:.1f}-{right_label:.1f}: {int(count):>4} {bar}")

    example_count = min(8, len(original_counts))
    examples = [
        f"{int(selected_counts[i])}/{int(original_counts[i])}"
        for i in range(example_count)
    ]
    lines.append(f"  first {example_count} samples      : {', '.join(examples)}")

    return lines


def write_summary(summary_path: Path, summary_lines: list[str]):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines).strip() + "\n")


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
    summary_lines = [
        "VA-AFS BlockGCN NPZ summary",
        f"input_npz               : {input_path}",
        f"tau                     : {args.tau}",
        f"k_max                   : {args.k_max}",
        f"window_size             : {args.window_size}",
        f"splits                  : {', '.join(args.splits)}",
    ]

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
            metadata[f"{split}_frame_reduction_ratio"] = np.array(
                1.0 - selected_counts.sum() / max(original_counts.sum(), 1),
                dtype=np.float32,
            )
            split_summary = make_count_summary(
                split=split,
                original_counts=original_counts,
                selected_counts=selected_counts,
            )
            summary_lines.extend(split_summary)
            print("\n".join(split_summary))
        else:
            print(f"Copying {x_key} unchanged: {npz[x_key].shape}")
            output[x_key] = npz[x_key]

    output.update(metadata)
    output["vaafs_tau"] = np.array(args.tau, dtype=np.float32)
    output["vaafs_k_max"] = np.array(args.k_max, dtype=np.int32)
    output["vaafs_window_size"] = np.array(args.window_size, dtype=np.int32)

    np.savez_compressed(output_path, **output)
    summary_path = output_path.with_suffix(".summary.txt")
    summary_lines.append("")
    summary_lines.append(f"output_npz              : {output_path}")
    summary_lines.append(f"summary_txt             : {summary_path}")
    write_summary(summary_path, summary_lines)

    print(f"Saved VA-AFS BlockGCN npz: {output_path}")
    print(f"Saved summary: {summary_path}")
    for split in ("train", "test"):
        ratio_key = f"{split}_processed_frame_ratio"
        reduction_key = f"{split}_frame_reduction_ratio"
        if ratio_key in metadata:
            print(f"{split} processed frame ratio: {metadata[ratio_key]:.3f}")
            print(f"{split} frame reduction ratio: {metadata[reduction_key]:.3f}")


if __name__ == "__main__":
    main()
