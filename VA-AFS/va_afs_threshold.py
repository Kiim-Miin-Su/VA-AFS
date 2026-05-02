import argparse
import os
import tempfile
from pathlib import Path

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants import TAU, K_MAX, WINDOW_SIZE  # ./constants.py


def compute_motion_features(window: np.ndarray, eps: float = 1e-6):
    """
    window shape: (W, V, C)
    """
    W, V, C = window.shape  # window_size, joint_counts, joint dimensions

    d = window[-1] - window[0]  # displacement

    if W >= 2:
        v = window[-1] - window[-2]
    else:
        v = np.zeros((V, C), dtype=np.float32)

    if W >= 3:
        v_now = window[-1] - window[-2]
        v_prev = window[-2] - window[-3]
        a = v_now - v_prev
    else:
        a = np.zeros((V, C), dtype=np.float32)

    sigma = np.std(window, axis=0)
    sigma = np.linalg.norm(sigma, axis=-1) + eps

    return d, v, a, sigma


def compute_frame_importance(
    window: np.ndarray,
    lambda_v: float = 0.5,
    lambda_a: float = 0.5,
    lambda_mean: float = 0.5,
    lambda_max: float = 0.5,
    eps: float = 1e-6,
):
    d, v, a, sigma = compute_motion_features(window, eps)

    d_norm = np.linalg.norm(d, axis=-1)
    v_norm = np.linalg.norm(v, axis=-1)
    a_norm = np.linalg.norm(a, axis=-1)

    motion = d_norm + lambda_v * v_norm + lambda_a * a_norm

    joint_scores = motion / sigma

    beta = lambda_mean * np.mean(joint_scores) + lambda_max * np.max(joint_scores)

    motion_summary = {
        "mean_displacement": float(np.mean(d_norm)),
        "mean_velocity": float(np.mean(v_norm)),
        "mean_acceleration": float(np.mean(a_norm)),
    }

    return beta, joint_scores, motion_summary


def sigmoid(x: float):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float):
    return np.log(p / (1.0 - p))


def va_afs_threshold_sampling(
    skeleton_xy: np.ndarray,
    window_size: int,
    tau: float,
    k_max: int,
    eps: float = 1e-6,
):
    """
    skeleton_xy shape: (T, V, 2)
    """
    T, V, C = skeleton_xy.shape

    selected_indices = []
    selected_reasons = []
    raw_frame_scores = []
    frame_scores = []
    z_scores = []
    beta_means = []
    beta_stds = []
    joint_scores_all = []
    mean_displacements = []
    mean_velocities = []
    mean_accelerations = []
    gates = np.zeros(T, dtype=np.int64)

    last_selected = -1
    beta_count = 0
    beta_mean = 0.0
    beta_m2 = 0.0

    for t in range(T):
        start = max(0, t - window_size + 1)
        window = skeleton_xy[start : t + 1]

        beta, joint_scores, motion_summary = compute_frame_importance(window)

        if beta_count >= 2:
            prev_mean = beta_mean
            prev_std = np.sqrt(beta_m2 / (beta_count - 1))
            z_score = (beta - prev_mean) / (prev_std + eps)
        else:
            prev_mean = beta_mean if beta_count > 0 else beta
            prev_std = 0.0
            z_score = 0.0

        score = sigmoid(z_score)

        raw_frame_scores.append(beta)
        frame_scores.append(score)
        z_scores.append(z_score)
        beta_means.append(prev_mean)
        beta_stds.append(prev_std)
        joint_scores_all.append(joint_scores)
        mean_displacements.append(motion_summary["mean_displacement"])
        mean_velocities.append(motion_summary["mean_velocity"])
        mean_accelerations.append(motion_summary["mean_acceleration"])

        is_warmup = t < window_size
        is_important = score >= tau
        is_forced = last_selected < 0 or (t - last_selected >= k_max)

        if is_warmup or is_important or is_forced:
            selected_indices.append(t)
            if is_warmup:
                selected_reasons.append("warmup")
            elif is_important and is_forced:
                selected_reasons.append("tau+k_max")
            elif is_important:
                selected_reasons.append("tau")
            else:
                selected_reasons.append("k_max")
            gates[t] = 1
            last_selected = t

        beta_count += 1
        delta = beta - beta_mean
        beta_mean += delta / beta_count
        delta2 = beta - beta_mean
        beta_m2 += delta * delta2

    selected_indices = np.array(selected_indices, dtype=np.int64)
    sampled = skeleton_xy[selected_indices]

    step_sizes = np.diff(selected_indices, prepend=selected_indices[0])

    metadata = {
        "selected_indices": selected_indices,
        "selected_reasons": np.array(selected_reasons),
        "frame_scores": np.array(frame_scores),
        "raw_frame_scores": np.array(raw_frame_scores),
        "z_scores": np.array(z_scores),
        "beta_means": np.array(beta_means),
        "beta_stds": np.array(beta_stds),
        "joint_scores": np.array(joint_scores_all),
        "mean_displacements": np.array(mean_displacements),
        "mean_velocities": np.array(mean_velocities),
        "mean_accelerations": np.array(mean_accelerations),
        "gates": gates,
        "step_sizes": step_sizes,
        "processed_frame_ratio": len(selected_indices) / T,
        "window_size": window_size,
        "tau": tau,
        "k_max": k_max,
        "score_mode": "online_z_sigmoid",
    }

    return sampled, metadata


def plot_selection(metadata: dict, output_path: str):
    frame_scores = metadata["frame_scores"]
    raw_frame_scores = metadata["raw_frame_scores"]
    beta_means = metadata["beta_means"]
    beta_stds = metadata["beta_stds"]
    selected_indices = metadata["selected_indices"]
    selected_reasons = metadata["selected_reasons"]
    mean_displacements = metadata["mean_displacements"]
    mean_velocities = metadata["mean_velocities"]
    mean_accelerations = metadata["mean_accelerations"]
    total_frames = len(frame_scores)
    selected_frames = len(selected_indices)
    processed_frame_ratio = metadata["processed_frame_ratio"]
    tau = metadata["tau"]
    k_max = metadata["k_max"]
    window_size = metadata["window_size"]
    x = np.arange(total_frames)

    fig, (ax_score, ax_beta, ax_motion) = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    ax_score.plot(
        x,
        frame_scores,
        label=(
            f"Online score sigmoid(z_t) "
            f"(tau={tau}, k_max={k_max}, window={window_size})"
        ),
    )
    ax_score.axhline(tau, color="red", linestyle="--", linewidth=1, label="tau")
    ax_score.set_ylim(-0.05, 1.05)

    reason_styles = {
        "warmup": {"marker": "^", "label": "Selected: warmup"},
        "tau": {"marker": "o", "label": "Selected: score >= tau"},
        "k_max": {"marker": "s", "label": "Selected: k_max forced"},
        "tau+k_max": {"marker": "D", "label": "Selected: tau + k_max"},
    }

    for reason, style in reason_styles.items():
        reason_indices = selected_indices[selected_reasons == reason]

        if len(reason_indices) == 0:
            continue

        ax_score.scatter(
            reason_indices,
            frame_scores[reason_indices],
            marker=style["marker"],
            label=style["label"],
        )

    ax_score.set_ylabel("Online Score")
    ax_score.set_title("VA-AFS Online Frame Selection")
    ax_score.text(
        0.01,
        0.95,
        (
            f"Frames: {total_frames} -> {selected_frames}\n"
            f"Selected ratio: {processed_frame_ratio:.3f} "
            f"({processed_frame_ratio * 100:.1f}%)\n"
            f"Warmup frames: first {window_size}\n"
            "z_t = (beta_t - mean(beta<t)) / (std(beta<t) + eps)"
        ),
        transform=ax_score.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    ax_score.legend(loc="upper right")

    ax_beta.plot(x, raw_frame_scores, label="Raw beta")
    ax_beta.plot(x, beta_means, linestyle="--", label="Prev beta mean")
    if 0.0 < tau < 1.0:
        adaptive_threshold = beta_means + logit(tau) * beta_stds
        ax_beta.plot(
            x,
            adaptive_threshold,
            linestyle=":",
            label="Raw beta threshold mapped from tau",
        )
    ax_beta.scatter(
        selected_indices,
        raw_frame_scores[selected_indices],
        marker="o",
        s=20,
        label="Selected raw beta",
    )
    ax_beta.set_ylabel("Raw Beta")
    ax_beta.legend(loc="upper right")

    ax_motion.plot(x, mean_displacements, label="Mean displacement")
    ax_motion.plot(x, mean_velocities, label="Mean velocity")
    ax_motion.plot(x, mean_accelerations, label="Mean acceleration")
    ax_motion.scatter(
        selected_indices,
        mean_displacements[selected_indices],
        marker="o",
        s=20,
        label="Selected frame displacement",
    )
    ax_motion.set_xlabel("Frame Index")
    ax_motion.set_ylabel("Change")
    ax_motion.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_output_dirs(output_dir: Path):
    threshold_npy_dir = output_dir / "threshold" / "npy"
    threshold_plot_dir = output_dir / "threshold" / "plots"
    selected_frames_dir = output_dir / "threshold" / "selected_frames"

    for path in (threshold_npy_dir, threshold_plot_dir, selected_frames_dir):
        path.mkdir(parents=True, exist_ok=True)

    return threshold_npy_dir, threshold_plot_dir, selected_frames_dir


def skeleton_stem_to_video_stem(stem: str):
    if stem.endswith("_skeleton"):
        return stem[: -len("_skeleton")]
    return stem


def find_video_for_skeleton(npy_path: Path, videos_dir: Path = Path("videos")):
    video_stem = skeleton_stem_to_video_stem(npy_path.stem)
    video_extensions = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

    if not videos_dir.exists():
        return None

    candidates = [
        path
        for path in videos_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in video_extensions
        and path.stem == video_stem
    ]

    if not candidates:
        return None

    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def save_selected_frame_images(
    video_path: Path,
    selected_indices: np.ndarray,
    output_dir: Path,
    prefix: str,
):
    for old_frame_path in output_dir.glob(f"{prefix}_frame_*.jpg"):
        old_frame_path.unlink()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    selected_set = set(int(idx) for idx in selected_indices)
    max_selected = max(selected_set) if selected_set else -1
    saved_count = 0
    frame_idx = 0

    while frame_idx <= max_selected:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        if frame_idx in selected_set:
            frame_path = output_dir / f"{prefix}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame_bgr)
            saved_count += 1

        frame_idx += 1

    cap.release()

    return saved_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", required=True, help="Skeleton npy path")
    parser.add_argument(
        "--video",
        default=None,
        help="Original video path. If omitted, the script tries to find it under videos/.",
    )
    parser.add_argument("--tau", type=float, default=TAU[-1])
    parser.add_argument("--k_max", type=int, default=K_MAX[-1])
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE[-1])
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    threshold_npy_dir, threshold_plot_dir, selected_frames_root = make_output_dirs(
        output_dir
    )

    skeleton = np.load(args.npy)  # (T, 33, 3)
    skeleton_xy = skeleton[:, :, :2]  # (T, 33, 2)

    # NaN이 있는 프레임은 0으로 임시 처리.
    # 나중에는 interpolation으로 바꾸는 게 좋음.
    skeleton_xy = np.nan_to_num(skeleton_xy, nan=0.0)

    sampled, metadata = va_afs_threshold_sampling(
        skeleton_xy=skeleton_xy,
        window_size=args.window_size,
        tau=args.tau,
        k_max=args.k_max,
    )

    stem = Path(args.npy).stem

    run_name = f"{stem}_tau{args.tau}_k{args.k_max}"
    sampled_path = threshold_npy_dir / f"{run_name}_sampled.npy"
    plot_path = threshold_plot_dir / f"{run_name}_selection.png"
    selected_frames_dir = selected_frames_root / run_name

    np.save(sampled_path, sampled)
    plot_selection(metadata, str(plot_path))

    video_path = (
        Path(args.video) if args.video else find_video_for_skeleton(Path(args.npy))
    )
    saved_frame_count = 0

    if video_path:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        selected_frames_dir.mkdir(parents=True, exist_ok=True)
        saved_frame_count = save_selected_frame_images(
            video_path=video_path,
            selected_indices=metadata["selected_indices"],
            output_dir=selected_frames_dir,
            prefix=skeleton_stem_to_video_stem(stem),
        )

    print(f"Original shape: {skeleton_xy.shape}")
    print(f"Sampled shape: {sampled.shape}")
    print(f"Processed frame ratio: {metadata['processed_frame_ratio']:.3f}")
    print(f"Selected indices: {metadata['selected_indices'][:30]} ...")
    print(f"Saved sampled skeleton: {sampled_path}")
    print(f"Saved plot: {plot_path}")
    if video_path:
        print(f"Saved selected frame images: {saved_frame_count} files")
        print(f"Selected frame image dir: {selected_frames_dir}")
    else:
        print("Selected frame images were not saved because no source video was found.")
        print("Pass --video /path/to/video to save selected frame images.")


if __name__ == "__main__":
    main()
