import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def make_output_dirs(output_dir: Path):
    skeleton_npy_dir = output_dir / "skeleton" / "npy"
    skeleton_csv_dir = output_dir / "skeleton" / "csv"
    preview_dir = output_dir / "previews" / "mp4"

    for path in (skeleton_npy_dir, skeleton_csv_dir, preview_dir):
        path.mkdir(parents=True, exist_ok=True)

    return skeleton_npy_dir, skeleton_csv_dir, preview_dir


def extract_pose_from_video(
    video_path: str,
    model_path: str = "models/pose_landmarker_lite.task",
    output_dir: str = "outputs",
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    skeleton_npy_dir, skeleton_csv_dir, preview_dir = make_output_dirs(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Download pose_landmarker_full.task and put it in models/."
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    stem = video_path.stem

    npy_path = skeleton_npy_dir / f"{stem}_skeleton.npy"
    csv_path = skeleton_csv_dir / f"{stem}_skeleton.csv"
    preview_path = preview_dir / f"{stem}_preview.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )

    all_frames = []
    rows = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_idx in tqdm(
            range(total_frames), desc=f"Extracting {video_path.name}"
        ):
            ret, frame_bgr = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )

            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # MediaPipe PoseLandmarker: 33 landmarks.
            # 저장 shape: (33, 3) = x, y, visibility/presence fallback
            joints = np.full((33, 3), np.nan, dtype=np.float32)

            if result.pose_landmarks:
                pose_landmarks = result.pose_landmarks[0]

                for j, lm in enumerate(pose_landmarks):
                    # x, y는 normalized coordinate. 0~1 범위.
                    # visibility가 없는 버전/환경도 있을 수 있어 getattr로 안전 처리.
                    visibility = getattr(lm, "visibility", np.nan)
                    presence = getattr(lm, "presence", np.nan)

                    conf = visibility
                    if np.isnan(conf):
                        conf = presence

                    joints[j, 0] = lm.x
                    joints[j, 1] = lm.y
                    joints[j, 2] = conf

                    rows.append(
                        {
                            "frame": frame_idx,
                            "joint": j,
                            "x": lm.x,
                            "y": lm.y,
                            "confidence": conf,
                        }
                    )

                draw_landmarks_on_frame(frame_bgr, pose_landmarks, width, height)

            all_frames.append(joints)
            writer.write(frame_bgr)

    cap.release()
    writer.release()

    skeleton = np.stack(all_frames, axis=0)  # (T, 33, 3)
    np.save(npy_path, skeleton)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"Saved skeleton npy: {npy_path}")
    print(f"Saved skeleton csv: {csv_path}")
    print(f"Saved preview video: {preview_path}")
    print(f"skeleton.shape = {skeleton.shape}")

    return skeleton


def draw_landmarks_on_frame(frame_bgr, pose_landmarks, width: int, height: int):
    """
    최신 Tasks API 결과를 간단히 OpenCV로 직접 그리는 함수.
    mp.solutions.drawing_utils를 쓰지 않기 위해 직접 구현.
    """
    # MediaPipe Pose 33 landmark connection index pairs.
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (27, 29),
        (27, 31),
        (29, 31),
        (24, 26),
        (26, 28),
        (28, 30),
        (28, 32),
        (30, 32),
    ]

    points = []

    for lm in pose_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))

    for a, b in connections:
        if a < len(points) and b < len(points):
            cv2.line(frame_bgr, points[a], points[b], (0, 255, 255), 2)

    for x, y in points:
        cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_lite.task",
        help="Path to pose_landmarker .task model",
    )
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    extract_pose_from_video(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
