import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ZIP_TARGETS = {
    "all_sqe.zip": Path("BlockGCN/data/NW-UCLA"),
    "nturgbd_skeletons_s001_to_s017.zip": Path("BlockGCN/data/nturgbd_raw"),
    "nturgbd_skeletons_s018_to_s032.zip": Path("BlockGCN/data/nturgbd_raw"),
    "videos.zip": Path("VA-AFS"),
}

EXPECTED_PATHS = {
    "all_sqe.zip": Path("BlockGCN/data/NW-UCLA/all_sqe"),
    "nturgbd_skeletons_s001_to_s017.zip": Path(
        "BlockGCN/data/nturgbd_raw/nturgb+d_skeletons"
    ),
    "nturgbd_skeletons_s018_to_s032.zip": Path(
        "BlockGCN/data/nturgbd_raw/nturgb+d_skeletons120"
    ),
    "videos.zip": Path("VA-AFS/videos"),
}


def has_real_files(path: Path) -> bool:
    if not path.exists():
        return False
    return any(child.is_file() and child.name != ".gitkeep" for child in path.rglob("*"))


def extract_zip(zip_path: Path, target_dir: Path, expected_path: Path, force: bool) -> None:
    if has_real_files(expected_path) and not force:
        print(f"skip: {expected_path} already has files")
        return

    if expected_path.exists() and force:
        shutil.rmtree(expected_path)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"unzip: {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)


def flatten_nested_dir(parent: Path, nested_name: str) -> None:
    nested = parent / nested_name
    if not parent.exists() or not nested.exists() or not nested.is_dir():
        return

    print(f"fix nested directory: {nested} -> {parent}")
    for child in nested.iterdir():
        target = parent / child.name
        if target.exists():
            continue
        shutil.move(str(child), str(target))

    try:
        nested.rmdir()
    except OSError:
        pass


def install_requirements(project_root: Path) -> None:
    requirements_path = project_root / "requirements-colab.txt"
    torchlight_dir = project_root / "BlockGCN" / "torchlight"

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(torchlight_dir)],
        check=True,
    )


def verify_imports(project_root: Path) -> None:
    env = os.environ.copy()
    torchlight_path = str(project_root / "BlockGCN" / "torchlight")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        torchlight_path
        if not current_pythonpath
        else f"{torchlight_path}{os.pathsep}{current_pythonpath}"
    )
    code = (
        "import cv2, matplotlib, mediapipe, numpy, pandas, sklearn, torch, yaml; "
        "import torch_topological; "
        "import torchlight; "
        "print('verify ok:', 'cuda=', torch.cuda.is_available())"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare this VA-AFS repository for Google Colab or VS Code Colab. "
            "Default data location is ../data relative to src/."
        )
    )
    parser.add_argument(
        "--data_dir",
        default="../data",
        help="Directory containing all_sqe.zip, NTU skeleton zip files, and videos.zip.",
    )
    parser.add_argument(
        "--project_root",
        default=None,
        help="Repository src directory. Defaults to the directory containing this file.",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract expected folders.")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install requirements-colab.txt and local BlockGCN/torchlight.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify Python imports after extraction/install.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent
    data_dir = (project_root / args.data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for zip_name, relative_target in ZIP_TARGETS.items():
        zip_path = data_dir / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Required zip not found: {zip_path}")

        extract_zip(
            zip_path=zip_path,
            target_dir=project_root / relative_target,
            expected_path=project_root / EXPECTED_PATHS[zip_name],
            force=args.force,
        )

    flatten_nested_dir(
        project_root / "BlockGCN/data/nturgbd_raw/nturgb+d_skeletons",
        "nturgb+d_skeletons",
    )
    flatten_nested_dir(
        project_root / "BlockGCN/data/nturgbd_raw/nturgb+d_skeletons120",
        "nturgb+d_skeletons120",
    )

    if args.install:
        install_requirements(project_root)

    if args.verify:
        verify_imports(project_root)

    print("Colab setup complete.")
    print(f"project_root: {project_root}")
    print(f"data_dir    : {data_dir}")


if __name__ == "__main__":
    main()
