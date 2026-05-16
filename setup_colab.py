import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


VERIFY_IMPORTS = (
    "cv2",
    "matplotlib",
    "mediapipe",
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "torch_topological",
    "torchlight",
    "yaml",
)

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


def log(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    log("cmd: " + " ".join(command))
    result = subprocess.run(command, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(f"command failed with exit code {result.returncode}: {' '.join(command)}")


def has_real_files(path: Path) -> bool:
    if not path.exists():
        return False
    return any(child.is_file() and child.name != ".gitkeep" for child in path.rglob("*"))


def extract_zip(zip_path: Path, target_dir: Path, expected_path: Path, force: bool) -> None:
    if has_real_files(expected_path) and not force:
        log(f"skip unzip: {expected_path} already has files")
        return

    if expected_path.exists() and force:
        log(f"remove existing extracted dir: {expected_path}")
        shutil.rmtree(expected_path)

    target_dir.mkdir(parents=True, exist_ok=True)
    log(f"unzip: {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)


def flatten_nested_dir(parent: Path, nested_name: str) -> None:
    nested = parent / nested_name
    if not parent.exists() or not nested.exists() or not nested.is_dir():
        return

    log(f"fix nested directory: {nested} -> {parent}")
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
    missing = missing_imports(project_root)
    if not missing:
        log("skip install: Colab requirements already import correctly")
        return

    log("missing imports before install: " + ", ".join(module for module, _ in missing))
    requirements_path = project_root / "requirements-colab.txt"
    torchlight_dir = project_root / "BlockGCN" / "torchlight"

    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(requirements_path),
        ]
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-e",
            str(torchlight_dir),
        ]
    )


def make_verify_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    torchlight_path = str(project_root / "BlockGCN" / "torchlight")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        torchlight_path
        if not current_pythonpath
        else f"{torchlight_path}{os.pathsep}{current_pythonpath}"
    )
    return env


def imports_available(project_root: Path) -> bool:
    return not missing_imports(project_root)


def missing_imports(project_root: Path) -> list[tuple[str, str]]:
    missing = []
    env = make_verify_env(project_root)
    for module in VERIFY_IMPORTS:
        code = f"import {module}"
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            missing.append((module, detail[-1] if detail else "unknown import error"))
    return missing


def verify_imports(project_root: Path) -> None:
    missing = missing_imports(project_root)
    if missing:
        details = "\n".join(f"  - {module}: {error}" for module, error in missing)
        raise RuntimeError(f"Import verification failed:\n{details}")

    code = (
        "import torch; "
        "print('verify ok:', 'cuda=', torch.cuda.is_available(), flush=True)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=make_verify_env(project_root),
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"verify command failed with exit code {result.returncode}")


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
    log(f"project_root: {project_root}")
    log(f"data_dir: {data_dir}")

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
        log("install requirements")
        install_requirements(project_root)

    if args.verify:
        log("verify imports")
        verify_imports(project_root)

    log("Colab setup complete.")


if __name__ == "__main__":
    main()
