#!/usr/bin/env python3
"""Tier-aware notebook sanitation hooks for pre-commit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "test" / "notebooks"
CI_MANIFEST = MANIFEST_DIR / "ci_notebooks.txt"
NIGHTLY_MANIFEST = MANIFEST_DIR / "nightly_notebooks.txt"
TOP_LEVEL_METADATA_KEYS = ("kernelspec", "language_info")
NBSTRIPOUT_EXTRA_KEYS = "metadata.kernelspec metadata.language_info"


def load_manifest(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def to_repo_relative(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def classify_notebooks(path_args: list[str]) -> tuple[list[Path], list[Path]]:
    ci_manifest = load_manifest(CI_MANIFEST)
    nightly_manifest = load_manifest(NIGHTLY_MANIFEST)
    ci_paths: list[Path] = []
    nightly_paths: list[Path] = []

    for path_arg in path_args:
        rel_path = to_repo_relative(path_arg)
        if rel_path is None or not rel_path.endswith(".ipynb"):
            continue

        file_path = REPO_ROOT / rel_path
        if not file_path.exists():
            continue

        if rel_path in ci_manifest:
            ci_paths.append(file_path)
        elif rel_path in nightly_manifest:
            nightly_paths.append(file_path)

    return ci_paths, nightly_paths


def sanitize_ci_notebooks(paths: list[Path]) -> list[str]:
    modified_paths: list[str] = []

    for path in paths:
        notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        changed = False
        for key in TOP_LEVEL_METADATA_KEYS:
            if key in notebook.metadata:
                del notebook.metadata[key]
                changed = True

        if changed:
            nbformat.write(notebook, path)
            modified_paths.append(path.relative_to(REPO_ROOT).as_posix())

    return modified_paths


def sanitize_nightly_notebooks(paths: list[Path]) -> None:
    if not paths:
        return

    command = [
        sys.executable,
        "-m",
        "nbstripout",
        "--keep-output",
        f"--extra-keys={NBSTRIPOUT_EXTRA_KEYS}",
        *[path.relative_to(REPO_ROOT).as_posix() for path in paths],
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def validate_ci_notebooks(paths: list[Path]) -> list[str]:
    violations: list[str] = []

    for path in paths:
        notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        for index, cell in enumerate(notebook.cells, start=1):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs") and cell.get("execution_count") is None:
                violations.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()}: code cell {index} "
                    "has outputs but a null execution_count"
                )

    return violations


def run_sanitize(path_args: list[str]) -> int:
    ci_paths, nightly_paths = classify_notebooks(path_args)
    ci_modified = sanitize_ci_notebooks(ci_paths)
    sanitize_nightly_notebooks(nightly_paths)

    if ci_modified:
        print("Removed noisy top-level metadata from CI-tier notebooks:")
        for rel_path in ci_modified:
            print(f"  - {rel_path}")

    if nightly_paths:
        print("Sanitized nightly-tier notebooks with nbstripout:")
        for path in nightly_paths:
            print(f"  - {path.relative_to(REPO_ROOT).as_posix()}")

    return 0


def run_validate_ci(path_args: list[str]) -> int:
    ci_paths, _ = classify_notebooks(path_args)
    violations = validate_ci_notebooks(ci_paths)

    if not violations:
        return 0

    print(
        "CI-tier notebooks must keep execution counts for code cells with saved "
        "outputs so strict nbval can replay them."
    )
    for violation in violations:
        print(f"  - {violation}")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier-aware notebook sanitation hooks for pre-commit."
    )
    parser.add_argument(
        "command",
        choices=("sanitize", "validate-ci"),
        help="Hook action to execute.",
    )
    parser.add_argument("paths", nargs="*", help="Notebook paths supplied by pre-commit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "sanitize":
        return run_sanitize(args.paths)
    return run_validate_ci(args.paths)


if __name__ == "__main__":
    raise SystemExit(main())
