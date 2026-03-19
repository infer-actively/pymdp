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
LEGACY_NOTEBOOK_PREFIX = "examples/legacy/"
TOP_LEVEL_METADATA_KEYS = ("kernelspec", "language_info")
NBSTRIPOUT_EXTRA_KEYS = "metadata.kernelspec metadata.language_info"


def load_manifest(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def strip_top_level_metadata(notebook: nbformat.NotebookNode) -> bool:
    changed = False
    for key in TOP_LEVEL_METADATA_KEYS:
        if key in notebook.metadata:
            del notebook.metadata[key]
            changed = True
    return changed


def canonicalize_execution_counts(notebook: nbformat.NotebookNode) -> bool:
    """
    Normalize execution counts for output-bearing code cells.

    `nbval` only needs output-bearing reference cells to have non-null execution
    counts; the exact local kernel history is not meaningful. Canonicalizing the
    counts keeps the notebooks stable under version control while preserving the
    invariant that saved outputs look like they came from an executed notebook.
    """

    changed = False
    next_count = 1

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        if not outputs:
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
            continue

        if cell.get("execution_count") != next_count:
            cell["execution_count"] = next_count
            changed = True

        for output in outputs:
            if "execution_count" in output and output["execution_count"] != next_count:
                output["execution_count"] = next_count
                changed = True

        next_count += 1

    return changed


def to_repo_relative(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def classify_notebooks(path_args: list[str]) -> tuple[list[Path], list[Path], list[str]]:
    ci_manifest = load_manifest(CI_MANIFEST)
    nightly_manifest = load_manifest(NIGHTLY_MANIFEST)
    ci_paths: list[Path] = []
    nightly_paths: list[Path] = []
    unclassified_paths: list[str] = []

    for path_arg in path_args:
        rel_path = to_repo_relative(path_arg)
        if rel_path is None or not rel_path.endswith(".ipynb"):
            continue

        file_path = REPO_ROOT / rel_path
        if not file_path.exists():
            continue

        if not rel_path.startswith("examples/") or rel_path.startswith(
            LEGACY_NOTEBOOK_PREFIX
        ):
            continue

        if rel_path in ci_manifest:
            ci_paths.append(file_path)
        elif rel_path in nightly_manifest:
            nightly_paths.append(file_path)
        else:
            unclassified_paths.append(rel_path)

    return ci_paths, nightly_paths, sorted(set(unclassified_paths))


def report_unclassified_notebooks(paths: list[str]) -> int:
    if not paths:
        return 0

    print(
        "Every non-legacy source notebook under examples/ must be listed in either "
        "test/notebooks/ci_notebooks.txt or test/notebooks/nightly_notebooks.txt."
    )
    print("Update the notebook manifests before committing these files:")
    for rel_path in paths:
        print(f"  - {rel_path}")
    return 1


def sanitize_ci_notebooks(paths: list[Path]) -> list[str]:
    modified_paths: list[str] = []

    for path in paths:
        notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        changed = strip_top_level_metadata(notebook)
        changed = canonicalize_execution_counts(notebook) or changed

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

    for path in paths:
        notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        changed = strip_top_level_metadata(notebook)
        changed = canonicalize_execution_counts(notebook) or changed
        if changed:
            nbformat.write(notebook, path)


def validate_manifest_notebooks(paths: list[Path]) -> list[str]:
    violations: list[str] = []

    for path in paths:
        notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        for index, cell in enumerate(notebook.cells, start=1):
            if cell.get("cell_type") != "code":
                continue
            outputs = cell.get("outputs", [])
            if not outputs:
                continue

            if cell.get("execution_count") is None:
                violations.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()}: code cell {index} "
                    "has outputs but a null execution_count"
                )
                continue

            for output in outputs:
                if "execution_count" in output and output["execution_count"] != cell.get(
                    "execution_count"
                ):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}: code cell {index} "
                        "has an output execution_count that does not match the cell"
                    )

    return violations


def run_sanitize(path_args: list[str]) -> int:
    ci_paths, nightly_paths, unclassified_paths = classify_notebooks(path_args)
    if unclassified_paths:
        return report_unclassified_notebooks(unclassified_paths)

    ci_modified = sanitize_ci_notebooks(ci_paths)
    sanitize_nightly_notebooks(nightly_paths)

    if ci_modified:
        print("Sanitized CI-tier notebooks:")
        for rel_path in ci_modified:
            print(f"  - {rel_path}")

    if nightly_paths:
        print("Sanitized nightly-tier notebooks with nbstripout:")
        for path in nightly_paths:
            print(f"  - {path.relative_to(REPO_ROOT).as_posix()}")

    return 0


def run_validate_counts(path_args: list[str]) -> int:
    ci_paths, nightly_paths, unclassified_paths = classify_notebooks(path_args)
    if unclassified_paths:
        return report_unclassified_notebooks(unclassified_paths)

    violations = validate_manifest_notebooks(ci_paths + nightly_paths)

    if not violations:
        return 0

    print(
        "Manifest-tested notebooks that keep saved outputs must also keep "
        "non-null execution counts for those cells."
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
        choices=("sanitize", "validate-ci", "validate-counts"),
        help="Hook action to execute.",
    )
    parser.add_argument("paths", nargs="*", help="Notebook paths supplied by pre-commit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "sanitize":
        return run_sanitize(args.paths)
    if args.command in {"validate-ci", "validate-counts"}:
        return run_validate_counts(args.paths)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
