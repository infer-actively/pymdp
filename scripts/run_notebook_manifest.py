#!/usr/bin/env python3
"""Run nbval notebook tests from a manifest file."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> list[str]:
    notebooks = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    missing = [entry for entry in notebooks if not (REPO_ROOT / entry).exists()]
    if missing:
        missing_str = "\n".join(f"  - {entry}" for entry in missing)
        raise FileNotFoundError(
            f"Manifest {path.relative_to(REPO_ROOT).as_posix()} references missing "
            f"notebooks:\n{missing_str}"
        )
    return notebooks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run notebook tests listed in a repo-relative manifest file."
    )
    parser.add_argument("manifest", help="Repo-relative path to the notebook manifest.")
    parser.add_argument(
        "--strict-output",
        action="store_true",
        help="Use strict nbval output matching instead of --nbval-lax.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional pytest arguments. Prefix them with -- to separate them.",
    )
    return parser.parse_args()


def has_explicit_numprocesses(pytest_args: list[str]) -> bool:
    for arg in pytest_args:
        if arg == "-n" or arg.startswith("-n") or arg.startswith("--numprocesses"):
            return True
    return False


def main() -> int:
    args = parse_args()
    manifest_path = (REPO_ROOT / args.manifest).resolve()

    try:
        manifest_rel = manifest_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Manifest must live inside the repository: {args.manifest}") from exc

    notebooks = load_manifest(manifest_path)
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    nbval_flag = "--nbval" if args.strict_output else "--nbval-lax"
    worker_args = [] if has_explicit_numprocesses(pytest_args) else ["-n0"]
    command = [
        sys.executable,
        "-m",
        "pytest",
        nbval_flag,
        *worker_args,
        *pytest_args,
        *notebooks,
    ]

    print(
        f"Running {len(notebooks)} notebook(s) from "
        f"{manifest_rel.as_posix()} with {nbval_flag}."
    )
    if worker_args:
        print("Notebook execution defaults to -n0 to avoid xdist/nbval conflicts.")
    for notebook in notebooks:
        print(f"  - {notebook}")

    completed = subprocess.run(command, cwd=REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
