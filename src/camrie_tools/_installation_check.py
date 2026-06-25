"""Installation checks for CAMRIE tools."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence


PYTHON_DEPENDENCIES = ("numpy", "SimpleITK", "vtk", "tqdm", "pypulseq", "matplotlib")
DEFAULT_JULIA_PROJECT = Path.home() / ".camrie" / "julia"


def _check_python_dependencies() -> list[str]:
    errors: list[str] = []
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    for module_name in PYTHON_DEPENDENCIES:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"missing Python dependency {module_name}: {exc}")
    return errors


def _check_package_data() -> list[str]:
    import camrie_tools

    simulate_batch = Path(camrie_tools.simulate_batch_path())
    if not simulate_batch.exists():
        return [f"missing bundled Julia script: {simulate_batch}"]
    return []


def _run_julia_check(
    expression: str,
    project_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["julia", f"--project={project_dir}", "--startup-file=no", "-e", expression],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _check_julia_dependencies(
    project_dir: Path,
    cpu: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []

    if shutil.which("julia") is None:
        return ["julia was not found on PATH"], notes

    if not project_dir.exists():
        return [
            f"CAMRIE Julia project was not found: {project_dir}. "
            "Run `camrie-install-julia --cpu` or `camrie-install-julia` first."
        ], notes

    os.environ["CAMRIE_JULIA_PROJECT"] = str(project_dir)
    os.environ["JULIA_PROJECT"] = str(project_dir)

    koma = _run_julia_check('using KomaInterface; println("KomaInterface OK")', project_dir)
    if koma.returncode != 0:
        errors.append("KomaInterface.jl is not importable:\n" + koma.stdout.strip())
    else:
        notes.append(koma.stdout.strip())

    if cpu:
        notes.append("CUDA.jl check skipped for CPU-only installation.")
        return errors, notes

    cuda = _run_julia_check(
        'using CUDA; println("CUDA.jl OK"); println("CUDA functional: ", CUDA.functional())',
        project_dir,
    )
    if cuda.returncode != 0:
        errors.append("CUDA.jl is not importable:\n" + cuda.stdout.strip())
    else:
        notes.append(cuda.stdout.strip())

    return errors, notes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check a CAMRIE tools installation.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Check a CPU-only Julia installation and skip CUDA.jl import checks.",
    )
    parser.add_argument(
        "--project-dir",
        default=str(DEFAULT_JULIA_PROJECT),
        help=f"CAMRIE Julia project directory. Default: {DEFAULT_JULIA_PROJECT}",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    project_dir = Path(args.project_dir).expanduser()

    print("Checking CAMRIE Python package...")
    errors = _check_python_dependencies()
    errors.extend(_check_package_data())

    print("Checking CAMRIE Julia packages...")
    print(f"Julia project: {project_dir}")
    julia_errors, notes = _check_julia_dependencies(project_dir=project_dir, cpu=args.cpu)
    errors.extend(julia_errors)

    for note in notes:
        if note:
            print(note)

    if errors:
        print("\nInstallation check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("CAMRIE installation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
