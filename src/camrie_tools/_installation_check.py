"""Installation checks for CAMRIE tools."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PYTHON_DEPENDENCIES = ("numpy", "SimpleITK", "vtk", "tqdm", "pypulseq", "matplotlib")


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


def _run_julia_check(expression: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["julia", "-e", expression],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _check_julia_dependencies() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []

    if shutil.which("julia") is None:
        return ["julia was not found on PATH"], notes

    koma = _run_julia_check('using KomaInterface; println("KomaInterface OK")')
    if koma.returncode != 0:
        errors.append("KomaInterface.jl is not importable:\n" + koma.stdout.strip())
    else:
        notes.append(koma.stdout.strip())

    cuda = _run_julia_check(
        'using CUDA; println("CUDA.jl OK"); println("CUDA functional: ", CUDA.functional())'
    )
    if cuda.returncode != 0:
        errors.append("CUDA.jl is not importable:\n" + cuda.stdout.strip())
    else:
        notes.append(cuda.stdout.strip())

    return errors, notes


def main() -> int:
    print("Checking CAMRIE Python package...")
    errors = _check_python_dependencies()
    errors.extend(_check_package_data())

    print("Checking CAMRIE Julia packages...")
    julia_errors, notes = _check_julia_dependencies()
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
