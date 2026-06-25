"""Helpers for installing Julia dependencies used by CAMRIE tools."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys


BASE_PACKAGES = ("KomaInterface",)
GPU_PACKAGES = ("CUDA",)


def build_julia_command(gpu: bool = False) -> list[str]:
    packages = list(BASE_PACKAGES)
    if gpu:
        packages.extend(GPU_PACKAGES)

    julia_expr = (
        "import Pkg; "
        f"Pkg.add({json.dumps(packages)}); "
        "Pkg.update(); "
        "Pkg.precompile()"
    )
    return ["julia", "-e", julia_expr]


def install_julia_deps(gpu: bool = False) -> int:
    if shutil.which("julia") is None:
        print(
            "ERROR: julia was not found on PATH. Install Julia first, then re-run this command.",
            file=sys.stderr,
        )
        return 127

    cmd = build_julia_command(gpu=gpu)
    print("Installing Julia packages: " + ", ".join(BASE_PACKAGES + (GPU_PACKAGES if gpu else ())))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install Julia dependencies for CAMRIE Koma simulations."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Install CUDA.jl in addition to the CPU Julia dependencies.",
    )
    args = parser.parse_args(argv)
    return install_julia_deps(gpu=args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
