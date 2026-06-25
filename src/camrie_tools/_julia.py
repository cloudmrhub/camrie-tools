"""Helpers for installing Julia dependencies used by CAMRIE tools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional


DEFAULT_REPOSITORY_URL = "git@github.com:cloudmrhub/KomaInterface.jl.git"
DEFAULT_BRANCH = "master"
BASE_PACKAGES = ("KomaInterface",)
GPU_PACKAGES = ("CUDA",)
DEFAULT_PACKAGES = BASE_PACKAGES + GPU_PACKAGES


def _normalize_repository_url(repository_url: Optional[str]) -> Optional[str]:
    if repository_url is None:
        return None

    if repository_url.startswith("git@") and ":" in repository_url:
        user_host, path = repository_url.split(":", 1)
        return f"ssh://{user_host}/{path}"
    return repository_url


def _package_spec(repository_url: Optional[str], branch: Optional[str]) -> str:
    julia_repository_url = _normalize_repository_url(repository_url)
    if julia_repository_url is None:
        return json.dumps(list(DEFAULT_PACKAGES))

    kwargs = [f"url={json.dumps(julia_repository_url)}"]
    if branch is not None:
        kwargs.append(f"rev={json.dumps(branch)}")
    return f"Pkg.PackageSpec({', '.join(kwargs)})"


def build_julia_command(
    repository_url: Optional[str] = DEFAULT_REPOSITORY_URL,
    branch: Optional[str] = DEFAULT_BRANCH,
) -> list[str]:
    if repository_url is None:
        package_commands = [f"Pkg.add({json.dumps(list(DEFAULT_PACKAGES))})"]
    else:
        package_commands = [f"Pkg.add({_package_spec(repository_url, branch)})"]
    if repository_url is not None:
        package_commands.append(f"Pkg.add({json.dumps(list(GPU_PACKAGES))})")

    julia_expr = (
        "import Pkg; "
        + "; ".join(package_commands)
        + "; "
        "Pkg.update(); "
        "Pkg.precompile()"
    )
    return ["julia", "-e", julia_expr]


def install_julia_deps(
    repository_url: Optional[str] = DEFAULT_REPOSITORY_URL,
    branch: Optional[str] = DEFAULT_BRANCH,
    install_dir: Optional[str] = None,
) -> int:
    if shutil.which("julia") is None:
        print(
            "ERROR: julia was not found on PATH. Install Julia first, then re-run this command.",
            file=sys.stderr,
        )
        return 127

    julia_repository_url = _normalize_repository_url(repository_url)
    cmd = build_julia_command(repository_url=julia_repository_url, branch=branch)
    packages = list(DEFAULT_PACKAGES)
    print("Installing Julia packages: " + ", ".join(packages))
    if julia_repository_url is not None:
        branch_text = f" at branch {branch}" if branch is not None else ""
        print(f"Using KomaInterface repository {julia_repository_url}{branch_text}")
        if repository_url != julia_repository_url:
            print(f"Normalized repository URL from {repository_url}")

    env = None
    if install_dir is not None:
        install_path = Path(install_dir).expanduser()
        install_path.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["JULIA_DEPOT_PATH"] = str(install_path)
        print(f"Using Julia depot: {install_path}")

    if julia_repository_url is not None and julia_repository_url.startswith("ssh://"):
        env = os.environ.copy() if env is None else env
        env.setdefault("JULIA_PKG_USE_CLI_GIT", "true")

    completed = subprocess.run(cmd, check=False, env=env)
    return completed.returncode


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the full Julia dependency stack for CAMRIE Koma simulations."
    )
    parser.add_argument(
        "--repository-url",
        default=DEFAULT_REPOSITORY_URL,
        help="Install KomaInterface.jl from this git repository URL instead of the registered package.",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Install KomaInterface.jl from this git branch, tag, or revision.",
    )
    parser.add_argument(
        "--install-dir",
        help="Install Julia packages into this Julia depot directory.",
    )
    args = parser.parse_args(argv)
    return install_julia_deps(
        repository_url=args.repository_url,
        branch=args.branch,
        install_dir=args.install_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
