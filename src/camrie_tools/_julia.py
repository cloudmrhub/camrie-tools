"""Install Julia dependencies used by CAMRIE tools.

The installer creates a dedicated Julia project for CAMRIE so its dependencies
do not conflict with packages installed in the user's default Julia environment.

Default behavior:
    - Install KomaInterface from the configured Git repository.
    - Install CUDA.jl for NVIDIA GPU support.

CPU-only behavior:
    - Install KomaInterface without explicitly installing CUDA.jl.
    - Selected with the ``--cpu`` command-line option.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence


DEFAULT_REPOSITORY_URL = (
    "https://github.com/cloudmrhub/KomaInterface.jl.git"
)
DEFAULT_BRANCH = "master"

DEFAULT_PROJECT_DIR = Path.home() / ".camrie" / "julia"

BASE_PACKAGES = ("KomaInterface",)
GPU_PACKAGES = ("CUDA",)


def _normalize_repository_url(
    repository_url: Optional[str],
) -> Optional[str]:
    """Normalize repository URLs for Julia's package manager.

    Julia accepts HTTPS URLs directly. SCP-style Git URLs such as

        git@github.com:cloudmrhub/KomaInterface.jl.git

    are converted to:

        ssh://git@github.com/cloudmrhub/KomaInterface.jl.git
    """
    if repository_url is None:
        return None

    repository_url = repository_url.strip()

    if not repository_url:
        return None

    if repository_url.startswith("git@") and ":" in repository_url:
        user_host, path = repository_url.split(":", 1)
        return f"ssh://{user_host}/{path}"

    return repository_url


def _julia_string(value: str) -> str:
    """Return a safely quoted Julia string literal."""
    return json.dumps(value)


def _repository_package_spec(
    repository_url: str,
    branch: Optional[str],
) -> str:
    """Build a Julia Pkg.PackageSpec expression for a Git repository."""
    arguments = [f"url={_julia_string(repository_url)}"]

    if branch:
        arguments.append(f"rev={_julia_string(branch)}")

    return f"Pkg.PackageSpec({', '.join(arguments)})"


def _build_package_commands(
    repository_url: Optional[str],
    branch: Optional[str],
    cpu: bool,
) -> list[str]:
    """Build Julia Pkg commands for the requested installation mode."""
    commands: list[str] = []

    if repository_url is None:
        commands.append(
            f"Pkg.add({_julia_string(BASE_PACKAGES[0])})"
        )
    else:
        package_spec = _repository_package_spec(
            repository_url=repository_url,
            branch=branch,
        )
        commands.append(f"Pkg.add({package_spec})")

    if not cpu:
        for package in GPU_PACKAGES:
            commands.append(
                f"Pkg.add({_julia_string(package)})"
            )

    return commands


def build_julia_command(
    project_dir: str | Path,
    repository_url: Optional[str] = DEFAULT_REPOSITORY_URL,
    branch: Optional[str] = DEFAULT_BRANCH,
    cpu: bool = False,
    update: bool = False,
) -> list[str]:
    """Build the Julia subprocess command used for installation.

    The command activates a dedicated Julia project, adds the selected
    dependencies, instantiates the environment, and precompiles it.

    By default it does not call ``Pkg.update()`` because an installer should
    not update unrelated packages in an existing environment. Pass
    ``update=True`` for the development/release-maintenance path that refreshes
    Julia packages before precompilation.
    """
    project_path = Path(project_dir).expanduser().resolve()
    normalized_url = _normalize_repository_url(repository_url)

    package_commands = _build_package_commands(
        repository_url=normalized_url,
        branch=branch,
        cpu=cpu,
    )

    julia_statements = [
        "import Pkg",
        f"Pkg.activate({_julia_string(str(project_path))})",
        *package_commands,
        *(["Pkg.update()"] if update else []),
        "Pkg.instantiate()",
        "Pkg.precompile()",
        'println("CAMRIE Julia dependencies installed successfully.")',
        'println("Julia project: ", Base.active_project())',
    ]

    julia_expression = "; ".join(julia_statements)

    return [
        "julia",
        f"--project={project_path}",
        "--startup-file=no",
        "-e",
        julia_expression,
    ]


def _build_environment(
    install_dir: Optional[str],
    repository_url: Optional[str],
) -> Optional[dict[str, str]]:
    """Build the environment passed to the Julia subprocess."""
    env: Optional[dict[str, str]] = None

    if install_dir is not None:
        depot_path = Path(install_dir).expanduser().resolve()
        depot_path.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["JULIA_DEPOT_PATH"] = str(depot_path)

        print(f"Using Julia depot: {depot_path}")

    normalized_url = _normalize_repository_url(repository_url)

    if normalized_url and normalized_url.startswith("ssh://"):
        if env is None:
            env = os.environ.copy()

        # Use the system Git client so SSH keys and SSH configuration work
        # in the same manner as a normal command-line Git operation.
        env.setdefault("JULIA_PKG_USE_CLI_GIT", "true")

    return env


def _print_installation_summary(
    project_dir: Path,
    repository_url: Optional[str],
    branch: Optional[str],
    cpu: bool,
    update: bool,
) -> None:
    """Print a human-readable installation summary."""
    mode = "CPU-only" if cpu else "GPU-capable"

    packages = list(BASE_PACKAGES)
    if not cpu:
        packages.extend(GPU_PACKAGES)

    print("=" * 72)
    print("CAMRIE Julia dependency installer")
    print("=" * 72)
    print(f"Installation mode : {mode}")
    print(f"Julia project     : {project_dir}")
    print(f"Packages          : {', '.join(packages)}")
    print(f"Update packages   : {'yes' if update else 'no'}")

    if repository_url is None:
        print("KomaInterface     : Julia General registry")
    else:
        print(f"KomaInterface     : {repository_url}")
        if branch:
            print(f"Git revision      : {branch}")

    print("=" * 72)


def _verify_installation(
    project_dir: Path,
    env: Optional[dict[str, str]],
    cpu: bool,
) -> int:
    """Verify that the installed Julia packages can be imported.

    For GPU-capable installations, CUDA.jl must be importable. A functional
    NVIDIA GPU is not required during installation; CUDA.functional() is
    reported for information only.
    """
    checks = [
        "using KomaInterface",
        'println("KomaInterface import: OK")',
    ]

    if not cpu:
        checks.extend(
            [
                "using CUDA",
                'println("CUDA.jl import: OK")',
                'println("CUDA functional: ", CUDA.functional())',
            ]
        )

    expression = "; ".join(checks)

    command = [
        "julia",
        f"--project={project_dir}",
        "--startup-file=no",
        "-e",
        expression,
    ]

    print("\nVerifying Julia installation...")

    completed = subprocess.run(
        command,
        check=False,
        env=env,
    )

    if completed.returncode != 0:
        print(
            "\nERROR: Julia packages were installed, but verification failed.",
            file=sys.stderr,
        )

    return completed.returncode


def install_julia_deps(
    repository_url: Optional[str] = DEFAULT_REPOSITORY_URL,
    branch: Optional[str] = DEFAULT_BRANCH,
    install_dir: Optional[str] = None,
    project_dir: str | Path = DEFAULT_PROJECT_DIR,
    cpu: bool = False,
    update: bool = False,
    verify: bool = True,
) -> int:
    """Install the Julia dependencies required by CAMRIE.

    Parameters
    ----------
    repository_url:
        Git repository containing KomaInterface.jl. Set to ``None`` to install
        the registered version of KomaInterface.
    branch:
        Git branch, tag, or commit. Ignored for a registered installation.
    install_dir:
        Optional Julia depot directory. This controls where package source,
        artifacts, registries, and precompiled caches are stored.
    project_dir:
        Dedicated Julia project directory used by CAMRIE.
    cpu:
        If ``True``, do not explicitly install CUDA.jl.
    update:
        If ``True``, run ``Pkg.update()`` in the CAMRIE Julia project before
        precompilation.
    verify:
        If ``True``, import the installed packages after installation.
    """
    julia_executable = shutil.which("julia")

    if julia_executable is None:
        print(
            "ERROR: Julia was not found on PATH.\n"
            "Install Julia and confirm that `julia --version` works, "
            "then run this command again.",
            file=sys.stderr,
        )
        return 127

    normalized_url = _normalize_repository_url(repository_url)
    project_path = Path(project_dir).expanduser().resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    _print_installation_summary(
        project_dir=project_path,
        repository_url=normalized_url,
        branch=branch,
        cpu=cpu,
        update=update,
    )

    env = _build_environment(
        install_dir=install_dir,
        repository_url=normalized_url,
    )

    command = build_julia_command(
        project_dir=project_path,
        repository_url=normalized_url,
        branch=branch,
        cpu=cpu,
        update=update,
    )

    print("\nInstalling Julia dependencies...")

    try:
        completed = subprocess.run(
            command,
            check=False,
            env=env,
        )
    except OSError as exc:
        print(
            f"ERROR: Could not start Julia: {exc}",
            file=sys.stderr,
        )
        return 1

    if completed.returncode != 0:
        print(
            "\nERROR: Julia dependency installation failed.\n"
            f"Julia returned exit code {completed.returncode}.\n\n"
            "Possible corrective actions:\n"
            "  1. Use --cpu on systems without NVIDIA GPU support.\n"
            "  2. Remove the CAMRIE project directory and retry.\n"
            "  3. Confirm that the requested Git branch or revision exists.\n"
            "  4. Confirm that SSH credentials work when using an SSH URL.",
            file=sys.stderr,
        )
        return completed.returncode

    if verify:
        verification_code = _verify_installation(
            project_dir=project_path,
            env=env,
            cpu=cpu,
        )

        if verification_code != 0:
            return verification_code

    print("\nCAMRIE Julia installation completed successfully.")
    print(f"Project directory: {project_path}")

    if cpu:
        print("Runtime backend: CPU")
    else:
        print(
            "Runtime backend: CUDA-capable. Whether a GPU is functional "
            "depends on the host and NVIDIA runtime."
        )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Install KomaInterface and the Julia dependencies used by "
            "CAMRIE simulations in an isolated Julia project."
        )
    )

    source_group = parser.add_mutually_exclusive_group()

    source_group.add_argument(
        "--repository-url",
        default=DEFAULT_REPOSITORY_URL,
        help=(
            "Install KomaInterface.jl from this Git repository. "
            f"Default: {DEFAULT_REPOSITORY_URL}"
        ),
    )

    source_group.add_argument(
        "--registered",
        action="store_true",
        help=(
            "Install the registered KomaInterface package instead of "
            "installing it from Git."
        ),
    )

    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=(
            "Git branch, tag, or commit used with --repository-url. "
            f"Default: {DEFAULT_BRANCH}"
        ),
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help=(
            "Install for CPU execution without explicitly installing CUDA.jl. "
            "Use this on CPU-only machines or Colab runtimes without a GPU."
        ),
    )

    parser.add_argument(
        "--project-dir",
        default=str(DEFAULT_PROJECT_DIR),
        help=(
            "Dedicated Julia project directory for CAMRIE. "
            f"Default: {DEFAULT_PROJECT_DIR}"
        ),
    )

    parser.add_argument(
        "--install-dir",
        help=(
            "Optional Julia depot directory. This controls where Julia stores "
            "registries, downloaded packages, artifacts, and compiled caches."
        ),
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Run Pkg.update() in the CAMRIE Julia project before precompiling. "
            "Use this deliberately when refreshing to the latest Julia-side "
            "dependencies."
        ),
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip package import verification after installation.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    repository_url: Optional[str]

    if args.registered:
        repository_url = None
        branch = None
    else:
        repository_url = args.repository_url
        branch = args.branch

    return install_julia_deps(
        repository_url=repository_url,
        branch=branch,
        install_dir=args.install_dir,
        project_dir=args.project_dir,
        cpu=args.cpu,
        update=args.update,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    raise SystemExit(main())
