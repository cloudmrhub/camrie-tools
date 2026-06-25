"""Python tools for CAMRIE."""

from __future__ import annotations

from importlib import resources

__version__ = "0.1.0"


def simulate_batch_path() -> str:
    """Return the installed path to the bundled Julia simulation entrypoint."""
    return str(resources.files(__package__).joinpath("simulate_batch.jl"))


def sequence_path(name: str = "T1-Weighted_Spin_Echo.seq") -> str:
    """Return the installed path to a bundled pulse sequence file."""
    return str(resources.files(__package__).joinpath("sequences", name))


__all__ = ["__version__", "simulate_batch_path", "sequence_path"]
