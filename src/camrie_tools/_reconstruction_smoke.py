"""End-to-end reconstruction smoke test using a simple package phantom."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_JULIA_PROJECT = Path.home() / ".camrie" / "julia"


def save_reconstruction_preview(
    reconstruction: np.ndarray,
    output_path: str | Path,
    title: str = "CAMRIE phantom reconstruction",
) -> str:
    """Save a PNG preview of a 2D magnitude reconstruction."""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    preview_path = Path(output_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    finite = reconstruction[np.isfinite(reconstruction)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else None

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        reconstruction,
        cmap="gray",
        origin="lower",
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Readout")
    ax.set_ylabel("Phase encoding")
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Magnitude",
    )
    fig.tight_layout()
    fig.savefig(preview_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(preview_path)


def create_circular_phantom(
    radius_mm: float = 45.0,
    fov_mm: float = 300.0,
    grid_size: int = 11,
    t1_ms: float = 900.0,
    t2_ms: float = 80.0,
) -> dict[str, Any]:
    """Create a small 2D circular spin phantom in Koma-compatible units."""
    axis_mm = np.linspace(-0.45 * fov_mm, 0.45 * fov_mm, int(grid_size), dtype=np.float32)
    xx, yy = np.meshgrid(axis_mm, axis_mm, indexing="xy")
    mask = (xx**2 + yy**2) <= float(radius_mm) ** 2

    x_mm = xx[mask].astype(np.float32)
    y_mm = yy[mask].astype(np.float32)
    z_mm = np.zeros_like(x_mm, dtype=np.float32)
    rho = np.ones_like(x_mm, dtype=np.float32)
    t1 = np.full_like(x_mm, float(t1_ms) * 1e-3, dtype=np.float32)
    t2 = np.full_like(x_mm, float(t2_ms) * 1e-3, dtype=np.float32)

    return {
        "name": "circular_smoke_phantom",
        "x": (x_mm * 1e-3).tolist(),
        "y": (y_mm * 1e-3).tolist(),
        "z": (z_mm * 1e-3).tolist(),
        "rho": rho.tolist(),
        "t1": t1.tolist(),
        "t2": t2.tolist(),
        "t2s": t2.tolist(),
        "dw": np.zeros_like(rho, dtype=np.float32).tolist(),
        "slice_geometry": {
            "center_mm": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "radius_mm": float(radius_mm),
            "fov_mm": float(fov_mm),
            "grid_size": int(grid_size),
            "t1_ms": float(t1_ms),
            "t2_ms": float(t2_ms),
        },
    }


def create_concentric_cylinder_phantom(
    inner_radius_mm: float = 30.0,
    outer_radius_mm: float = 60.0,
    fov_mm: float = 300.0,
    grid_size: int = 151,
    inner_pd: float = 1.0,
    inner_t1_ms: float = 800.0,
    inner_t2_ms: float = 60.0,
    outer_pd: float = 0.8,
    outer_t1_ms: float = 1200.0,
    outer_t2_ms: float = 80.0,
) -> dict[str, Any]:
    """Create the two-compartment phantom used by the local CAMRIE smoke path."""
    axis_mm = np.linspace(-0.45 * fov_mm, 0.45 * fov_mm, int(grid_size), dtype=np.float32)
    xx, yy = np.meshgrid(axis_mm, axis_mm, indexing="xy")
    radius2 = xx**2 + yy**2

    inner_mask = radius2 <= float(inner_radius_mm) ** 2
    outer_mask = (radius2 <= float(outer_radius_mm) ** 2) & ~inner_mask
    mask = inner_mask | outer_mask

    x_mm = xx[mask].astype(np.float32)
    y_mm = yy[mask].astype(np.float32)
    z_mm = np.zeros_like(x_mm, dtype=np.float32)

    rho = np.empty_like(x_mm, dtype=np.float32)
    t1 = np.empty_like(x_mm, dtype=np.float32)
    t2 = np.empty_like(x_mm, dtype=np.float32)

    selected_inner = inner_mask[mask]
    rho[selected_inner] = float(inner_pd)
    t1[selected_inner] = float(inner_t1_ms) * 1e-3
    t2[selected_inner] = float(inner_t2_ms) * 1e-3
    rho[~selected_inner] = float(outer_pd)
    t1[~selected_inner] = float(outer_t1_ms) * 1e-3
    t2[~selected_inner] = float(outer_t2_ms) * 1e-3

    return {
        "name": "concentric_cylinder_smoke_phantom",
        "x": (x_mm * 1e-3).tolist(),
        "y": (y_mm * 1e-3).tolist(),
        "z": (z_mm * 1e-3).tolist(),
        "rho": rho.tolist(),
        "t1": t1.tolist(),
        "t2": t2.tolist(),
        "t2s": t2.tolist(),
        "dw": np.zeros_like(rho, dtype=np.float32).tolist(),
        "slice_geometry": {
            "center_mm": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "inner_radius_mm": float(inner_radius_mm),
            "outer_radius_mm": float(outer_radius_mm),
            "fov_mm": float(fov_mm),
            "grid_size": int(grid_size),
            "inner": {
                "pd": float(inner_pd),
                "t1_ms": float(inner_t1_ms),
                "t2_ms": float(inner_t2_ms),
            },
            "outer": {
                "pd": float(outer_pd),
                "t1_ms": float(outer_t1_ms),
                "t2_ms": float(outer_t2_ms),
            },
        },
    }


def run_reconstruction_smoke(
    output_dir: str,
    sequence_file: Optional[str] = None,
    use_gpu: bool = False,
    b0: float = 1.5,
    n_threads: int = 2,
    grid_size: int = 151,
    phantom_kind: str = "concentric",
    radius_mm: float = 45.0,
    inner_radius_mm: float = 30.0,
    outer_radius_mm: float = 60.0,
    inner_pd: float = 1.0,
    inner_t1_ms: float = 800.0,
    inner_t2_ms: float = 60.0,
    outer_pd: float = 0.8,
    outer_t1_ms: float = 1200.0,
    outer_t2_ms: float = 80.0,
    julia_project: Optional[str] = None,
) -> dict[str, Any]:
    import camrie_tools
    from camrie_tools.MRI_pipeline import (
        read_pulseq_params,
        reconstruct_from_kspace,
        run_simulation_julia_batch,
        save_phantom_json,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    project_path = Path(
        julia_project
        or os.environ.get("CAMRIE_JULIA_PROJECT")
        or os.environ.get("JULIA_PROJECT")
        or DEFAULT_JULIA_PROJECT
    ).expanduser()
    if not project_path.exists():
        raise FileNotFoundError(
            "The CAMRIE Julia project was not found.\n"
            f"Expected project directory: {project_path}\n"
            "Run `camrie-install-julia --cpu` on CPU-only systems, or "
            "`camrie-install-julia` on CUDA-capable systems."
        )
    os.environ["CAMRIE_JULIA_PROJECT"] = str(project_path)
    os.environ["JULIA_PROJECT"] = str(project_path)

    seq_path = Path(sequence_file or camrie_tools.sequence_path())
    params = read_pulseq_params(str(seq_path))
    fov = params.get("fov_mm", [300.0, 300.0])
    fov_mm = float(max(fov[0], fov[1]))
    if phantom_kind == "circle":
        phantom = create_circular_phantom(
            radius_mm=radius_mm,
            fov_mm=fov_mm,
            grid_size=grid_size,
        )
        phantom_path = out / "circular_phantom.json"
        phantom_title = "CAMRIE circular phantom reconstruction"
    elif phantom_kind == "concentric":
        phantom = create_concentric_cylinder_phantom(
            inner_radius_mm=inner_radius_mm,
            outer_radius_mm=outer_radius_mm,
            fov_mm=fov_mm,
            grid_size=grid_size,
            inner_pd=inner_pd,
            inner_t1_ms=inner_t1_ms,
            inner_t2_ms=inner_t2_ms,
            outer_pd=outer_pd,
            outer_t1_ms=outer_t1_ms,
            outer_t2_ms=outer_t2_ms,
        )
        phantom_path = out / "concentric_cylinder_phantom.json"
        phantom_title = "CAMRIE concentric-cylinder phantom reconstruction"
    else:
        raise ValueError(f"Unknown phantom kind: {phantom_kind}")
    save_phantom_json(phantom, str(phantom_path))

    sim_dir = out / "sim_000"
    batch_results = run_simulation_julia_batch(
        sequence_file=str(seq_path),
        phantom_paths=[str(phantom_path)],
        rotation_paths=[None],
        sim_output_dirs=[str(sim_dir)],
        b0=b0,
        use_gpu=use_gpu,
        n_threads=n_threads,
        simT2s=False,
    )
    kspace, info = batch_results[0]
    reconstruction = reconstruct_from_kspace(
        kspace,
        expected_shape=(int(params["nP"]), int(params["nF"])),
        oversampling=int(params.get("oversampling", 1)),
        orientation=params.get("orientation"),
        etl=params.get("etl", 1),
        echo_spacing_ms=params.get("echo_spacing_ms"),
        te_eff_ms=params.get("te_eff_ms"),
    )
    magnitude = np.abs(reconstruction).astype(np.float32)

    kspace_path = out / "kspace.npy"
    recon_path = out / "reconstruction_magnitude.npy"
    preview_path = out / "reconstruction_preview.png"
    np.save(kspace_path, kspace)
    np.save(recon_path, magnitude)

    peak = float(np.max(magnitude))
    total_signal = float(np.sum(magnitude))
    if not np.isfinite(magnitude).all():
        raise RuntimeError("reconstruction contains non-finite values")
    if peak <= 0 or total_signal <= 0:
        raise RuntimeError("reconstruction has no positive signal")

    save_reconstruction_preview(
        magnitude,
        preview_path,
        title=(
            f"{phantom_title}\n"
            f"{len(phantom['rho'])} spins"
        ),
    )

    summary = {
        "sequence": str(seq_path),
        "phantom": str(phantom_path),
        "phantom_kind": phantom_kind,
        "phantom_geometry": phantom["slice_geometry"],
        "spin_count": len(phantom["rho"]),
        "kspace_shape": list(kspace.shape),
        "reconstruction_shape": list(magnitude.shape),
        "peak": peak,
        "total_signal": total_signal,
        "use_gpu": bool(use_gpu),
        "julia_project": str(project_path),
        "info": info,
        "outputs": {
            "kspace": str(kspace_path),
            "reconstruction_magnitude": str(recon_path),
            "reconstruction_preview": str(preview_path),
        },
    }
    summary_path = out / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a small Koma reconstruction smoke test."
    )
    parser.add_argument("--sequence", help="Pulseq .seq file to simulate.")
    parser.add_argument("--output", help="Output directory. Defaults to a temporary directory.")
    parser.add_argument("--use-gpu", action="store_true", help="Use Koma/CUDA GPU simulation.")
    parser.add_argument("--b0", type=float, default=1.5, help="Main field strength in tesla.")
    parser.add_argument("--threads", type=int, default=2, help="Julia thread count.")
    parser.add_argument("--grid-size", type=int, default=151, help="Phantom grid size.")
    parser.add_argument(
        "--phantom",
        choices=["concentric", "circle"],
        default="concentric",
        help="Phantom to simulate. Default: concentric.",
    )
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=45.0,
        help="Circle phantom radius, used only with --phantom circle.",
    )
    parser.add_argument(
        "--inner-radius-mm",
        type=float,
        default=30.0,
        help="Inner radius for the concentric phantom.",
    )
    parser.add_argument(
        "--outer-radius-mm",
        type=float,
        default=60.0,
        help="Outer radius for the concentric phantom.",
    )
    parser.add_argument("--inner-pd", type=float, default=1.0, help="Inner-core proton density.")
    parser.add_argument("--inner-t1-ms", type=float, default=800.0, help="Inner-core T1 in ms.")
    parser.add_argument("--inner-t2-ms", type=float, default=60.0, help="Inner-core T2 in ms.")
    parser.add_argument("--outer-pd", type=float, default=0.8, help="Outer-ring proton density.")
    parser.add_argument("--outer-t1-ms", type=float, default=1200.0, help="Outer-ring T1 in ms.")
    parser.add_argument("--outer-t2-ms", type=float, default=80.0, help="Outer-ring T2 in ms.")
    parser.add_argument(
        "--julia-project",
        default=None,
        help=f"CAMRIE Julia project directory. Default: {DEFAULT_JULIA_PROJECT}",
    )
    args = parser.parse_args(argv)

    output_dir = args.output or tempfile.mkdtemp(prefix="camrie_recon_smoke_")
    summary = run_reconstruction_smoke(
        output_dir=output_dir,
        sequence_file=args.sequence,
        use_gpu=args.use_gpu,
        b0=args.b0,
        n_threads=args.threads,
        grid_size=args.grid_size,
        phantom_kind=args.phantom,
        radius_mm=args.radius_mm,
        inner_radius_mm=args.inner_radius_mm,
        outer_radius_mm=args.outer_radius_mm,
        inner_pd=args.inner_pd,
        inner_t1_ms=args.inner_t1_ms,
        inner_t2_ms=args.inner_t2_ms,
        outer_pd=args.outer_pd,
        outer_t1_ms=args.outer_t1_ms,
        outer_t2_ms=args.outer_t2_ms,
        julia_project=args.julia_project,
    )
    print("CAMRIE reconstruction smoke test passed.")
    print(f"Output directory: {output_dir}")
    print(f"Spin count: {summary['spin_count']}")
    print(f"Phantom: {summary['phantom_kind']}")
    print(f"k-space shape: {summary['kspace_shape']}")
    print(f"reconstruction shape: {summary['reconstruction_shape']}")
    print(f"peak signal: {summary['peak']:.6g}")
    print(f"Julia project: {summary['julia_project']}")
    print(f"Preview PNG: {summary['outputs']['reconstruction_preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
