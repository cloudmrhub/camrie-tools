"""End-to-end reconstruction smoke test using a circular phantom."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np


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
    t1 = np.full_like(x_mm, float(t1_ms), dtype=np.float32)
    t2 = np.full_like(x_mm, float(t2_ms), dtype=np.float32)

    return {
        "name": "circular_smoke_phantom",
        "x": (x_mm * 1e-3).tolist(),
        "y": (y_mm * 1e-3).tolist(),
        "z": (z_mm * 1e-3).tolist(),
        "rho": rho.tolist(),
        "t1": t1.tolist(),
        "t2": t2.tolist(),
        "t2s": (t2 * 0.5).tolist(),
        "dw": np.zeros_like(rho, dtype=np.float32).tolist(),
        "slice_geometry": {
            "center_mm": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "radius_mm": float(radius_mm),
            "fov_mm": float(fov_mm),
            "grid_size": int(grid_size),
        },
    }


def run_reconstruction_smoke(
    output_dir: str,
    sequence_file: Optional[str] = None,
    use_gpu: bool = False,
    b0: float = 1.5,
    n_threads: int = 2,
    grid_size: int = 21,
    radius_mm: float = 45.0,
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

    seq_path = Path(sequence_file or camrie_tools.sequence_path())
    params = read_pulseq_params(str(seq_path))
    fov = params.get("fov_mm", [300.0, 300.0])
    phantom = create_circular_phantom(
        radius_mm=radius_mm,
        fov_mm=float(max(fov[0], fov[1])),
        grid_size=grid_size,
    )
    phantom_path = out / "circular_phantom.json"
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
    np.save(kspace_path, kspace)
    np.save(recon_path, magnitude)

    peak = float(np.max(magnitude))
    total_signal = float(np.sum(magnitude))
    if not np.isfinite(magnitude).all():
        raise RuntimeError("reconstruction contains non-finite values")
    if peak <= 0 or total_signal <= 0:
        raise RuntimeError("reconstruction has no positive signal")

    summary = {
        "sequence": str(seq_path),
        "phantom": str(phantom_path),
        "spin_count": len(phantom["rho"]),
        "kspace_shape": list(kspace.shape),
        "reconstruction_shape": list(magnitude.shape),
        "peak": peak,
        "total_signal": total_signal,
        "use_gpu": bool(use_gpu),
        "info": info,
        "outputs": {
            "kspace": str(kspace_path),
            "reconstruction_magnitude": str(recon_path),
        },
    }
    summary_path = out / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a small circular-phantom Koma reconstruction smoke test."
    )
    parser.add_argument("--sequence", help="Pulseq .seq file to simulate.")
    parser.add_argument("--output", help="Output directory. Defaults to a temporary directory.")
    parser.add_argument("--use-gpu", action="store_true", help="Use Koma/CUDA GPU simulation.")
    parser.add_argument("--b0", type=float, default=1.5, help="Main field strength in tesla.")
    parser.add_argument("--threads", type=int, default=2, help="Julia thread count.")
    parser.add_argument("--grid-size", type=int, default=11, help="Circular phantom grid size.")
    parser.add_argument("--radius-mm", type=float, default=45.0, help="Circular phantom radius.")
    args = parser.parse_args(argv)

    output_dir = args.output or tempfile.mkdtemp(prefix="camrie_recon_smoke_")
    summary = run_reconstruction_smoke(
        output_dir=output_dir,
        sequence_file=args.sequence,
        use_gpu=args.use_gpu,
        b0=args.b0,
        n_threads=args.threads,
        grid_size=args.grid_size,
        radius_mm=args.radius_mm,
    )
    print("CAMRIE reconstruction smoke test passed.")
    print(f"Output directory: {output_dir}")
    print(f"Spin count: {summary['spin_count']}")
    print(f"k-space shape: {summary['kspace_shape']}")
    print(f"reconstruction shape: {summary['reconstruction_shape']}")
    print(f"peak signal: {summary['peak']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
