#!/usr/bin/env python3
"""
MRI_pipeline_3d.py — 3D MRI Simulation Pipeline (KomaMRI)

Designed for 3D sequences created by gre3d.py and tse3d.py.

Key differences from the 2D pipeline:
  - One volumetric phantom covers the entire slab (no per-slice loop)
  - Single KomaMRI simulation call for the whole volume
  - Reconstruction: 3D k-space (Nz × Ny × Nf) → 3D IFFT → volume image
  - Output: NIfTI (.nii.gz) volumetric image

Usage:
    python MRI_pipeline_3d.py \\
        --rho  path/to/rho.nii.gz \\
        --t1   path/to/T1.nii.gz  \\
        --t2   path/to/T2.nii.gz  \\
        --seq  sequences/IXI3D_T1_GRE.seq \\
        --out  output/IXI3D_T1_GRE/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional HDF5
HDF5_AVAILABLE = False
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    pass

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")

try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None  # graceful fallback — nearest-neighbour resampling used


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Sequence parameter reader (3D)
# ═══════════════════════════════════════════════════════════════════════════════

def read_pulseq_params_3d(seq_path: str) -> Dict[str, Any]:
    """
    Read 3D sequence parameters from a Pulseq .seq file [DEFINITIONS] block.

    Returns a dict with at minimum:
      nx, ny, nz, tr_ms, te_ms, fov_mm, fov_slab_mm, acquisition, etl
    """
    params: Dict[str, Any] = {}
    in_def = False

    with open(seq_path) as f:
        content = f.read()

    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "[DEFINITIONS]":
            in_def = True
            continue
        if in_def and stripped.startswith("["):
            in_def = False
            continue
        if not in_def:
            continue
        if " " not in stripped:
            continue
        key, *val_parts = stripped.split()
        val_str = " ".join(val_parts)
        try:
            val = float(val_str)
        except ValueError:
            val = val_str.strip()
        params[key] = val

    def _get(key: str, default=None):
        return params.get(key, default)

    nx = int(_get("Nx", 256))
    ny = int(_get("Ny", 256))
    nz = int(_get("Nz", 1))
    tr_ms = float(_get("TR", 0.01)) * 1e3  # stored in s → convert ms
    te_ms = float(_get("TE", 0.005)) * 1e3
    te_eff_ms = float(_get("TE_eff", te_ms / 1e3)) * 1e3

    fov_raw = _get("FOV", None)
    if fov_raw is not None and not isinstance(fov_raw, str):
        # Scalar (first dimension)
        fov_mm = [float(fov_raw) * 1e3] * 2
    elif isinstance(fov_raw, str):
        parts = [float(x) for x in fov_raw.split()]
        fov_mm = [p * 1e3 for p in parts[:2]]
    else:
        fov_mm = [240.0, 240.0]

    slab_mm = float(_get("SlabThickness", fov_mm[0] / 1e3)) * 1e3
    partition_mm = float(_get("PartitionThickness", slab_mm / nz))
    acq = str(_get("Acquisition", "3D_GRE"))
    etl = int(_get("ETL", 1))
    flip = float(_get("FlipAngleDeg", 90.0))

    # Count ADC events to verify
    n_adc = content.count("\n[ADC]") + content.count(" [ADC]")  # rough
    in_adc = False
    adc_count = 0
    for line in content.splitlines():
        if line.strip() == "[ADC]":
            in_adc = True
            continue
        if in_adc and line.strip().startswith("["):
            in_adc = False
            continue
        if in_adc and line.strip() and not line.strip().startswith("#"):
            adc_count += 1

    return {
        "nx": nx, "ny": ny, "nz": nz,
        "tr_ms": tr_ms, "te_ms": te_ms, "te_eff_ms": te_eff_ms,
        "fov_mm": fov_mm,
        "slab_mm": slab_mm,
        "partition_mm": partition_mm,
        "acquisition": acq,
        "etl": etl,
        "flip_deg": flip,
        "adc_events": adc_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Volumetric phantom extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI, return (data_float32, voxel_size_mm as 1D float array)."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    vox = np.abs(np.array(img.header.get_zooms()[:3], dtype=float))
    return data, vox


def _resample_volume(vol: np.ndarray, src_vox_mm: np.ndarray, tgt_vox_mm: float) -> np.ndarray:
    """Resample volume to isotropic target voxel size using scipy zoom."""
    # Convert to plain Python floats so scipy/numpy round() works correctly
    zoom_factors = [float(v) / float(tgt_vox_mm) for v in np.asarray(src_vox_mm).ravel()]
    if zoom is not None:
        return zoom(vol, zoom_factors, order=1).astype(np.float32)
    # Nearest-neighbour fallback
    out_shape = tuple(int(round(s * z)) for s, z in zip(vol.shape, zoom_factors))
    out = np.zeros(out_shape, dtype=np.float32)
    for i in range(out_shape[0]):
        si = min(int(i / zoom_factors[0]), vol.shape[0] - 1)
        for j in range(out_shape[1]):
            sj = min(int(j / zoom_factors[1]), vol.shape[1] - 1)
            for k in range(out_shape[2]):
                sk = min(int(k / zoom_factors[2]), vol.shape[2] - 1)
                out[i, j, k] = vol[si, sj, sk]
    return out


def extract_phantom_3d(
    rho_path: str,
    t1_path: str,
    t2_path: str,
    fov_mm: Tuple[float, float],   # (read_mm, phase_mm)
    slab_mm: float,                # thickness of acquired slab (mm)
    nx: int,
    ny: int,
    nz: int,
    isocenter_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rho_scale: float = 100.0,
    t1_scale: float = 1e-3,        # NIfTI units → seconds
    t2_scale: float = 1e-3,
    t2_floor_s: float = 1e-4,     # minimum T2 to avoid zero-division
) -> Dict[str, np.ndarray]:
    """
    Extract a flat-list 3D phantom (x, y, z, rho, t1, t2) from NIfTI files.

    Coordinates are in metres, centred at isocenter_mm.
    The extracted region is:
      X: ±fov_mm[0]/2  (readout direction)
      Y: ±fov_mm[1]/2  (phase direction)
      Z: ±slab_mm/2    (partition / slice direction)

    Parameters
    ----------
    rho_path, t1_path, t2_path : NIfTI file paths.
    fov_mm                     : (read FOV, phase FOV) in mm.
    slab_mm                    : Acquired slab thickness in mm.
    nx, ny, nz                 : Target 3D matrix size.
    isocenter_mm               : Centre of acquisition in NIfTI voxel space (mm).
    rho_scale                  : Scale factor for proton density.
    t1_scale, t2_scale         : Scale factor if NIfTI stores T1/T2 in ms (default).
    t2_floor_s                 : Minimum T2 value (s) to replace zeros.

    Returns
    -------
    dict with keys: x, y, z, rho, t1, t2 — each a float32 1D array.
    """
    rho_data, rho_vox = _load_nifti(rho_path)
    t1_data, t1_vox = _load_nifti(t1_path)
    t2_data, t2_vox = _load_nifti(t2_path)

    # Target resolution: match sequence resolution
    res_read_mm = fov_mm[0] / nx
    res_phase_mm = fov_mm[1] / ny
    res_part_mm = slab_mm / nz
    tgt_vox_mm = min(res_read_mm, res_phase_mm, res_part_mm)

    # Resample all volumes to target resolution (if needed)
    def _resample_if_needed(vol, src_vox):
        avg_src = float(np.mean(src_vox))
        if abs(avg_src - tgt_vox_mm) > 0.05:
            return _resample_volume(vol, src_vox, tgt_vox_mm)
        return vol

    rho_r = _resample_if_needed(rho_data, rho_vox)
    t1_r = _resample_if_needed(t1_data, t1_vox)
    t2_r = _resample_if_needed(t2_data, t2_vox)

    # Pad to common shape
    max_shape = tuple(max(a, b, c) for a, b, c in zip(rho_r.shape, t1_r.shape, t2_r.shape))

    def _pad(vol):
        pad = [(0, m - s) for s, m in zip(vol.shape, max_shape)]
        return np.pad(vol, pad, mode="constant")

    rho_r = _pad(rho_r)
    t1_r = _pad(t1_r)
    t2_r = _pad(t2_r)

    S = rho_r.shape   # (Sx, Sy, Sz) in voxel space
    cx, cy, cz = (
        isocenter_mm[0] / tgt_vox_mm + S[0] / 2,
        isocenter_mm[1] / tgt_vox_mm + S[1] / 2,
        isocenter_mm[2] / tgt_vox_mm + S[2] / 2,
    )

    # Desired voxel index ranges centred on isocenter
    half_nx = nx / 2
    half_ny = ny / 2
    half_nz = nz / 2

    ix_lo = max(0, int(math.floor(cx - half_nx)))
    ix_hi = min(S[0], ix_lo + nx)
    iy_lo = max(0, int(math.floor(cy - half_ny)))
    iy_hi = min(S[1], iy_lo + ny)
    iz_lo = max(0, int(math.floor(cz - half_nz)))
    iz_hi = min(S[2], iz_lo + nz)

    rho_crop = rho_r[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
    t1_crop = t1_r[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
    t2_crop = t2_r[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]

    # Actual extracted shape
    sx, sy, sz = rho_crop.shape

    # Physical coordinates (m), centred at isocenter
    x_coords = (np.arange(sx) - sx / 2) * tgt_vox_mm * 1e-3   # m
    y_coords = (np.arange(sy) - sy / 2) * tgt_vox_mm * 1e-3
    z_coords = (np.arange(sz) - sz / 2) * tgt_vox_mm * 1e-3

    xg, yg, zg = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    x_flat = xg.ravel().astype(np.float32)
    y_flat = yg.ravel().astype(np.float32)
    z_flat = zg.ravel().astype(np.float32)

    rho_flat = (rho_crop / rho_crop.max() * rho_scale if rho_crop.max() > 0
                else rho_crop).ravel().astype(np.float32)
    t1_flat = (t1_crop * t1_scale).ravel().astype(np.float32)
    t2_flat = np.maximum(t2_crop * t2_scale, t2_floor_s).ravel().astype(np.float32)
    t2s_flat = np.full_like(t2_flat, 0.05)   # assume T2* = 50 ms (3T default)

    # Mask out background (rho ≈ 0)
    mask = rho_flat > (0.01 * rho_scale)
    return {
        "x": x_flat[mask],
        "y": y_flat[mask],
        "z": z_flat[mask],
        "rho": rho_flat[mask],
        "t1": t1_flat[mask],
        "t2": t2_flat[mask],
        "t2s": t2s_flat[mask],
        "dw": np.zeros(mask.sum(), dtype=np.float32),
        "n_spins": int(mask.sum()),
        "voxel_mm": tgt_vox_mm,
        "crop_shape": (sx, sy, sz),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Phantom I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_phantom_hdf5_3d(phantom: Dict, path: str) -> str:
    if not HDF5_AVAILABLE:
        raise RuntimeError("h5py not installed.  pip install h5py")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        for k in ("x", "y", "z", "rho", "t1", "t2", "t2s", "dw"):
            f.create_dataset(k, data=phantom[k])
        f.attrs["name"] = "phantom_3d"
        f.attrs["source"] = "MRI_pipeline_3d"
    return path


def save_phantom_json_3d(phantom: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = {k: phantom[k].tolist() for k in ("x", "y", "z", "rho", "t1", "t2", "t2s", "dw")}
    d["name"] = "phantom_3d"
    with open(path, "w") as f:
        json.dump(d, f)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Julia simulation
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation_julia_3d(
    sequence_file: str,
    phantom_path: str,
    output_dir: str,
    b0: float = 3.0,
    use_gpu: bool = False,
    n_threads: int = 4,
    julia_script: Optional[str] = None,
) -> str:
    """
    Call the KomaMRI Julia batch simulator for a single 3D phantom.

    The function writes a one-entry batch JSON, calls the Julia script,
    and returns the output directory path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = [{"phantom": str(phantom_path), "output": str(out_dir)}]
    batch_json = str(out_dir / "batch.json")
    with open(batch_json, "w") as f:
        json.dump(batch, f)

    if julia_script is None:
        # Look relative to this file, then relative to cwd
        candidates = [
            Path(__file__).parent.parent / "dev" / "simulate_batch_final.jl",
            Path(__file__).parent / "simulate_batch_final.jl",
            Path("dev/simulate_batch_final.jl"),
            Path("simulate_batch_final.jl"),
        ]
        for c in candidates:
            if c.exists():
                julia_script = str(c)
                break
        if julia_script is None:
            raise FileNotFoundError(
                "Could not find simulate_batch_final.jl.  "
                "Pass --julia-script explicitly."
            )

    cmd = [
        "julia", f"--threads={n_threads}",
        julia_script,
        str(b0),
        str(sequence_file),
        batch_json,
        "true" if use_gpu else "false",
        str(n_threads),
    ]

    print(f"\n  Julia cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Julia simulation failed (exit code {result.returncode}).")

    return str(out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 3D Reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_3d(
    kspace_path: str,
    nx: int,
    ny: int,
    nz: int,
    acquisition: str = "3D_GRE",
    etl: int = 1,
    apply_hamming: bool = True,
    remove_os: bool = False,
    n_adc: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct a 3D volume from raw k-space (numpy .npz format).

    Parameters
    ----------
    kspace_path : Path to k.npz (keys: first array assumed to be k-space).
    nx, ny, nz  : Reconstructed matrix dimensions.
    acquisition : '3D_GRE' or '3D_TSE'.
    etl         : Echo train length (for TSE reordering).
    apply_hamming : Apply 3D Hamming window before IFFT.
    remove_os   : Remove readout oversampling (keep central nx of n_adc).
    n_adc       : Number of ADC samples per readout (if different from nx).

    Returns
    -------
    vol : (nz, ny, nx) float32 magnitude volume.
    """
    data = np.load(kspace_path, allow_pickle=False)
    # npzwrite(file, array) in Julia's NPZ.jl writes a bare .npy regardless of
    # the .npz extension → np.load returns an ndarray directly.
    # npzwrite(file, Dict(...)) writes a true .npz → np.load returns NpzFile.
    if isinstance(data, np.ndarray):
        K = data
    else:
        key = list(data.keys())[0]
        K = data[key]   # shape: (n_profiles, n_samples)

    if n_adc is None:
        n_adc = K.shape[1]

    n_profiles = K.shape[0]
    expected = ny * nz
    print(f"  K-space: {n_profiles} profiles × {n_adc} samples  (expected {expected})")

    # ── Oversampling removal ────────────────────────────────────────────
    if remove_os and n_adc > nx:
        start = (n_adc - nx) // 2
        K = K[:, start: start + nx]
        n_adc = nx

    # ── Reshape to 3D k-space: (Nz, Ny, Nf) ────────────────────────────
    if "GRE" in acquisition.upper():
        # Loop order: outer kz, inner ky  → profile = kz*Ny + ky
        if n_profiles != ny * nz:
            print(f"  [Warning] Profile count mismatch: got {n_profiles}, expected {ny*nz}. "
                  f"Padding/truncating.")
            K = _pad_or_truncate(K, ny * nz)
        K3d = K.reshape(nz, ny, n_adc)

    elif "TSE" in acquisition.upper():
        # Loop order: outer kz, middle echo_group, inner echo
        # For reconstruction, treat similarly — profile ordering is
        # kz → echo_group → echo, which gives us ky lines in PE order.
        # We sort by ky index for standard 3D reconstruction.
        n_ex = math.ceil(ny / etl)
        expected_tse = nz * n_ex * etl
        if n_profiles != expected_tse:
            print(f"  [Warning] TSE profile count mismatch: got {n_profiles}, "
                  f"expected {expected_tse}.")
            K = _pad_or_truncate(K, expected_tse)
        # Reshape keeping kz outermost
        K_kz = K.reshape(nz, n_ex * etl, n_adc)   # (Nz, Ny_padded, Nf)
        # Trim to ny lines (last few may be dummies)
        K3d = K_kz[:, :ny, :]                       # (Nz, Ny, Nf)
    else:
        raise ValueError(f"Unknown acquisition type '{acquisition}'.")

    # ── Hamming window (3D) ─────────────────────────────────────────────
    if apply_hamming:
        wz = np.hamming(nz).astype(np.float32)
        wy = np.hamming(ny).astype(np.float32)
        wf = np.hamming(n_adc).astype(np.float32)
        W = wz[:, None, None] * wy[None, :, None] * wf[None, None, :]
        K3d = K3d * W

    # ── 3D IFFT ─────────────────────────────────────────────────────────
    vol_complex = np.fft.ifftshift(
        np.fft.ifftn(np.fft.fftshift(K3d), axes=(0, 1, 2)),
        axes=(0, 1, 2)
    )

    vol = np.abs(vol_complex).astype(np.float32)
    print(f"  Reconstructed: {vol.shape}  max={vol.max():.3f}")
    return vol


def _pad_or_truncate(arr: np.ndarray, target_rows: int) -> np.ndarray:
    n = arr.shape[0]
    if n >= target_rows:
        return arr[:target_rows]
    pad = np.zeros((target_rows - n, arr.shape[1]), dtype=arr.dtype)
    return np.vstack([arr, pad])


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Save NIfTI
# ═══════════════════════════════════════════════════════════════════════════════

def save_nifti_3d(
    vol: np.ndarray,
    out_path: str,
    voxel_mm: float = 1.25,
    fov_mm: Tuple[float, float] = (240.0, 240.0),
    slab_mm: float = 187.5,
) -> str:
    """Save a 3D volume as a NIfTI file with correct voxel dimensions."""
    nz, ny, nx = vol.shape
    res_x = fov_mm[0] / nx
    res_y = fov_mm[1] / ny
    res_z = slab_mm / nz

    affine = np.diag([res_x, res_y, res_z, 1.0])
    # Shift origin to centre
    affine[:3, 3] = [-fov_mm[0] / 2, -fov_mm[1] / 2, -slab_mm / 2]

    img = nib.Nifti1Image(vol, affine)
    img.header.set_zooms([res_x, res_y, res_z])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nib.save(img, out_path)
    print(f"  Saved NIfTI: {out_path}  ({vol.shape})")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Main 3D pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline_3d(
    rho_path: str,
    t1_path: str,
    t2_path: str,
    sequence_file: str,
    output_dir: str,
    isocenter_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    b0: float = 3.0,
    use_gpu: bool = False,
    n_threads: int = 4,
    apply_hamming: bool = True,
    remove_os: bool = False,
    use_hdf5: bool = True,
    julia_script: Optional[str] = None,
    t2_floor_s: float = 1e-4,
    rho_scale: float = 100.0,
    t1_scale: float = 1e-3,
    t2_scale: float = 1e-3,
    final_nifti_path: Optional[str] = None,
) -> str:
    """
    Full 3D MRI simulation pipeline.

    1. Read sequence parameters.
    2. Extract volumetric phantom from NIfTI files.
    3. Save phantom (HDF5 preferred, JSON fallback).
    4. Run KomaMRI simulation (Julia).
    5. Reconstruct 3D volume from k-space.
    6. Save as NIfTI.

    Returns
    -------
    Path to the output NIfTI file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  3D MRI Pipeline  [{ts}]")
    print(f"  Sequence: {sequence_file}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}")

    # ── 1. Read sequence params ──────────────────────────────────────────
    print("\n[1/5] Reading sequence parameters...")
    params = read_pulseq_params_3d(sequence_file)
    nx, ny, nz = params["nx"], params["ny"], params["nz"]
    fov_mm = params["fov_mm"]
    slab_mm = params["slab_mm"]
    acquisition = params["acquisition"]
    etl = params["etl"]
    print(f"  Matrix: {nx}×{ny}×{nz}  FOV: {fov_mm[0]:.1f}×{fov_mm[1]:.1f}×{slab_mm:.1f} mm³")
    print(f"  Acquisition: {acquisition}  ETL: {etl}")
    print(f"  TR={params['tr_ms']:.2f} ms  TE={params['te_ms']:.2f} ms")

    # ── 2. Extract phantom ───────────────────────────────────────────────
    print("\n[2/5] Extracting 3D phantom...")
    phantom = extract_phantom_3d(
        rho_path=rho_path, t1_path=t1_path, t2_path=t2_path,
        fov_mm=(fov_mm[0], fov_mm[1]),
        slab_mm=slab_mm,
        nx=nx, ny=ny, nz=nz,
        isocenter_mm=isocenter_mm,
        rho_scale=rho_scale,
        t1_scale=t1_scale, t2_scale=t2_scale,
        t2_floor_s=t2_floor_s,
    )
    print(f"  Phantom: {phantom['n_spins']:,} non-background spins")
    print(f"  Crop shape: {phantom['crop_shape']}  voxel: {phantom['voxel_mm']:.3f} mm")

    # ── 3. Save phantom ──────────────────────────────────────────────────
    print("\n[3/5] Saving phantom...")
    phantom_dir = out_dir / "phantom"
    phantom_dir.mkdir(exist_ok=True)

    if use_hdf5 and HDF5_AVAILABLE:
        phantom_path = str(phantom_dir / "phantom_3d.h5")
        save_phantom_hdf5_3d(phantom, phantom_path)
    else:
        phantom_path = str(phantom_dir / "phantom_3d.json")
        save_phantom_json_3d(phantom, phantom_path)
    print(f"  Saved: {phantom_path}")

    # ── 4. Simulate ──────────────────────────────────────────────────────
    print("\n[4/5] Running KomaMRI simulation (Julia)...")
    sim_dir = out_dir / "simulation"
    run_simulation_julia_3d(
        sequence_file=sequence_file,
        phantom_path=phantom_path,
        output_dir=str(sim_dir),
        b0=b0,
        use_gpu=use_gpu,
        n_threads=n_threads,
        julia_script=julia_script,
    )

    kspace_path = str(sim_dir / "k.npz")
    if not Path(kspace_path).exists():
        raise FileNotFoundError(
            f"Simulation did not produce k.npz at {kspace_path}. "
            "Check Julia output for errors."
        )

    # ── 5. Reconstruct ───────────────────────────────────────────────────
    print("\n[5/5] Reconstructing 3D volume...")
    vol = reconstruct_3d(
        kspace_path=kspace_path,
        nx=nx, ny=ny, nz=nz,
        acquisition=acquisition,
        etl=etl,
        apply_hamming=apply_hamming,
        remove_os=remove_os,
    )

    # ── 6. Save NIfTI ────────────────────────────────────────────────────
    if final_nifti_path is None:
        seq_stem = Path(sequence_file).stem
        final_nifti_path = str(out_dir / f"{seq_stem}_recon.nii.gz")

    save_nifti_3d(
        vol, final_nifti_path,
        fov_mm=(fov_mm[0], fov_mm[1]),
        slab_mm=slab_mm,
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"  Output NIfTI: {final_nifti_path}")
    print(f"{'='*60}")

    return final_nifti_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D MRI simulation pipeline (KomaMRI + PyPulseq).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rho", required=True, help="Proton density NIfTI (.nii/.nii.gz).")
    parser.add_argument("--t1",  required=True, help="T1 map NIfTI (values in ms).")
    parser.add_argument("--t2",  required=True, help="T2 map NIfTI (values in ms).")
    parser.add_argument("--seq", required=True, help="3D Pulseq .seq file.")
    parser.add_argument("--out", required=True, help="Output directory.")

    parser.add_argument("--isocenter", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"), help="Isocenter offset (mm).")
    parser.add_argument("--b0", type=float, default=3.0, help="Main field strength (T).")
    parser.add_argument("--threads", type=int, default=4, help="Julia/CPU threads.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CUDA) in KomaMRI.")
    parser.add_argument("--no-hamming", action="store_true", help="Disable Hamming window.")
    parser.add_argument("--remove-os", action="store_true", help="Remove readout oversampling.")
    parser.add_argument("--json-phantom", action="store_true",
                        help="Save phantom as JSON (default: HDF5 if h5py available).")
    parser.add_argument("--julia-script", default=None,
                        help="Path to simulate_batch_final.jl (auto-detected if omitted).")
    parser.add_argument("--out-nifti", default=None,
                        help="Final NIfTI output path (default: <out>/<seq>_recon.nii.gz).")
    parser.add_argument("--t1-scale", type=float, default=1e-3,
                        help="T1 NIfTI unit → seconds scale factor (default 1e-3 for ms).")
    parser.add_argument("--t2-scale", type=float, default=1e-3,
                        help="T2 NIfTI unit → seconds scale factor.")

    args = parser.parse_args()

    run_pipeline_3d(
        rho_path=args.rho,
        t1_path=args.t1,
        t2_path=args.t2,
        sequence_file=args.seq,
        output_dir=args.out,
        isocenter_mm=tuple(args.isocenter),
        b0=args.b0,
        use_gpu=args.gpu,
        n_threads=args.threads,
        apply_hamming=not args.no_hamming,
        remove_os=args.remove_os,
        use_hdf5=not args.json_phantom,
        julia_script=args.julia_script,
        t1_scale=args.t1_scale,
        t2_scale=args.t2_scale,
        final_nifti_path=args.out_nifti,
    )


if __name__ == "__main__":
    main()
