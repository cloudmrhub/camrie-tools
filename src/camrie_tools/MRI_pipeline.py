#!/usr/bin/env python3
"""
mri_pipeline_final.py — Corrected Unified MRI Simulation Pipeline

All known bugs fixed:
- Sub-voxel method write-block scoping
- flip_phase auto-detection (k_y trajectory walk)
- k-space flipud before IFFT (not image-domain)
- Separated display vs placement flips
- Oversampling removal keeps central half
- Trilinear interpolation index clamping
- T2 default = large value (not zero)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# HDF5 support (optional)
HDF5_AVAILABLE = False
try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-Voxel Spin Placement
# ═══════════════════════════════════════════════════════════════════════════════

SPIN_METHODS_SUPPORTED = {
    "stratified", "halton", "sobol", "poisson_disk", "sobol_rotated",
    "random", "adaptive_edge", "adaptive_rho", "importance_grad",
    "octree_adaptive", "voxel_center_boundary_refine", "source_voxel_sampling",
}
DEFAULT_SPIN_METHOD = "voxel_center_boundary_refine"


def generate_subvoxel_offsets(
    dx: float, dy: float, dz: float, n_spins: int = 8,
    method: str = "voxel_center_boundary_refine",
) -> np.ndarray:
    method = str(method).strip().lower()
    n_spins = max(int(n_spins), 0)
    if n_spins == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if method == "stratified":
        n_per_axis = max(1, int(math.ceil(n_spins ** (1 / 3))))
        edges_x = np.linspace(-dx / 2, dx / 2, n_per_axis + 1)
        edges_y = np.linspace(-dy / 2, dy / 2, n_per_axis + 1)
        edges_z = np.linspace(-dz / 2, dz / 2, n_per_axis + 1)
        cx = (edges_x[:-1] + edges_x[1:]) / 2
        cy = (edges_y[:-1] + edges_y[1:]) / 2
        cz = (edges_z[:-1] + edges_z[1:]) / 2
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        jx = (edges_x[1] - edges_x[0]) * 0.4
        jy = (edges_y[1] - edges_y[0]) * 0.4
        jz = (edges_z[1] - edges_z[0]) * 0.4
        offsets_all = np.column_stack([
            gx.ravel() + np.random.uniform(-jx, jx, gx.size),
            gy.ravel() + np.random.uniform(-jy, jy, gy.size),
            gz.ravel() + np.random.uniform(-jz, jz, gz.size),
        ]).astype(np.float32)
        if offsets_all.shape[0] == n_spins:
            return offsets_all
        keep = np.random.permutation(offsets_all.shape[0])[:n_spins]
        return offsets_all[keep]

    elif method == "halton":
        try:
            from scipy.stats.qmc import Halton
            sampler = Halton(d=3, scramble=True)
            samples = sampler.random(n=n_spins)
            samples[:, 0] = (samples[:, 0] - 0.5) * dx
            samples[:, 1] = (samples[:, 1] - 0.5) * dy
            samples[:, 2] = (samples[:, 2] - 0.5) * dz
            return samples.astype(np.float32)
        except ImportError:
            pass

    elif method == "sobol":
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=3, scramble=True)
            m = int(math.ceil(math.log2(max(1, n_spins))))
            samples = sampler.random_base2(m=m)[:n_spins, :]
            samples[:, 0] = (samples[:, 0] - 0.5) * dx
            samples[:, 1] = (samples[:, 1] - 0.5) * dy
            samples[:, 2] = (samples[:, 2] - 0.5) * dz
            return samples.astype(np.float32)
        except ImportError:
            pass

    elif method == "voxel_center_boundary_refine":
        offsets: List[np.ndarray] = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        face = 0.45
        face_pts = [
            [face * dx, 0, 0], [-face * dx, 0, 0],
            [0, face * dy, 0], [0, -face * dy, 0],
            [0, 0, face * dz], [0, 0, -face * dz],
        ]
        edge = 0.35
        edge_pts = [
            [edge * dx, edge * dy, 0], [edge * dx, -edge * dy, 0],
            [-edge * dx, edge * dy, 0], [-edge * dx, -edge * dy, 0],
            [edge * dx, 0, edge * dz], [edge * dx, 0, -edge * dz],
            [-edge * dx, 0, edge * dz], [-edge * dx, 0, -edge * dz],
            [0, edge * dy, edge * dz], [0, edge * dy, -edge * dz],
            [0, -edge * dy, edge * dz], [0, -edge * dy, -edge * dz],
        ]
        for p in face_pts + edge_pts:
            offsets.append(np.array(p, dtype=np.float32))
        arr = np.asarray(offsets, dtype=np.float32)
        if arr.shape[0] > 1:
            jit = np.column_stack([
                np.random.uniform(-0.04 * dx, 0.04 * dx, arr.shape[0] - 1),
                np.random.uniform(-0.04 * dy, 0.04 * dy, arr.shape[0] - 1),
                np.random.uniform(-0.04 * dz, 0.04 * dz, arr.shape[0] - 1),
            ]).astype(np.float32)
            arr[1:, :] += jit
        if n_spins <= arr.shape[0]:
            return arr[:n_spins]
        pad = n_spins - arr.shape[0]
        extra = np.column_stack([
            np.random.uniform(-dx / 2, dx / 2, pad),
            np.random.uniform(-dy / 2, dy / 2, pad),
            np.random.uniform(-dz / 2, dz / 2, pad),
        ]).astype(np.float32)
        return np.vstack([arr, extra]).astype(np.float32)

    # random fallback
    return np.column_stack([
        np.random.uniform(-dx / 2, dx / 2, n_spins),
        np.random.uniform(-dy / 2, dy / 2, n_spins),
        np.random.uniform(-dz / 2, dz / 2, n_spins),
    ]).astype(np.float32)


def decorrelate_offsets_per_voxel(
    offsets: np.ndarray, n_voxels: int,
    dx: float, dy: float, dz: float, shift_frac: float = 0.12,
) -> np.ndarray:
    n_voxels = max(int(n_voxels), 0)
    if n_voxels == 0:
        return np.zeros((0, int(offsets.shape[0]), 3), dtype=np.float32)
    base = np.asarray(offsets, dtype=np.float32)
    out = np.broadcast_to(base[np.newaxis, :, :], (n_voxels, base.shape[0], 3)).copy()
    theta = np.random.uniform(0, 2 * np.pi, n_voxels).astype(np.float32)
    c = np.cos(theta)[:, np.newaxis]
    s = np.sin(theta)[:, np.newaxis]
    x0, y0 = out[:, :, 0].copy(), out[:, :, 1].copy()
    out[:, :, 0] = c * x0 - s * y0
    out[:, :, 1] = s * x0 + c * y0
    for ax in range(3):
        flip = np.where(np.random.uniform(0, 1, n_voxels) < 0.5, -1, 1).astype(np.float32)[:, np.newaxis]
        out[:, :, ax] *= flip
    sh = np.column_stack([
        np.random.uniform(-shift_frac * dx, shift_frac * dx, n_voxels),
        np.random.uniform(-shift_frac * dy, shift_frac * dy, n_voxels),
        np.random.uniform(-shift_frac * dz, shift_frac * dz, n_voxels),
    ]).astype(np.float32)
    out += sh[:, np.newaxis, :]
    out[:, :, 0] = np.clip(out[:, :, 0], -dx / 2, dx / 2)
    out[:, :, 1] = np.clip(out[:, :, 1], -dy / 2, dy / 2)
    out[:, :, 2] = np.clip(out[:, :, 2], -dz / 2, dz / 2)
    return out.astype(np.float32, copy=False)


def normalize_spin_methods(spin_method):
    raw = []
    if spin_method is None:
        raw = [DEFAULT_SPIN_METHOD]
    elif isinstance(spin_method, str):
        for token in spin_method.replace(",", " ").split():
            t = token.strip().lower()
            if t:
                raw.append(t)
    else:
        for item in spin_method:
            if item is None:
                continue
            for token in str(item).replace(",", " ").split():
                t = token.strip().lower()
                if t:
                    raw.append(t)
    if not raw:
        raw = [DEFAULT_SPIN_METHOD]
    methods = []
    for m in raw:
        if m not in SPIN_METHODS_SUPPORTED:
            raise ValueError(f"Unknown spin method '{m}'")
        if m not in methods:
            methods.append(m)
    return methods


def split_spin_count(total_spins: int, n_methods: int) -> List[int]:
    total_spins = max(int(total_spins), 0)
    n_methods = max(int(n_methods), 1)
    base = total_spins // n_methods
    rem = total_spins % n_methods
    return [base + (1 if i < rem else 0) for i in range(n_methods)]


def robust_unit_interval(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 95))
    if hi <= lo + 1e-8:
        lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# ★ FIX #7: Vectorised Trilinear Interpolation with lower-bound clamping
# ═══════════════════════════════════════════════════════════════════════════════

def _trilinear_interp_vectorised(
    arr: np.ndarray, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray,
) -> np.ndarray:
    nz, ny_arr, nx_arr = arr.shape

    ix0 = np.floor(ix).astype(np.intp)
    iy0 = np.floor(iy).astype(np.intp)
    iz0 = np.floor(iz).astype(np.intp)

    # ★ FIX: clamp lower bound to prevent negative-index wrap-around
    ix0 = np.clip(ix0, 0, nx_arr - 1)
    iy0 = np.clip(iy0, 0, ny_arr - 1)
    iz0 = np.clip(iz0, 0, nz - 1)

    ix1 = np.minimum(ix0 + 1, nx_arr - 1)
    iy1 = np.minimum(iy0 + 1, ny_arr - 1)
    iz1 = np.minimum(iz0 + 1, nz - 1)

    fx = (ix - ix0).astype(np.float32)
    fy = (iy - iy0).astype(np.float32)
    fz = (iz - iz0).astype(np.float32)

    c000 = arr[iz0, iy0, ix0]; c001 = arr[iz0, iy0, ix1]
    c010 = arr[iz0, iy1, ix0]; c011 = arr[iz0, iy1, ix1]
    c100 = arr[iz1, iy0, ix0]; c101 = arr[iz1, iy0, ix1]
    c110 = arr[iz1, iy1, ix0]; c111 = arr[iz1, iy1, ix1]

    c00 = c000 * (1 - fx) + c001 * fx
    c01 = c010 * (1 - fx) + c011 * fx
    c10 = c100 * (1 - fx) + c101 * fx
    c11 = c110 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c01 * fy
    c1 = c10 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


# ═══════════════════════════════════════════════════════════════════════════════
# HDF5 / JSON Phantom I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_phantom_hdf5(phantom: Dict[str, Any], output_path: str) -> str:
    if not HDF5_AVAILABLE:
        raise ImportError("h5py required for HDF5")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for key in ["x", "y", "z", "rho", "t1", "t2", "t2s", "dw"]:
            data = phantom.get(key)
            if data is not None:
                f.create_dataset(key, data=np.asarray(data, dtype=np.float32),
                                 compression="gzip", compression_opts=1, chunks=True)
        if "slice_geometry" in phantom:
            f.attrs["slice_geometry"] = json.dumps(
                phantom["slice_geometry"],
                default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
        if "name" in phantom:
            f.attrs["name"] = phantom["name"]
    return output_path


def save_phantom_json(phantom: Dict[str, Any], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    serialisable = {}
    for key, val in phantom.items():
        serialisable[key] = val.tolist() if isinstance(val, np.ndarray) else val
    with open(output_path, "w") as fh:
        json.dump(serialisable, fh)
    return output_path


def load_phantom_auto(filepath: str) -> Dict[str, Any]:
    if filepath.endswith(".h5") or filepath.endswith(".hdf5"):
        if not HDF5_AVAILABLE:
            raise ImportError("h5py required")
        phantom = {}
        with h5py.File(filepath, "r") as f:
            for key in ["x", "y", "z", "rho", "t1", "t2", "t2s", "dw"]:
                if key in f:
                    phantom[key] = f[key][:].tolist()
            if "slice_geometry" in f.attrs:
                phantom["slice_geometry"] = json.loads(f.attrs["slice_geometry"])
            if "name" in f.attrs:
                phantom["name"] = f.attrs["name"]
        return phantom
    else:
        with open(filepath) as fh:
            return json.load(fh)


# ═══════════════════════════════════════════════════════════════════════════════
# ★ FIX #2: Pulseq Reader with flip_phase auto-detection
# ═══════════════════════════════════════════════════════════════════════════════

def _build_phase_reorder_from_ky(ky_trace: List[float]) -> Dict[str, Any]:
    """Decide whether phase-encode rows should be reordered before IFFT."""
    meta: Dict[str, Any] = {
        "phase_reorder": False,
        "phase_reorder_indices": None,
        "phase_reorder_reason": "insufficient_adc",
    }
    if len(ky_trace) < 2:
        return meta

    ky = np.asarray(ky_trace, dtype=np.float64)
    dky = np.diff(ky)
    monotonic = bool(np.all(dky >= -1e-12) or np.all(dky <= 1e-12))
    unique_lines = int(np.unique(np.round(ky, 9)).size)
    min_unique = max(4, int(math.ceil(0.95 * ky.size)))

    if unique_lines < min_unique:
        meta["phase_reorder_reason"] = f"non_unique_ky({unique_lines}/{ky.size})"
        return meta
    if monotonic:
        meta["phase_reorder_reason"] = "already_monotonic"
        return meta

    meta["phase_reorder"] = True
    meta["phase_reorder_indices"] = np.argsort(ky, kind="mergesort").astype(int).tolist()
    meta["phase_reorder_reason"] = "non_monotonic_ky"
    return meta


def _rf_flip_angle_deg(rf) -> Optional[float]:
    """Flip angle of a pypulseq RF event, in degrees.

    Prefers rf.flip_angle, falling back to integrating the waveform, because
    some pypulseq versions do not expose flip_angle. Signal is in Hz, so
    FA[rad] = 2*pi * sum(|signal|) * dt.
    """
    fa = getattr(rf, "flip_angle", None)
    if fa is not None:
        try:
            return float(np.rad2deg(float(fa)))
        except Exception:
            pass
    sig = getattr(rf, "signal", None)
    tt = getattr(rf, "t", None)
    if sig is None or tt is None:
        return None
    sig = np.atleast_1d(np.asarray(sig, dtype=complex))
    tt = np.atleast_1d(np.asarray(tt, dtype=float))
    if sig.size < 2 or tt.size < 2:
        return None
    dt = float(np.median(np.diff(tt)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    return float(np.rad2deg(2 * np.pi * float(np.sum(np.abs(sig))) * dt))


def _classify_rf_flip_angles(seq) -> Tuple[Optional[float], Optional[float]]:
    """Return (excitation_fa, refocusing_fa) in degrees, either may be None.

    Refocusing pulses are identified as a distinctly larger flip angle than the
    excitation (e.g. 90/180). A single flip angle means no refocusing pulses,
    which is the gradient-echo case.
    """
    angles = []
    for idx in range(1, len(seq.block_events) + 1):
        try:
            blk = seq.get_block(idx)
        except Exception:
            continue
        rf = getattr(blk, "rf", None)
        if rf is not None:
            fa = _rf_flip_angle_deg(rf)
            if fa is not None and np.isfinite(fa):
                angles.append(round(float(fa), 1))
    if not angles:
        return None, None
    distinct = sorted(set(angles))
    exc = distinct[0]
    refoc = distinct[-1] if len(distinct) > 1 and distinct[-1] > 1.5 * exc else None
    return exc, refoc

# ─────────────────────────────────────────────────────────────────────────────
# mtrk (SDL / YARRA) field-lookup helpers
# ─────────────────────────────────────────────────────────────────────────────
# The mtrk JSON format does not expose every acquisition parameter under a single
# fixed key, and spellings vary between exporters.  The tuples below list the
# accepted (case-insensitive) key names for each parameter, and the helpers below
# resolve values robustly across the "infos"/"settings" maps and the per-object
# definitions in "objects".

# Phase-encode line count.  Primary key is "pelines"; accept common variants.
_MTRK_PELINE_KEYS = (
    "pelines", "pe_lines", "phaselines", "phase_lines", "phaseencodes", "nphase",
)
# Frequency-encode / readout sample count.  Mirrors the "pelines" naming pattern.
# (Not present in "infos" for the reference sequences; kept for forward-compat.)
_MTRK_FREQ_KEYS = (
    "felines", "fe_lines", "freqlines", "freq_lines", "readout_points",
    "readout_samples", "readoutsamples", "readout_lines", "readoutlines",
    "nf", "samples",
)
# Field of view (mm).
_MTRK_FOV_KEYS = ("fov", "field_of_view")
# Slice thickness.
_MTRK_ST_KEYS = (
    "slice_thickness", "slicethickness", "slice_thickness_mm", "st", "thickness",
)
# Readout oversampling factor.
_MTRK_OS_KEYS = ("readout_os", "readout_oversampling", "readoutos", "oversampling")


def _mtrk_ci_get(mapping: Any, candidate_keys, default=None):
    """Case-insensitive lookup of the first matching key in ``mapping``.

    Returns ``default`` if ``mapping`` is not a dict or no candidate matches.
    """
    if not isinstance(mapping, dict):
        return default
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in candidate_keys:
        if str(key).lower() in lowered:
            return lowered[str(key).lower()]
    return default


def _mtrk_iter_objects(seq: Dict[str, Any]):
    """Yield (name, object_dict) for every entry in the mtrk ``objects`` map."""
    objects = seq.get("objects", {})
    if isinstance(objects, dict):
        for name, obj in objects.items():
            if isinstance(obj, dict):
                yield name, obj


def _mtrk_referenced_object_name(seq: Dict[str, Any], action: str):
    """Name of the first object referenced by a step whose ``action`` matches.

    mtrk instructions reference objects by name (e.g. an ``"action": "adc"`` step
    points at the ADC object via ``"object"``), which is the most reliable way to
    find the object actually used by the sequence.
    """
    instructions = seq.get("instructions", {})
    if not isinstance(instructions, dict):
        return None
    for block in instructions.values():
        if not isinstance(block, dict):
            continue
        for step in block.get("steps", []) or []:
            if isinstance(step, dict) and step.get("action") == action:
                name = step.get("object")
                if name:
                    return name
    return None


def _mtrk_find_adc_object(seq: Dict[str, Any]):
    """Return the ADC object dict (preferring the one referenced by the sequence)."""
    objects = seq.get("objects", {})
    ref = _mtrk_referenced_object_name(seq, "adc")
    if ref and isinstance(objects, dict) and isinstance(objects.get(ref), dict):
        return objects[ref]
    for _name, obj in _mtrk_iter_objects(seq):
        if str(obj.get("type", "")).lower() == "adc":
            return obj
    return None


def _mtrk_find_excitation_rf_object(seq: Dict[str, Any]):
    """Return the excitation RF object dict, which carries the slice thickness.

    Prefers ``purpose == "excitation"``; otherwise the first RF object exposing a
    ``thickness`` field; otherwise any RF object.
    """
    rf_objs = [obj for _n, obj in _mtrk_iter_objects(seq)
               if str(obj.get("type", "")).lower() == "rf"]
    for obj in rf_objs:
        if str(obj.get("purpose", "")).lower() == "excitation":
            return obj
    for obj in rf_objs:
        if "thickness" in obj:
            return obj
    return rf_objs[0] if rf_objs else None


def read_mtrk_params(seq_path: str) -> Dict[str, Any]:
    import json
    import numbers

    with open(seq_path) as fh:
        seq = json.load(fh)

    if "infos" not in seq:
        raise KeyError(f'unable to find "infos" field in mtrk file "{seq_path}"')
    info = seq["infos"]
    settings = seq.get("settings", {})

    # ── FOV (mm) ──────────────────────────────────────────────────────────────
    # mtrk stores FOV in mm under "fov" (case-insensitive); may be scalar or list.
    fov_raw = _mtrk_ci_get(info, _MTRK_FOV_KEYS, 300)
    if isinstance(fov_raw, numbers.Number):
        fov_mm = [float(fov_raw), float(fov_raw)]
    elif isinstance(fov_raw, (list, tuple, np.ndarray)):
        fov_list = [float(v) for v in fov_raw]
        if len(fov_list) == 0:
            fov_mm = [300.0, 300.0]
        elif len(fov_list) == 1:
            fov_mm = [fov_list[0], fov_list[0]]
        else:
            fov_mm = fov_list[0:2]
    else:
        raise ValueError(f"error parsing fov_mm: {fov_raw!r}")

    # ── Readout oversampling ──────────────────────────────────────────────────
    # Lives in "settings" (e.g. {"readout_os": 2}); accept spelling variants.
    os_raw = _mtrk_ci_get(settings, _MTRK_OS_KEYS, None)
    oversampling = int(os_raw) if isinstance(os_raw, numbers.Number) and int(os_raw) >= 1 else 1

    # ── Phase-encode lines (nP) ───────────────────────────────────────────────
    pelines_raw = _mtrk_ci_get(info, _MTRK_PELINE_KEYS, None)
    nP = int(pelines_raw) if isinstance(pelines_raw, numbers.Number) and int(pelines_raw) > 0 else 128

    # ── Frequency-encode samples (nF) ─────────────────────────────────────────
    # Resolve the *final* (non-oversampled) readout matrix size, in priority order:
    #   1) an explicit "infos" field (sibling of "pelines"; not present in the
    #      reference sequences but supported for forward-compatibility),
    #   2) the ADC object's "samples" count — where mtrk actually records the
    #      readout length (confirmed by duration/dwelltime: samples*dwell==duration).
    #      This is the base-resolution readout count; "readout_os" is applied on
    #      top of it (nF_raw below), not baked into "samples".
    #   3) the FOV aspect ratio as a last resort, since the freq-encode direction
    #      corresponds to fov_mm[0]:  nF = round(nP * fov_mm[0] / fov_mm[1]).
    nF = None
    nF_source = None
    freq_raw = _mtrk_ci_get(info, _MTRK_FREQ_KEYS, None)
    if isinstance(freq_raw, numbers.Number) and int(freq_raw) > 0:
        nF, nF_source = int(freq_raw), "infos"
    if nF is None:
        adc_obj = _mtrk_find_adc_object(seq)
        samples = adc_obj.get("samples") if isinstance(adc_obj, dict) else None
        if isinstance(samples, numbers.Number) and int(samples) > 0:
            nF, nF_source = int(samples), "adc.samples"
    if nF is None:
        if fov_mm[1] > 0:
            nF = max(1, int(round(nP * fov_mm[0] / fov_mm[1])))
        else:
            nF = nP
        nF_source = "fov_aspect"

    nF_raw = oversampling * nF

    # ── Slice thickness (mm) ──────────────────────────────────────────────────
    # The reference mtrk format does not carry slice thickness in "infos" or
    # "settings"; it lives on the excitation RF object as "thickness" (already mm).
    # Check the documented locations first, then fall back to the RF object.
    seq_slice_thickness_mm = None
    st_raw = _mtrk_ci_get(info, _MTRK_ST_KEYS, None)
    if st_raw is None:
        st_raw = _mtrk_ci_get(settings, _MTRK_ST_KEYS, None)
    if st_raw is None:
        rf_obj = _mtrk_find_excitation_rf_object(seq)
        if isinstance(rf_obj, dict):
            st_raw = rf_obj.get("thickness")
    if isinstance(st_raw, numbers.Number) and float(st_raw) > 0:
        seq_slice_thickness_mm = float(st_raw)  # mtrk RF thickness is in mm
    else:
        print("mtrk: slice_thickness_mm not found in sequence file; "
              "caller must provide via geometry")

    detected_ro_sign = -1
    # flip_phase was hardcoded False, which produced phase-encode-FLIPPED images.
    # Measured against a chiral (right-triangle) phantom by correlating the
    # assembled reconstruction against the input rho on the same body grid:
    #   PD-Weighted_Spin_Echo.mtrk  identity 0.485 vs flipud 0.956 -> needs True
    #   T1-Weighted_Spin_Echo.mtrk  identity 0.433 vs flipud 0.933 -> needs True
    #   T2-Weighted_Spin_Echo.mtrk  identity 0.419 vs flipud 0.904 -> needs True
    # and confirmed directly via flip_phase_override on the T1 file (0.9329
    # correct vs 0.4329 flipped). The matching .seq files all report True and all
    # reconstruct correctly, so True is consistent across both readers for the
    # same physical sequences.
    #
    # This is an EMPIRICAL value, not derived from the file. Deriving it from the
    # phase-encoding equation was attempted and failed: the equation is a scale
    # factor on a gradient array, and eq(0)*sum(array) predicts False for files
    # that demonstrably need True, so at least one polarity in the mtrk 'phase'
    # logical-axis -> Koma gy chain is still unaccounted for.
    #
    # NOT VALIDATED FOR: mtrk_spoiled_gre.mtrk, which matches no simple flip
    # (best correlation 0.43) and uses a different equation style; and
    # T1-Weighted_Spoiled_GRE.mtrk, which cannot be simulated at all because of
    # the KomaInterface mtrk binning crash.
    flip_phase = True
    needs_resort = False
    phase_reorder_reason = "already_monotonic"

    params = {
        "nF": nF, "nF_raw": nF_raw, "nP": nP,
        "oversampling": oversampling, "fov_mm": fov_mm,
        "slice_thickness_mm": seq_slice_thickness_mm,
        "duration_s": None, "n_blocks": None,
        "source": "mtrk",
        "etl": 1,
        "echo_spacing_ms": None,
        "te_eff_ms": None,
        "orientation": {
            "detected_ro_sign": detected_ro_sign,
            "flip_phase": flip_phase,
            "from_seq": False,
            "ky_trajectory": None,
            "needs_resort": needs_resort,
            "phase_reorder_indices": None,
            "phase_reorder_reason": phase_reorder_reason
        },
    }
    st_str = f"{seq_slice_thickness_mm:.1f}" if seq_slice_thickness_mm is not None else "None"
    print(f"Sequence: nF_raw={nF_raw}, nF={nF} (from {nF_source}), nP={nP}, "
            f"OS={oversampling}x, FOV={fov_mm[0]:.0f}x{fov_mm[1]:.0f}mm, "
            f"ST={st_str}mm, "
            f"Gx_sign={detected_ro_sign:+d}, flip_phase={flip_phase}, "
            f"resort_ky={needs_resort} ({phase_reorder_reason})")

    return params

def read_pulseq_params(seq_path: str) -> Dict[str, Any]:
    try:
        import pypulseq as pp
        seq = pp.Sequence()
        seq.read(seq_path)
        definitions = seq.definitions

        fov_x = definitions.get("FOV", [0.3, 0.3, 0.005])
        if isinstance(fov_x, np.ndarray):
            fov_vals = [float(fov_x[i]) for i in range(len(fov_x))]
        elif isinstance(fov_x, (list, tuple)):
            fov_vals = [float(f) for f in fov_x]
        else:
            fov_vals = [float(fov_x), float(fov_x), 0.005]
        fov_mm = [fov_vals[0] * 1000, fov_vals[1] * 1000]
        if len(fov_mm) < 2:
            fov_mm = [fov_mm[0], fov_mm[0]]

        _st = definitions.get("SliceThickness", None)
        if _st is not None:
            seq_slice_thickness_mm = float(_st) * 1000
        elif len(fov_vals) >= 3 and fov_vals[2] > 0:
            seq_slice_thickness_mm = fov_vals[2] * 1000
        else:
            seq_slice_thickness_mm = None

        nF = 0
        nP = 0
        adc_blocks = []
        detected_ro_sign = None

        for block_idx in range(1, len(seq.block_events) + 1):
            try:
                block = seq.get_block(block_idx)
                if hasattr(block, "adc") and block.adc is not None:
                    adc_blocks.append(block_idx)
                    nF = max(nF, int(block.adc.num_samples))
                    if detected_ro_sign is None:
                        gx = getattr(block, "gx", None)
                        if gx is not None:
                            if hasattr(gx, "waveform"):
                                _mid = gx.waveform
                                _flat = _mid[len(_mid) // 4: 3 * len(_mid) // 4]
                                detected_ro_sign = -1 if np.mean(_flat) < 0 else 1
                            elif hasattr(gx, "amplitude"):
                                detected_ro_sign = -1 if gx.amplitude < 0 else 1
            except Exception:
                continue

        if detected_ro_sign is None:
            detected_ro_sign = -1
        nP = len(adc_blocks)

        nF_raw = nF
        if nF == 2 * nP:
            oversampling = 2
            nF_final = nF // 2
        else:
            oversampling = 1
            nF_final = nF

        _dur = seq.duration()
        duration = float(_dur[0]) if hasattr(_dur, "__iter__") else float(_dur)
        detected_ro_sign = int(detected_ro_sign)

        # Extract full ky trajectory: record ky at every ADC event.
        # Robust to pypulseq versions where rf.flip_angle is unavailable.
        # An EXCITATION starts a new TR and resets ky to 0; only a REFOCUSING
        # pulse mirrors it (ky -> -ky). Negating on every RF is refocusing logic
        # and corrupts gradient-echo trains, where each TR begins with a fresh
        # excitation: for a GRE that yielded 28/128 unique, non-monotonic ky.
        # Spin echoes are unaffected because their net per-TR ky returns to zero
        # before the next excitation, which makes negation and reset equivalent.
        _exc_fa, _refoc_fa = _classify_rf_flip_angles(seq)
        _can_classify = _exc_fa is not None
        _ky_accum = 0.0
        _rf_count = 0
        _ky_at_first_adc = None
        _ky_trajectory = []
        for _bidx in range(1, len(seq.block_events) + 1):
            try:
                _blk = seq.get_block(_bidx)
                _rf_obj = getattr(_blk, "rf", None)
                _has_rf = bool(_rf_obj)
                _has_adc = bool(getattr(_blk, "adc", None))
                _gy = getattr(_blk, "gy", None)
                if _has_rf:
                    _rf_count += 1
                    if not _can_classify:
                        # No usable flip angles: keep the legacy behaviour rather
                        # than guess on an exotic file.
                        if _rf_count >= 2:
                            _ky_accum = -_ky_accum
                    else:
                        _fa = _rf_flip_angle_deg(_rf_obj)
                        _is_refoc = (
                            _refoc_fa is not None
                            and _fa is not None
                            and abs(_fa - _refoc_fa) < abs(_fa - _exc_fa)
                        )
                        _ky_accum = -_ky_accum if _is_refoc else 0.0
                if _gy is not None and hasattr(_gy, "area"):
                    _ky_accum += float(_gy.area)
                if _has_adc:
                    if _ky_at_first_adc is None:
                        _ky_at_first_adc = float(_ky_accum)
                    _ky_trajectory.append(float(_ky_accum))
            except Exception:
                continue

        _ky_arr = np.array(_ky_trajectory, dtype=float)
        flip_phase = (_ky_at_first_adc is not None and _ky_at_first_adc > 0)
        _phase_reorder = _build_phase_reorder_from_ky(_ky_trajectory)
        _needs_resort = bool(_phase_reorder["phase_reorder"])

        # Extract TSE-specific metadata from .seq definitions
        _etl = definitions.get("TurboFactor", None)
        _echo_spacing = definitions.get("TE", None)   # TE = echo spacing in TSE
        _te_eff = definitions.get("TEeff", None)
        etl = int(float(_etl)) if _etl is not None else 1
        echo_spacing_ms = float(_echo_spacing) * 1000 if _echo_spacing is not None else None
        te_eff_ms = float(_te_eff) * 1000 if _te_eff is not None else echo_spacing_ms
        if etl > 1:
            print(f"  TSE params  : ETL={etl}, echo_spacing={echo_spacing_ms:.1f}ms, te_eff={te_eff_ms:.1f}ms")

        params = {
            "nF": nF_final, "nF_raw": nF_raw, "nP": nP,
            "oversampling": oversampling, "fov_mm": fov_mm,
            "slice_thickness_mm": seq_slice_thickness_mm,
            "duration_s": duration, "n_blocks": len(seq.block_events),
            "source": "pypulseq",
            "etl": etl,
            "echo_spacing_ms": echo_spacing_ms,
            "te_eff_ms": te_eff_ms,
            "orientation": {
                "detected_ro_sign": detected_ro_sign,
                "flip_phase": flip_phase,
                "from_seq": False,
                "ky_trajectory": _ky_arr.tolist() if len(_ky_arr) > 0 else None,
                "needs_resort": bool(_needs_resort),
                "phase_reorder_indices": _phase_reorder["phase_reorder_indices"],
                "phase_reorder_reason": _phase_reorder["phase_reorder_reason"],
            },
        }
        print(f"Sequence: nF_raw={nF_raw}, nF={nF_final}, nP={nP}, "
              f"OS={oversampling}x, FOV={fov_mm[0]:.0f}x{fov_mm[1]:.0f}mm, "
              f"Gx_sign={detected_ro_sign:+d}, flip_phase={flip_phase}, "
              f"resort_ky={_needs_resort} ({_phase_reorder['phase_reorder_reason']})")
        return params

    except ImportError:
        print("pypulseq not installed, using fallback")
        return read_pulseq_params_fallback(seq_path)
    except Exception as e:
        print(f"pypulseq failed ({e}), using fallback")
        return read_pulseq_params_fallback(seq_path)


def read_pulseq_params_fallback(seq_path: str) -> Dict[str, Any]:
    params = {"nF": 256, "nF_raw": 256, "nP": 128, "oversampling": 1,
              "fov_mm": [300.0, 300.0], "source": "fallback"}

    with open(seq_path, "r") as f:
        content = f.read()

    in_def = False
    _st_mm = None
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("[DEFINITIONS]"):
            in_def = True; continue
        if line.startswith("[") and in_def:
            in_def = False
        if in_def and line:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                if key.upper() == "FOV":
                    try:
                        vals = [float(v) for v in parts[1:3]]
                        params["fov_mm"] = [vals[0] * 1000, vals[1] * 1000]
                        if len(parts) >= 4:
                            _st_mm = float(parts[3]) * 1000
                    except Exception:
                        pass
                elif key.upper() == "SLICETHICKNESS":
                    try:
                        _st_mm = float(parts[1]) * 1000
                    except Exception:
                        pass
                elif key.upper() == "TURBOFACTOR":
                    try:
                        params["etl"] = int(float(parts[1]))
                    except Exception:
                        pass
                elif key.upper() == "TE":
                    try:
                        params["echo_spacing_ms"] = float(parts[1]) * 1000.0
                    except Exception:
                        pass
                elif key.upper() == "TEEFF":
                    try:
                        params["te_eff_ms"] = float(parts[1]) * 1000.0
                    except Exception:
                        pass
    params["slice_thickness_mm"] = _st_mm

    # Count ADC events
    in_adc = False
    nF_raw = 0
    for line in content.split("\n"):
        if line.startswith("[ADC]"):
            in_adc = True; continue
        if line.startswith("[") and in_adc:
            in_adc = False
        if in_adc and line.strip() and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    nF_raw = max(nF_raw, int(parts[1]))
                except Exception:
                    pass
    if nF_raw > 0:
        params["nF_raw"] = nF_raw

    in_blocks = False
    adc_count = 0
    for line in content.split("\n"):
        if line.startswith("[BLOCKS]"):
            in_blocks = True; continue
        if line.startswith("[") and in_blocks:
            in_blocks = False
        if in_blocks and line.strip() and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 8:
                try:
                    if int(parts[6]) > 0:
                        adc_count += 1
                except Exception:
                    pass
    if adc_count > 0:
        params["nP"] = adc_count
    if params["nF_raw"] == 2 * params["nP"]:
        params["oversampling"] = 2
        params["nF"] = params["nF_raw"] // 2
    else:
        params["nF"] = params["nF_raw"]
        params["oversampling"] = 1

    # Detect readout gradient sign
    detected_ro_sign = -1
    first_gx_id = None
    in_blocks = False
    for line in content.split("\n"):
        if line.startswith("[BLOCKS]"):
            in_blocks = True; continue
        if line.startswith("[") and in_blocks:
            break
        if in_blocks and line.strip() and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 8:
                try:
                    gx_id = int(parts[3])
                    adc_id = int(parts[6])
                    if adc_id > 0 and gx_id > 0:
                        first_gx_id = gx_id
                        break
                except Exception:
                    pass
    if first_gx_id is not None:
        for sect in ["[TRAP]", "[GRADIENTS]"]:
            in_s = False
            for line in content.split("\n"):
                if line.startswith(sect):
                    in_s = True; continue
                if line.startswith("[") and in_s:
                    break
                if in_s and line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            if int(parts[0]) == first_gx_id:
                                detected_ro_sign = -1 if float(parts[1]) < 0 else 1
                                break
                        except Exception:
                            pass

    # Heuristic ky-trajectory extraction in fallback parser.
    # Without pypulseq we don't have RF flip angles, so we infer:
    #   - first seen RF id = excitation (new TR -> reset ky accumulator)
    #   - subsequent same RF id = excitation (reset)
    #   - other RF ids = refocusing (negate ky accumulator)
    _grad_amp: dict = {}
    for _sect in ["[GRADIENTS]", "[TRAP]"]:
        _in_s = False
        for _line in content.split("\n"):
            if _line.startswith(_sect):
                _in_s = True; continue
            if _line.startswith("[") and _in_s:
                break
            if _in_s and _line.strip() and not _line.startswith("#"):
                _p = _line.split()
                try:
                    _grad_amp.setdefault(int(_p[0]), float(_p[1]))
                except Exception:
                    pass

    _ky_accum = 0.0
    _ky_trajectory = []
    _exc_rf_id = None
    _in_blk = False
    for _line in content.split("\n"):
        if _line.startswith("[BLOCKS]"):
            _in_blk = True; continue
        if _line.startswith("[") and _in_blk:
            break
        if _in_blk and _line.strip() and not _line.startswith("#"):
            _p = _line.split()
            if len(_p) == 8:
                try:
                    _rf_id = int(_p[2])
                    _gy_id = int(_p[4])
                    _adc_id = int(_p[6])
                    if _rf_id > 0:
                        if _exc_rf_id is None:
                            _exc_rf_id = _rf_id
                            _ky_accum = 0.0
                        elif _rf_id == _exc_rf_id:
                            _ky_accum = 0.0
                        else:
                            _ky_accum = -_ky_accum
                    if _gy_id > 0:
                        _ky_accum += _grad_amp.get(_gy_id, 0.0)
                    if _adc_id > 0:
                        _ky_trajectory.append(_ky_accum)
                except Exception:
                    pass

    _ky_arr = np.array(_ky_trajectory, dtype=float)
    flip_phase = len(_ky_arr) > 0 and _ky_arr[0] > 0
    _phase_reorder = _build_phase_reorder_from_ky(_ky_trajectory)
    _needs_resort = bool(_phase_reorder["phase_reorder"])

    params["orientation"] = {
        "detected_ro_sign": detected_ro_sign,
        "flip_phase": flip_phase,
        "from_seq": False,
        "ky_trajectory": _ky_arr.tolist() if len(_ky_arr) > 0 else None,
        "needs_resort": bool(_needs_resort),
        "phase_reorder_indices": _phase_reorder["phase_reorder_indices"],
        "phase_reorder_reason": _phase_reorder["phase_reorder_reason"],
    }
    print(f"Fallback: nF_raw={params['nF_raw']}, nF={params['nF']}, nP={params['nP']}, "
          f"Gx_sign={detected_ro_sign:+d}, flip_phase={flip_phase}, "
          f"resort_ky={_needs_resort} ({_phase_reorder['phase_reorder_reason']})")
    return params

def read_sequence_params(seq_path: str) -> Dict[str,Any]:
    if seq_path.endswith(".mtrk"):
        return read_mtrk_params(seq_path)
    elif seq_path.endswith(".seq"):
        return read_pulseq_params(seq_path)
    else:
        raise(ValueError(f"seq_path must end in \".mtrk\" or \".seq\"; found {seq_path}"))

# ═══════════════════════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return v / norm


def build_rotation_matrix(slice_normal: np.ndarray) -> np.ndarray:
    z_seq = normalize(slice_normal.astype(np.float64))
    for c in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
        if abs(np.dot(c, z_seq)) < 0.99:
            x_candidate = c
            break
    x_seq = normalize(x_candidate - np.dot(x_candidate, z_seq) * z_seq)
    y_seq = normalize(np.cross(z_seq, x_seq))
    return np.stack([x_seq, y_seq, z_seq], axis=0)


@dataclass
class SliceSpec:
    normal: np.ndarray
    center_mm: np.ndarray
    position_along_normal: float
    R_body_to_seq: np.ndarray
    index: int

    def to_dict(self) -> Dict:
        return {
            "normal": self.normal.tolist(),
            "center_mm": self.center_mm.tolist(),
            "position_along_normal": self.position_along_normal,
            "R_body_to_seq": self.R_body_to_seq.tolist(),
            "index": self.index,
        }


@dataclass
class SeriesSpec:
    isocenter_mm: np.ndarray
    slice_normal: np.ndarray
    R_body_to_seq: np.ndarray
    slices: List[SliceSpec]
    fov_mm: Tuple[float, float]
    seq_fov_mm: Tuple[float, float]
    slice_thickness_mm: float

    def to_dict(self) -> Dict:
        return {
            "isocenter_mm": self.isocenter_mm.tolist(),
            "slice_normal": self.slice_normal.tolist(),
            "R_body_to_seq": self.R_body_to_seq.tolist(),
            "fov_mm": list(self.fov_mm),
            "seq_fov_mm": list(self.seq_fov_mm),
            "slice_thickness_mm": self.slice_thickness_mm,
            "n_slices": len(self.slices),
            "slices": [s.to_dict() for s in self.slices],
        }


def compute_series_geometry(
    isocenter_mm, slice_normal, num_slices, slice_thickness_mm,
    slice_gap_mm=0.0, fov_mm=(200.0, 200.0), seq_fov_mm=(300.0, 300.0),
) -> SeriesSpec:
    slice_normal = normalize(np.array(slice_normal, dtype=np.float64))
    isocenter_mm = np.array(isocenter_mm, dtype=np.float64)
    R = build_rotation_matrix(slice_normal)
    spacing = slice_thickness_mm + slice_gap_mm
    slices = []
    for i in range(num_slices):
        offset = (i - (num_slices - 1) / 2.0) * spacing
        center = isocenter_mm + offset * slice_normal
        slices.append(SliceSpec(
            normal=slice_normal, center_mm=center,
            position_along_normal=float(np.dot(center, slice_normal)),
            R_body_to_seq=R, index=i))
    return SeriesSpec(
        isocenter_mm=isocenter_mm, slice_normal=slice_normal, R_body_to_seq=R,
        slices=slices, fov_mm=fov_mm, seq_fov_mm=seq_fov_mm,
        slice_thickness_mm=slice_thickness_mm)


# ═══════════════════════════════════════════════════════════════════════════════
# ★ FIX #1: Phantom Extraction — sub-voxel write block properly scoped
# ═══════════════════════════════════════════════════════════════════════════════

def extract_phantom_for_slice(
    rho_img, t1_img, t2_img, slice_spec, fov_mm, resolution_mm,
    spin_factor=1, slice_thickness_mm=5.0, slice_padding=1,
    isotropic_spin_mm=True, rho_scale=100.0, t1_scale=1e-3, t2_scale=1e-3,
    spins_per_voxel=0, spin_method=DEFAULT_SPIN_METHOD, 
    t2star_factor=1.0, spin_axes='xy',
):
    """
    Spin-placement semantics
    ------------------------
    spin_factor = N   : primary lattice density.  Spins per direction are
                        proportional to the voxel dimension so the lattice is
                        isotropic in 3-D space:
                          nx_per_vox = round(N * res_x / d_min)
                          ny_per_vox = round(N * res_y / d_min)
                          nz_per_vox = round(N * thickness / d_min)
                          spin_spacing_i = dim_i / n_i_per_vox
                        where d_min = min(res_x, res_y, thickness).
                        spin_factor=1 always gives exactly 1 spin per voxel.
    spins_per_voxel = M : M *additional* spins placed once per original
                        sequence voxel (not per lattice node), spanning the
                        full voxel extent (resolution_mm × slice_thickness_mm).
                        Total extra spins = n_nonzero_orig_voxels × M.
                        0 = lattice only.
    """
    R = slice_spec.R_body_to_seq
    center_mm = slice_spec.center_mm
    sf = max(int(spin_factor), 1)
    spin_methods = normalize_spin_methods(spin_method)

    # ── Fix 1: proportional lattice — isotropic spin spacing in 3-D ──────
    print(f"Distributing spins along {spin_axes} axes.")
    d_min = min(resolution_mm[0], resolution_mm[1], slice_thickness_mm)
    if 'x' in spin_axes:
        nx_per_vox = max(1, round(sf * resolution_mm[0] / d_min))
    else:
        nx_per_vox = 1    
    if 'y' in spin_axes:
        ny_per_vox = max(1, round(sf * resolution_mm[1] / d_min))
    else:
        ny_per_vox = 1
    if 'z' in spin_axes:
        nz_per_vox = max(1, round(sf * slice_thickness_mm / d_min))
    else:
        nz_per_vox = 1
    spin_spacing_x = resolution_mm[0] / nx_per_vox
    spin_spacing_y = resolution_mm[1] / ny_per_vox
    spin_spacing_z = slice_thickness_mm / nz_per_vox

    half_extent_z = slice_padding * slice_thickness_mm
    if (slice_padding == 0) or ('z' not in spin_axes) or (nz_per_vox < 2):
        z_seq = np.array([0.0])
    else:
        nz = int(np.ceil(2 * half_extent_z / spin_spacing_z))
        nz = (nz + 1) if nz % 2 != 0 else nz # make nz even so that it is symmetric around the center of the slice
        z_seq = np.linspace(-half_extent_z, half_extent_z, nz)

    nx_pts = int(np.ceil(fov_mm[0] / spin_spacing_x))
    ny_pts = int(np.ceil(fov_mm[1] / spin_spacing_y))
    x_seq = (np.arange(nx_pts) - (nx_pts - 1) / 2.0) * spin_spacing_x
    y_seq = (np.arange(ny_pts) - (ny_pts - 1) / 2.0) * spin_spacing_y

    gx, gy, gz = np.meshgrid(x_seq, y_seq, z_seq, indexing="ij")
    pts_seq = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    pts_body = (R.T @ pts_seq.T).T + center_mm[np.newaxis, :]

    rho_arr = sitk.GetArrayFromImage(rho_img)
    t1_arr = sitk.GetArrayFromImage(t1_img)
    # ★ FIX #10: T2 default = large value (not zero) when missing
    t2_arr = (sitk.GetArrayFromImage(t2_img) if t2_img is not None
              else np.full_like(rho_arr, 1000.0))

    origin = np.array(rho_img.GetOrigin(), dtype=np.float64)
    spacing = np.array(rho_img.GetSpacing(), dtype=np.float64)
    if np.max(np.abs(spacing)) < 0.1:  # meter-unit NIfTI: convert to mm
        origin *= 1000.0
        spacing *= 1000.0
    direction = np.array(rho_img.GetDirection(), dtype=np.float64).reshape(3, 3)
    size = np.array(rho_img.GetSize(), dtype=np.float64)

    # Use direction.T for orthogonal matrices (more stable than inv)
    dir_inv = direction.T
    idx_continuous = (dir_inv @ (pts_body - origin[np.newaxis, :]).T).T / spacing[np.newaxis, :]

    ix, iy, iz = idx_continuous[:, 0], idx_continuous[:, 1], idx_continuous[:, 2]
    valid = ((ix >= 0) & (ix < size[0] - 1) &
             (iy >= 0) & (iy < size[1] - 1) &
             (iz >= 0) & (iz < size[2] - 1))

    ix_v, iy_v, iz_v = ix[valid], iy[valid], iz[valid]
    pts_seq_v = pts_seq[valid]

    empty_phantom = {k: [] for k in ["x", "y", "z", "rho", "t1", "t2", "t2s", "dw"]}
    empty_phantom["name"] = f"slice_{slice_spec.index}"
    empty_phantom["slice_geometry"] = slice_spec.to_dict()

    if len(ix_v) == 0:
        return empty_phantom, 0

    rho_v = _trilinear_interp_vectorised(rho_arr, ix_v, iy_v, iz_v)
    rho_koma = rho_v / rho_scale if rho_scale != 0 else rho_v
    nonzero = rho_koma > 0

    if not np.any(nonzero):
        return empty_phantom, 0

    ix_nz, iy_nz, iz_nz = ix_v[nonzero], iy_v[nonzero], iz_v[nonzero]
    pts_nz = pts_seq_v[nonzero]
    rho_nz = rho_koma[nonzero]
    t1_nz = _trilinear_interp_vectorised(t1_arr, ix_nz, iy_nz, iz_nz) * t1_scale
    t2_nz = _trilinear_interp_vectorised(t2_arr, ix_nz, iy_nz, iz_nz) * t2_scale

    # Lattice is always the base; extra spins are appended below if requested.
    all_pts = pts_nz.astype(np.float32)
    all_rho = rho_nz.astype(np.float32)
    all_t1  = t1_nz.astype(np.float32)
    all_t2  = t2_nz.astype(np.float32)

    # ── Fix 2 & 3: Extra spins per *original* sequence voxel ─────────────
    # Build a 1× density grid of original voxel centres (one per seq voxel,
    # independent of spin_factor).  For each nonzero-rho voxel, generate M
    # additional spins placed randomly across the FULL voxel extent
    # (resolution_mm[0/1] × slice_thickness_mm) using spin_method.
    # The spin_factor lattice (all_pts) is not modified.
    if spins_per_voxel > 0:
        # ── Original voxel centre grid (1× density) ──────────────────────
        vdx = resolution_mm[0]          # full voxel extent used for offsets
        vdy = resolution_mm[1]
        vdz = slice_thickness_mm

        v_half_z = slice_padding * vdz
        if slice_padding == 0:
            vz_grid = np.array([0.0])
        else:
            vnz_grid = int(np.ceil(2 * v_half_z / vdz)) + 1
            vz_grid = np.linspace(-v_half_z, v_half_z, vnz_grid)

        vnx = int(np.ceil(fov_mm[0] / vdx))
        vny = int(np.ceil(fov_mm[1] / vdy))
        vx_grid = (np.arange(vnx) - (vnx - 1) / 2.0) * vdx
        vy_grid = (np.arange(vny) - (vny - 1) / 2.0) * vdy

        vgx, vgy, vgz = np.meshgrid(vx_grid, vy_grid, vz_grid, indexing="ij")
        vpts_seq = np.column_stack([vgx.ravel(), vgy.ravel(), vgz.ravel()])
        vpts_body = (R.T @ vpts_seq.T).T + center_mm[np.newaxis, :]

        vidx = (dir_inv @ (vpts_body - origin[np.newaxis, :]).T).T / spacing[np.newaxis, :]
        vix, viy, viz = vidx[:, 0], vidx[:, 1], vidx[:, 2]
        vvalid = ((vix >= 0) & (vix < size[0] - 1) &
                  (viy >= 0) & (viy < size[1] - 1) &
                  (viz >= 0) & (viz < size[2] - 1))
        vix_v, viy_v, viz_v = vix[vvalid], viy[vvalid], viz[vvalid]
        vpts_seq_v = vpts_seq[vvalid]

        if len(vix_v) > 0:
            vrho_v = _trilinear_interp_vectorised(rho_arr, vix_v, viy_v, viz_v)
            vrho_koma = vrho_v / rho_scale if rho_scale != 0 else vrho_v
            vnz_mask = vrho_koma > 0

            vix_nz = vix_v[vnz_mask];  viy_nz = viy_v[vnz_mask];  viz_nz = viz_v[vnz_mask]
            vpts_nz = vpts_seq_v[vnz_mask]
            vrho_nz = vrho_koma[vnz_mask]
            vt1_nz  = _trilinear_interp_vectorised(t1_arr, vix_nz, viy_nz, viz_nz) * t1_scale
            vt2_nz  = _trilinear_interp_vectorised(t2_arr, vix_nz, viy_nz, viz_nz) * t2_scale

            n_vox = vpts_nz.shape[0]   # nonzero original sequence voxels

            method_counts = split_spin_count(spins_per_voxel, len(spin_methods))
            active = [(m, c) for m, c in zip(spin_methods, method_counts) if c > 0]

            if active:
                extra_sub = sum(c for _, c in active)
                ex_pts = np.empty((n_vox * extra_sub, 3), dtype=np.float32)
                ex_rho = np.empty(n_vox * extra_sub, dtype=np.float32)
                ex_t1  = np.empty(n_vox * extra_sub, dtype=np.float32)
                ex_t2  = np.empty(n_vox * extra_sub, dtype=np.float32)

                # Helpers for adaptive methods (based on voxel-centre locations)
                edge_norm = None
                if any(m in {"adaptive_edge", "importance_grad", "octree_adaptive"}
                       for m, _ in active):
                    grad_z, grad_y, grad_x = np.gradient(rho_arr.astype(np.float32))
                    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
                    edge_strength = _trilinear_interp_vectorised(grad_mag, vix_nz, viy_nz, viz_nz)
                    edge_norm = robust_unit_interval(edge_strength)

                rho_norm = (robust_unit_interval(vrho_nz)
                            if any(m == "adaptive_rho" for m, _ in active) else None)

                cursor = 0
                for method_name, n_sub in active:
                    # All offsets span the FULL voxel extent (vdx × vdy × vdz) — Fix 3
                    if method_name == "random":
                        pts_method = np.empty((n_vox, n_sub, 3), dtype=np.float32)
                        pts_method[:, :, 0] = vpts_nz[:, np.newaxis, 0] + np.random.uniform(
                            -vdx / 2, vdx / 2, (n_vox, n_sub)).astype(np.float32)
                        pts_method[:, :, 1] = vpts_nz[:, np.newaxis, 1] + np.random.uniform(
                            -vdy / 2, vdy / 2, (n_vox, n_sub)).astype(np.float32)
                        pts_method[:, :, 2] = vpts_nz[:, np.newaxis, 2] + np.random.uniform(
                            -vdz / 2, vdz / 2, (n_vox, n_sub)).astype(np.float32)

                    elif method_name == "sobol_rotated":
                        offsets = generate_subvoxel_offsets(vdx, vdy, vdz, n_sub, "sobol")
                        theta = np.random.uniform(0, 2 * np.pi, n_vox).astype(np.float32)
                        cv, sv = np.cos(theta)[:, np.newaxis], np.sin(theta)[:, np.newaxis]
                        ox = offsets[np.newaxis, :, 0]; oy = offsets[np.newaxis, :, 1]
                        oz = offsets[np.newaxis, :, 2]
                        zsign = np.where(np.random.uniform(0, 1, n_vox) < 0.5,
                                         -1, 1).astype(np.float32)[:, np.newaxis]
                        pts_method = np.empty((n_vox, n_sub, 3), dtype=np.float32)
                        pts_method[:, :, 0] = vpts_nz[:, np.newaxis, 0] + cv * ox - sv * oy
                        pts_method[:, :, 1] = vpts_nz[:, np.newaxis, 1] + sv * ox + cv * oy
                        pts_method[:, :, 2] = vpts_nz[:, np.newaxis, 2] + oz * zsign

                    elif method_name == "importance_grad":
                        g = edge_norm if edge_norm is not None else np.zeros(n_vox, dtype=np.float32)
                        exp_v = (1.9 - 1.6 * g).astype(np.float32)
                        u = np.random.uniform(0, 1, (n_vox, n_sub, 3)).astype(np.float32)
                        signs = np.where(np.random.uniform(0, 1, (n_vox, n_sub, 3)) < 0.5,
                                         -1, 1).astype(np.float32)
                        mag = 0.5 * (u ** exp_v[:, np.newaxis, np.newaxis])
                        off = signs * mag
                        off[:, :, 0] *= vdx;  off[:, :, 1] *= vdy;  off[:, :, 2] *= vdz
                        pts_method = vpts_nz[:, np.newaxis, :] + off

                    elif method_name == "octree_adaptive":
                        g = edge_norm if edge_norm is not None else np.zeros(n_vox, dtype=np.float32)
                        p_fine = np.clip(0.20 + 0.65 * g, 0.05, 0.90).astype(np.float32)
                        p_mid  = np.clip(0.55 - 0.25 * g, 0.05, 0.80).astype(np.float32)
                        rv = np.random.uniform(0, 1, (n_vox, n_sub)).astype(np.float32)
                        lvl = np.where(rv < p_fine[:, np.newaxis], 0.125,
                                       np.where(rv < (p_fine + p_mid)[:, np.newaxis],
                                                0.25, 0.5)).astype(np.float32)
                        signs = np.where(np.random.uniform(0, 1, (n_vox, n_sub, 3)) < 0.5,
                                         -1, 1).astype(np.float32)
                        off = np.empty((n_vox, n_sub, 3), dtype=np.float32)
                        off[:, :, 0] = signs[:, :, 0] * lvl * vdx
                        off[:, :, 1] = signs[:, :, 1] * lvl * vdy
                        off[:, :, 2] = signs[:, :, 2] * lvl * vdz
                        jit = np.column_stack([
                            np.random.uniform(-0.05*vdx, 0.05*vdx, n_vox * n_sub),
                            np.random.uniform(-0.05*vdy, 0.05*vdy, n_vox * n_sub),
                            np.random.uniform(-0.05*vdz, 0.05*vdz, n_vox * n_sub),
                        ]).astype(np.float32).reshape(n_vox, n_sub, 3)
                        pts_method = vpts_nz[:, np.newaxis, :] + off + jit

                    elif method_name == "source_voxel_sampling":
                        centers = vpts_nz.astype(np.float32)
                        pts_method = np.empty((n_vox, n_sub, 3), dtype=np.float32)
                        pts_method[:, :, 0] = centers[:, np.newaxis, 0] + np.random.uniform(
                            -0.25*vdx, 0.25*vdx, (n_vox, n_sub)).astype(np.float32)
                        pts_method[:, :, 1] = centers[:, np.newaxis, 1] + np.random.uniform(
                            -0.25*vdy, 0.25*vdy, (n_vox, n_sub)).astype(np.float32)
                        pts_method[:, :, 2] = centers[:, np.newaxis, 2] + np.random.uniform(
                            -0.25*vdz, 0.25*vdz, (n_vox, n_sub)).astype(np.float32)
                        pts_method[:, 0, :] = centers

                    else:
                        # Generic methods: offsets span the full voxel (Fix 3)
                        offsets = generate_subvoxel_offsets(vdx, vdy, vdz, n_sub, method_name)
                        if method_name in {"stratified", "halton", "adaptive_edge", "adaptive_rho"}:
                            offsets_vox = decorrelate_offsets_per_voxel(
                                offsets, n_vox, vdx, vdy, vdz, 0.12)
                        else:
                            offsets_vox = offsets[np.newaxis, :, :]

                        if method_name == "adaptive_rho" and rho_norm is not None:
                            scale = (0.35 + 0.65 * rho_norm).astype(np.float32)
                            pts_method = (vpts_nz[:, np.newaxis, :]
                                          + scale[:, np.newaxis, np.newaxis] * offsets_vox)
                        elif method_name == "adaptive_edge" and edge_norm is not None:
                            scale = (0.30 + 0.70 * edge_norm).astype(np.float32)
                            pts_method = (vpts_nz[:, np.newaxis, :]
                                          + scale[:, np.newaxis, np.newaxis] * offsets_vox)
                        else:
                            pts_method = vpts_nz[:, np.newaxis, :] + offsets_vox

                    span = n_vox * n_sub
                    ex_pts[cursor:cursor + span] = pts_method.reshape(-1, 3).astype(np.float32,
                                                                                     copy=False)
                    ex_rho[cursor:cursor + span] = np.repeat(vrho_nz.astype(np.float32), n_sub)
                    ex_t1 [cursor:cursor + span] = np.repeat(vt1_nz.astype(np.float32),  n_sub)
                    ex_t2 [cursor:cursor + span] = np.repeat(vt2_nz.astype(np.float32),  n_sub)
                    cursor += span

                all_pts = np.concatenate([all_pts, ex_pts], axis=0)
                all_rho = np.concatenate([all_rho, ex_rho])
                all_t1  = np.concatenate([all_t1,  ex_t1])
                all_t2  = np.concatenate([all_t2,  ex_t2])

    phantom = {
        "name": f"slice_{slice_spec.index}",
        "x": (all_pts[:, 0] * 1e-3).astype(np.float32),
        "y": (all_pts[:, 1] * 1e-3).astype(np.float32),
        "z": (all_pts[:, 2] * 1e-3).astype(np.float32),
        "rho": all_rho.astype(np.float32),
        "t1": all_t1.astype(np.float32),
        "t2": all_t2.astype(np.float32),
        "t2s": (all_t2 * float(t2star_factor)).astype(np.float32),  # T2* <= T2; T2* = T2 * t2star_factor (0.5 ≈ realistic field inhomogeneity)
        "dw": np.zeros(len(all_rho), dtype=np.float32),
    }
    n_spins = len(all_rho)

    phantom["slice_geometry"] = slice_spec.to_dict()
    phantom["slice_geometry"]["slice_padding"] = slice_padding
    phantom["slice_geometry"]["z_extent_mm"] = 2 * half_extent_z
    phantom["slice_geometry"]["spin_spacing_mm"] = {
        "x": float(spin_spacing_x), "y": float(spin_spacing_y), "z": float(spin_spacing_z)}
    phantom["slice_geometry"]["lattice_pts_per_voxel"] = int(nx_per_vox * ny_per_vox * nz_per_vox)
    phantom["slice_geometry"]["spins_per_orig_voxel"]  = int(spins_per_voxel)   # extra spins per seq voxel
    phantom["slice_geometry"]["spin_methods"] = spin_methods
    phantom["slice_geometry"]["value_scaling"] = {
        "rho_divisor": float(rho_scale),
        "t1_multiplier": float(t1_scale),
        "t2_multiplier": float(t2_scale),
    }
    return phantom, n_spins


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Julia Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation_julia_batch(
    sequence_file, phantom_paths, rotation_paths, sim_output_dirs,
    b0=3.0, use_gpu=False, n_threads=4,
    rotation_json_global=None, fov_json=None, simT2s=False
):
    n = len(phantom_paths)
    batch_entries = []
    for i in range(n):
        entry = {"phantom": phantom_paths[i], "output": sim_output_dirs[i], "simT2s": simT2s}
        rot = rotation_paths[i]
        if rot and rot != rotation_json_global:
            entry["rotation"] = rot
        batch_entries.append(entry)
        print(entry)

    batch_dir = os.path.dirname(sim_output_dirs[0]) if sim_output_dirs else "."
    os.makedirs(batch_dir, exist_ok=True)
    batch_json_path = os.path.join(batch_dir, "batch_manifest.json")
    with open(batch_json_path, "w") as fh:
        json.dump(batch_entries, fh, indent=2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    batch_script = os.path.join(script_dir, "simulate_batch.jl")

    cmd = [
        "julia", f"--threads={n_threads}", "-O3",
        f"{batch_script}",
        f"--B0={b0}",
        f"--seq={sequence_file}",
        f"--batch={batch_json_path}"
    ]
    if use_gpu:
        cmd.append("--gpu")
    if rotation_json_global:
        cmd.append(f"--rotation={rotation_json_global}")
    if fov_json:
        cmd.append(f"--fov={fov_json}")

    print(f"\n[Batch Julia] {n} entries — {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    results = []
    for i in range(n):
        info_path = os.path.join(sim_output_dirs[i], "info.json")
        with open(info_path) as fh:
            info = json.load(fh)
        kspace = np.load(info["KS"])
        if hasattr(kspace, "keys"):
            kspace = kspace[list(kspace.keys())[0]]
        # ★ FIX #9: NO transpose — (Np, Nf) layout is correct
        results.append((kspace, info))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ★ FIX #6: Oversampling Removal — keep CENTRAL half
# ═══════════════════════════════════════════════════════════════════════════════

def remove_readout_oversampling_kspace(kspace: np.ndarray) -> np.ndarray:
    nP, nF_raw = kspace.shape
    nF = nF_raw // 2
    # mapVBVD flagRemoveOS semantics: 2x readout oversampling doubles the
    # ACQUIRED readout FOV (k_max and resolution are unchanged), so removal is
    # a hybrid-domain crop of the central half of the FOV.  The 1-D transform
    # must be centred, otherwise a centre-DC k-space maps the object onto the
    # array edges and the "central half" crop keeps the periphery instead.
    img_os = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(kspace, axes=1), axis=1), axes=1)
    # keep central half (not outer quarters)
    start = nF_raw // 4
    img_cropped = img_os[:, start:start + nF]
    kspace_new = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(img_cropped, axes=1), axis=1), axes=1)
    print(f"  [OS removal] readout: {nF_raw} → {nF}")
    return kspace_new


# ═══════════════════════════════════════════════════════════════════════════════
# ★ FIX #3: Reconstruction — k-space flipud BEFORE IFFT
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_from_kspace(
    kspace, expected_shape=(128, 128), oversampling=1,
    apply_hamming=True, remove_os=False, orientation=None,
    dc_correction=False,
    etl=1, echo_spacing_ms=None, te_eff_ms=None,
    t2_correction_ms=80.0,phase_reorder_override=False
):
    if remove_os:
        nP, nF_raw = kspace.shape
        if nF_raw == 2 * nP or nF_raw == 2 * expected_shape[1]:
            kspace = remove_readout_oversampling_kspace(kspace)
        else:
            print(f"  [OS removal] skipped: {kspace.shape} not 2x oversampled")

    ro_sign = -1
    flip_phase = False
    ky_trajectory = None
    needs_resort = False
    phase_reorder_indices = None
    if orientation is not None:
        ro_sign = orientation.get("detected_ro_sign", -1)
        flip_phase = orientation.get("flip_phase", False)
        ky_trajectory = orientation.get("ky_trajectory", None)
        needs_resort = orientation.get("needs_resort", False)
        phase_reorder_indices = orientation.get("phase_reorder_indices", None)
    
    # ★ K-space profile resorting for TSE with k0 > 1.
    # When phase-encode lines are acquired out of ky order (e.g. ETL=16, k0=3),
    # profiles must be sorted into their correct ky rows before IFFT.
    # We use the ky_trajectory extracted from the .seq file (via pypulseq).
    if not phase_reorder_override:
        if phase_reorder_indices is not None:
            try:
                sort_idx = np.asarray(phase_reorder_indices, dtype=np.int64)
                nP_ks = kspace.shape[0]
                if (
                    sort_idx.ndim == 1
                    and sort_idx.size == nP_ks
                    and sort_idx.min() >= 0
                    and sort_idx.max() < nP_ks
                ):
                    kspace = kspace[sort_idx, :]
                    flip_phase = False
                    print(f"  [recon] ky-sorted {nP_ks} profiles (precomputed index)")
                else:
                    print("  [recon] ky-sort skipped: invalid precomputed index")
            except Exception:
                print("  [recon] ky-sort skipped: failed to parse precomputed index")
        elif needs_resort and ky_trajectory is not None:
            ky_arr = np.array(ky_trajectory)
            nP_ks = kspace.shape[0]
            if len(ky_arr) == nP_ks:
                sort_idx = np.argsort(ky_arr, kind="mergesort")
                kspace = kspace[sort_idx, :]
                flip_phase = False  # resorting handles orientation; no additional flipud needed
                print(f"  [recon] ky-sorted {nP_ks} profiles (TSE out-of-order PE corrected)")
    else:
        print(f"  [recon] ky-sort skipped: override")
        
    # ★ DC correction: replace kx=0 column with interpolated neighbors.
    if dc_correction:
        nP_dc, nF_dc = kspace.shape
        cx = nF_dc // 2
        if 1 <= cx <= nF_dc - 2:
            kspace = kspace.copy()
            kspace[:, cx] = (kspace[:, cx - 1] + kspace[:, cx + 1]) / 2.0

    # ★ TSE echo-amplitude normalisation (smooth T2 compensation)
    # The block PE ordering in TSE means adjacent k-space rows can come from
    # different echoes with very different T2-weighted amplitudes.  Block-wise
    # normalisation (scaling each echo block to equal mean) removes the gross
    # modulation but creates sharp amplitude steps at block boundaries that
    # produce Gibbs-like ripples.
    #
    # Instead, we compute a per-row RMS amplitude profile and smooth it with
    # a wide Gaussian kernel, then divide each row by the smoothed envelope.
    # This flattens the T2 modulation without introducing discontinuities.
    if etl is not None and etl > 1:
        nP_ks = kspace.shape[0]
        n_echo = int(etl)
        n_ex = nP_ks // n_echo
        if n_ex * n_echo == nP_ks and n_ex >= 2:
            row_mag = np.sqrt(np.mean(np.abs(kspace) ** 2, axis=1))  # RMS per row

            # Smooth with Gaussian kernel (sigma = n_ex/2 rows ≈ half a block)
            from scipy.ndimage import gaussian_filter1d
            sigma = max(n_ex // 2, 4)
            smoothed = gaussian_filter1d(row_mag.astype(np.float64), sigma=sigma, mode="nearest")

            # Normalise each row: target = peak of smoothed envelope
            target = np.max(smoothed)
            if target > 0:
                correction = np.ones(nP_ks, dtype=np.float64)
                for i in range(nP_ks):
                    if smoothed[i] > 1e-12:
                        correction[i] = target / smoothed[i]
                kspace = kspace * correction[:, np.newaxis]
                max_corr = np.max(correction)
                print(f"  [TSE T2 comp] smooth normalisation (sigma={sigma}, "
                      f"max correction: {max_corr:.2f}×)")

    if apply_hamming:
        nY, nX = kspace.shape
        kspace = kspace * np.outer(np.hamming(nY), np.hamming(nX))

    # flip_phase correction in k-space BEFORE IFFT (for SE with negative-first PE)
    if flip_phase:
        kspace = np.flipud(kspace)

    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

    nP, nF_raw = img.shape
    nP_exp, nF_exp = expected_shape

    if not remove_os:
        if oversampling > 1 and nF_raw == oversampling * nF_exp:
            start_f = (nF_raw - nF_exp) // 2
            img = img[:, start_f:start_f + nF_exp]
        elif nF_raw == 2 * nP:
            nF_final = nP
            start_f = (nF_raw - nF_final) // 2
            img = img[:, start_f:start_f + nF_final]

    if nP == 2 * nP_exp:
        start_p = nP // 4
        img = img[start_p:start_p + nP_exp, :]

    # Flip readout (cols) only for negative Gx
    if ro_sign < 0:
        img = np.fliplr(img)

    return np.abs(img).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Volume Assembly
# ═══════════════════════════════════════════════════════════════════════════════


def _normalise_sitk_to_mm(img):
    """Return a copy of img with spacing/origin scaled to mm (handles meter-unit NIfTIs)."""
    import numpy as _np
    spacing = _np.array(img.GetSpacing())
    if _np.max(_np.abs(spacing)) < 0.1:
        img = sitk.Image(img)
        img.SetSpacing((spacing * 1000.0).tolist())
        img.SetOrigin((_np.array(img.GetOrigin()) * 1000.0).tolist())
    return img


def place_slice_in_body(recon, slice_spec, seq_fov_mm, slice_thickness_mm, body_ref):
    arr_3d = recon[None, :, :].astype(np.float32)
    img = sitk.GetImageFromArray(arr_3d)
    nx, ny = recon.shape[1], recon.shape[0]
    img.SetSpacing((seq_fov_mm[0] / nx, seq_fov_mm[1] / ny, slice_thickness_mm))
    R_inv = slice_spec.R_body_to_seq.T
    img.SetDirection(tuple(R_inv.flatten(order="C").tolist()))
    corner_seq = np.array([-seq_fov_mm[0] / 2, -seq_fov_mm[1] / 2, 0.0])
    corner_body = R_inv @ corner_seq + slice_spec.center_mm
    img.SetOrigin(tuple(corner_body.tolist()))
    return sitk.Resample(img, body_ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def assemble_volume(slices, body_ref):
    assembled = sitk.Image(body_ref.GetSize(), sitk.sitkFloat32)
    assembled.CopyInformation(body_ref)
    weight = sitk.Image(body_ref.GetSize(), sitk.sitkFloat32)
    weight.CopyInformation(body_ref)
    for sl in slices:
        assembled = assembled + sl
        weight = weight + sitk.Cast(sl > 0, sitk.sitkFloat32)
    return sitk.Divide(assembled, sitk.Maximum(weight, 1e-6))


# ═══════════════════════════════════════════════════════════════════════════════
# Phantom Extraction + Save Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_and_save_phantom(
    slice_spec, rho_img, t1_img, t2_img, fov_mm, resolution_mm, spin_factor,
    output_dir, num_slices, slice_thickness_mm=5.0, slice_padding=1,
    isotropic_spin_mm=True, rho_scale=100.0, t1_scale=1e-3, t2_scale=1e-3,
    use_hdf5=True, spins_per_voxel=0, spin_method=DEFAULT_SPIN_METHOD, 
    spin_axes='xy', t2star_factor=1.0,
):
    idx = slice_spec.index
    print(f"\n--- Slice {idx + 1}/{num_slices}: center="
          f"[{slice_spec.center_mm[0]:.1f}, {slice_spec.center_mm[1]:.1f}, "
          f"{slice_spec.center_mm[2]:.1f}] mm ---")

    phantom, n_spins = extract_phantom_for_slice(
        rho_img, t1_img, t2_img, slice_spec, fov_mm, resolution_mm, spin_factor,
        slice_thickness_mm=slice_thickness_mm, slice_padding=slice_padding,
        isotropic_spin_mm=isotropic_spin_mm, rho_scale=rho_scale,
        t1_scale=t1_scale, t2_scale=t2_scale,
        spins_per_voxel=spins_per_voxel, spin_method=spin_method, 
        spin_axes=spin_axes, t2star_factor=t2star_factor)

    if n_spins == 0:
        print(f"  WARNING: Slice {idx} — no spins, skipping")
        return None

    if use_hdf5 and HDF5_AVAILABLE:
        phantom_path = os.path.join(output_dir, f"phantom_{idx:03d}.h5")
        save_phantom_hdf5(phantom, phantom_path)
    else:
        phantom_path = os.path.join(output_dir, f"phantom_{idx:03d}.json")
        save_phantom_json(phantom, phantom_path)

    print(f"  Saved: {phantom_path} ({n_spins} spins)")
    return phantom_path


# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup_intermediate_outputs(output_dir):
    for pattern in ["phantom_*.h5", "phantom_*.json", "rotation.json",
                    "fov_rescale.json", "batch_manifest.json"]:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            if os.path.isfile(path):
                os.remove(path)
    for path in glob.glob(os.path.join(output_dir, "sim_*")):
        if os.path.isdir(path):
            shutil.rmtree(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    rho_path, t1_path, t2_path, sequence_file, output_dir,
    isocenter_mm, slice_normal, num_slices,
    slice_thickness_mm=None, slice_gap_mm=0.0,
    fov_mm=None, seq_fov_mm=None, matrix=None,
    spin_factor=1, b0=3.0, use_gpu=False, n_threads=4, parallel_slices=4,
    apply_hamming=True, remove_os=False, dc_correction=False,
    export_dicom=False, slice_padding=1,
    isotropic_spin_mm=True, rho_scale=100.0, t1_scale=1e-3, t2_scale=1e-3,
    use_hdf5=True, spins_per_voxel=0, spin_method=DEFAULT_SPIN_METHOD,
    flip_phase_override=None, final_nifti_path=None, debug=False,
    phase_reorder_override=False, simT2s=False, spin_axes='xy', 
    t2star_factor=1.0,
):
    os.makedirs(output_dir, exist_ok=True)
    seq_params = read_sequence_params(sequence_file)

    if seq_fov_mm is None:
        seq_fov_mm = tuple(seq_params["fov_mm"])
    if matrix is None:
        matrix = (seq_params["nP"], seq_params["nF"])
    oversampling = seq_params.get("oversampling", 1)
    orientation = seq_params.get("orientation")

    # ★ FIX #5: flip_phase override
    if flip_phase_override is not None:
        if orientation is None:
            orientation = {}
        orientation["flip_phase"] = flip_phase_override
        print(f"  Phase-flip override: {flip_phase_override}")

    # Resolve slice thickness
    seq_st = seq_params.get("slice_thickness_mm")
    if slice_thickness_mm is None:
        slice_thickness_mm = seq_st if seq_st is not None else 5.0

    if fov_mm is None:
        fov_mm = seq_fov_mm

    print(f"\nPipeline: {num_slices} slices, "
          f"FOV={seq_fov_mm[0]:.0f}x{seq_fov_mm[1]:.0f}mm, "
          f"matrix={matrix}, ST={slice_thickness_mm}mm")

    rho_img = _normalise_sitk_to_mm(sitk.ReadImage(rho_path))
    t1_img  = _normalise_sitk_to_mm(sitk.ReadImage(t1_path))
    t2_img  = (_normalise_sitk_to_mm(sitk.ReadImage(t2_path))
               if t2_path and os.path.exists(t2_path) else None)

    slice_normal = normalize(np.array(slice_normal, dtype=np.float64))
    isocenter_mm = np.array(isocenter_mm, dtype=np.float64)

    series_spec = compute_series_geometry(
        isocenter_mm, slice_normal, num_slices, slice_thickness_mm,
        slice_gap_mm, fov_mm, seq_fov_mm)

    rotation_json = os.path.join(output_dir, "rotation.json")
    with open(rotation_json, "w") as f:
        json.dump({"R_body_to_seq": series_spec.R_body_to_seq.tolist(),
                    "slice_normal": series_spec.slice_normal.tolist()}, f, indent=2)

    original_fov_mm = tuple(seq_params["fov_mm"])
    fov_json = None
    if (abs(seq_fov_mm[0] - original_fov_mm[0]) > 0.1 or
            abs(seq_fov_mm[1] - original_fov_mm[1]) > 0.1):
        fov_json = os.path.join(output_dir, "fov_rescale.json")
        with open(fov_json, "w") as f:
            json.dump({"seq_fov_m": [original_fov_mm[0] / 1000, original_fov_mm[1] / 1000],
                        "target_fov_m": [seq_fov_mm[0] / 1000, seq_fov_mm[1] / 1000]}, f)

    resolution_mm = (seq_fov_mm[0] / matrix[1], seq_fov_mm[1] / matrix[0])
    spin_methods = normalize_spin_methods(spin_method)

    parallel_slices = max(1, int(parallel_slices))

    # Phase 1: Extract phantoms (parallelized)
    print(
        f"\n{'=' * 60}\n"
        f"Phase 1: Extracting phantoms ({num_slices} slices, {parallel_slices} workers)\n"
        f"{'=' * 60}"
    )
    phantom_paths = {}
    with ThreadPoolExecutor(max_workers=parallel_slices) as executor:
        futures = {executor.submit(
            _extract_and_save_phantom,
            s, rho_img, t1_img, t2_img, fov_mm, resolution_mm, spin_factor,
            output_dir, num_slices, slice_thickness_mm=slice_thickness_mm,
            slice_padding=slice_padding, isotropic_spin_mm=isotropic_spin_mm,
            rho_scale=rho_scale, t1_scale=t1_scale, t2_scale=t2_scale,
            use_hdf5=use_hdf5, spins_per_voxel=spins_per_voxel,
            spin_method=spin_methods, t2star_factor=t2star_factor,
            spin_axes=spin_axes
        ): s.index for s in series_spec.slices}
        for future in as_completed(futures):
            idx = futures[future]
            phantom_paths[idx] = future.result()

    valid_slices = [s for s in series_spec.slices if phantom_paths[s.index] is not None]
    print(f"\n{len(valid_slices)}/{num_slices} slices have spins")

    # Phase 2: Batch Julia simulation
    print(f"\n{'=' * 60}\nPhase 2: Batch simulation ({len(valid_slices)} slices)\n{'=' * 60}")
    batch_phantom_paths = [phantom_paths[s.index] for s in valid_slices]
    batch_rotation_paths = [rotation_json] * len(valid_slices)
    batch_sim_dirs = [os.path.join(output_dir, f"sim_{s.index:03d}") for s in valid_slices]

    print(f"run_pipeline: simT2s {simT2s}")
    batch_results = run_simulation_julia_batch(
        sequence_file, batch_phantom_paths, batch_rotation_paths, batch_sim_dirs,
        b0=b0, use_gpu=use_gpu, n_threads=n_threads,
        rotation_json_global=rotation_json, fov_json=fov_json, simT2s=simT2s)

    # ★ FIX #4: Separate display flips from placement
    def _post_process(s, recon):
        # Display-only flip for PNG preview (radiological convention)
        display = np.flipud(recon)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(display, cmap="gray")
        ax.set_title(f"Slice {s.index}")
        fig.savefig(os.path.join(output_dir, f"recon_{s.index:03d}.png"), dpi=100)
        plt.close(fig)
        # Pass UNFLIPPED recon to placement
        return place_slice_in_body(recon, s, seq_fov_mm, slice_thickness_mm, rho_img)

    # Phase 3: Reconstruct & place slices (parallelized)
    print(
        f"\n{'=' * 60}\n"
        f"Phase 3: Reconstructing ({len(valid_slices)} slices, {parallel_slices} workers)\n"
        f"{'=' * 60}"
    )
    recon_images = [None] * len(valid_slices)
    recon_sitk = [None] * len(valid_slices)
    kspace_list = [None] * len(valid_slices)

    def _recon_slice_task(i, s):
        """Reconstruct and post-process a single slice (thread-safe)."""
        kspace, info = batch_results[i]
        if debug:
            ks_path = os.path.join(output_dir, f"kspace_{s.index:03d}.npy")
            np.save(ks_path, kspace)
            print(f"  [debug] k-space saved: {ks_path}  shape={kspace.shape}")
        recon = reconstruct_from_kspace(
            kspace, matrix, oversampling,
            apply_hamming=apply_hamming, remove_os=remove_os,
            dc_correction=dc_correction,
            orientation=orientation,
            etl=seq_params.get("etl", 1),
            echo_spacing_ms=seq_params.get("echo_spacing_ms"),
            te_eff_ms=seq_params.get("te_eff_ms"),
            t2_correction_ms=80.0,
            phase_reorder_override=phase_reorder_override)
        print(f"  [Slice {s.index}] reconstructed: {recon.shape}")
        return i, kspace, recon, _post_process(s, recon)

    with ThreadPoolExecutor(max_workers=parallel_slices) as executor:
        futures = {executor.submit(_recon_slice_task, i, s): i for i, s in enumerate(valid_slices)}
        for future in as_completed(futures):
            i, ks, recon, placed = future.result()
            kspace_list[i] = ks
            recon_images[i] = recon
            recon_sitk[i] = placed

    # Assemble reconstruction volume
    print("\n--- Assembling volume ---")
    volume = assemble_volume(recon_sitk, rho_img)
    volume_path = os.path.join(output_dir, "reconstruction.nii.gz")
    sitk.WriteImage(volume, volume_path)
    print(f"  Reconstruction NIfTI: {volume_path}")

    # ── K-space NIfTI (freq × phase × slices, 1 coil) ────────────────────
    # Stack: each kspace_list[i] is complex64 (Np, Nf).
    # Resulting array shape: (Nslices, Np, Nf)  →  ITK size (Nf, Np, Nslices).
    # We save real and imaginary parts as separate 3D NIfTIs with k-space
    # spacing in 1/mm units so the geometry is physically meaningful.
    print("\n--- Saving k-space NIfTIs ---")
    try:
        n_valid = len(valid_slices)
        ks_stack = np.stack(
            [kspace_list[i].astype(np.complex64) for i in range(n_valid)],
            axis=0,
        )  # (Nslices, Np, Nf)

        nslices_ks, np_ks, nf_ks = ks_stack.shape
        # K-space spacing: delta_k = 1 / FOV  (units: 1/mm)
        dkf = 1.0 / seq_fov_mm[0] if seq_fov_mm[0] > 0 else 1.0   # freq direction
        dkp = 1.0 / seq_fov_mm[1] if seq_fov_mm[1] > 0 else 1.0   # phase direction
        dks = float(series_spec.slice_thickness_mm)                  # slice step in mm (physical)

        ks_origin    = (-(nf_ks - 1) / 2.0 * dkf,
                        -(np_ks - 1) / 2.0 * dkp,
                        -(nslices_ks - 1) / 2.0 * dks)
        ks_spacing   = (dkf, dkp, dks)
        ks_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        def _save_kspace_component(arr_zyx, suffix):
            """Save a real float32 (Nslices, Np, Nf) array as NIfTI."""
            sitk_img = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
            sitk_img.SetSpacing(ks_spacing)
            sitk_img.SetOrigin(ks_origin)
            sitk_img.SetDirection(ks_direction)
            path = os.path.join(output_dir, f"kspace_{suffix}.nii.gz")
            sitk.WriteImage(sitk_img, path)
            print(f"  K-space {suffix} NIfTI: {path}  shape={arr_zyx.shape}")
            return path

        _save_kspace_component(np.real(ks_stack), "real")
        _save_kspace_component(np.imag(ks_stack), "imag")
        _save_kspace_component(np.abs(ks_stack),  "magnitude")
    except Exception as _ks_exc:
        print(f"  WARNING: could not save k-space NIfTIs: {_ks_exc}")

    with open(os.path.join(output_dir, "series_spec.json"), "w") as f:
        json.dump(series_spec.to_dict(), f, indent=2)

    if not debug and final_nifti_path is None:
        cleanup_intermediate_outputs(output_dir)

    if final_nifti_path:
        os.makedirs(os.path.dirname(os.path.abspath(final_nifti_path)) or ".", exist_ok=True)
        shutil.copy2(volume_path, final_nifti_path)
        print(f"Final volume: {final_nifti_path}")
        if not debug:
            shutil.rmtree(output_dir)

    print(f"\n{'=' * 60}\nPipeline complete! Volume: {final_nifti_path or volume_path}\n{'=' * 60}")
    return volume, series_spec


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MRI Simulation Pipeline (Final Corrected)")

    parser.add_argument("--rho", default="data/rhoh.nii.gz")
    parser.add_argument("--t1", default="data/t1.nii.gz")
    parser.add_argument("--t2", default="data/t2.nii.gz")
    parser.add_argument("--sequence", "-s",
                        default="data/sdl_pypulseq_TE10_TR4000_os2_largeCrush_xSpoil.seq")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: auto-generated under /tmp).")
    parser.add_argument("--final-nifti", default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--normal", "-n", type=float, nargs=3, required=True)
    parser.add_argument("--isocenter", type=float, nargs=3, default=None)
    parser.add_argument("--num-slices", type=int, default=5)
    parser.add_argument("--slice-thickness", type=float, default=None)
    parser.add_argument("--slice-gap", type=float, default=0.0)
    parser.add_argument("--slice-padding", type=float, default=1)
    parser.add_argument("--fov", type=float, nargs=2, default=None)
    parser.add_argument("--seq-fov", type=float, nargs=2, default=None)
    parser.add_argument("--matrix", type=int, nargs=2, default=None)
    parser.add_argument(
        "--spin-factor", type=int, default=1,
        metavar="N",
        help="Primary spin lattice density: N spins per in-plane direction per "
             "sequence voxel (spacing = resolution/N).  Default: 1 (one spin at "
             "the voxel centre).")
    parser.add_argument("--rho-scale", type=float, default=100.0)
    parser.add_argument("--t1-scale", type=float, default=1e-3)
    parser.add_argument("--t2-scale", type=float, default=1e-3)
    parser.add_argument("--legacy-z-spin-spacing", action="store_true", default=False)
    parser.add_argument(
        "--t2star-factor", type=float, default=1.0, metavar="F",
        help="T2* = T2 * F for all spins. Default 1.0 (T2*=T2, no extra dephasing). "
             "Use 0.5 to model realistic B0 inhomogeneity (T2* ≈ T2/2), "
             "which is recommended for GRE sequences.")
    parser.add_argument("--hdf5", dest="use_hdf5", action="store_true", default=True)
    parser.add_argument("--no-hdf5", dest="use_hdf5", action="store_false")
    parser.add_argument(
        "--spins-per-voxel", type=int, default=0,
        metavar="M",
        help="Additional sub-voxel spins placed at each spin-factor lattice node "
             "using --spin-method (0 = lattice only).  Total spins per lattice "
             "node = 1 + M.  "
             "Migration: old --spins-per-voxel 1 (no expansion) → new 0; "
             "old --spins-per-voxel N → new N-1 for the same total.")
    parser.add_argument("--spin-method", type=str, nargs="+", default=[DEFAULT_SPIN_METHOD])
    parser.add_argument("--b0", type=float, default=3.0)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--parallel-slices", type=int, default=4)
    parser.add_argument("--hamming", dest="hamming", action="store_true", default=True)
    parser.add_argument("--no-hamming", dest="hamming", action="store_false")
    parser.add_argument("--dc-correction", dest="dc_correction", action="store_true", default=False)
    parser.add_argument("--no-dc-correction", dest="dc_correction", action="store_false")
    parser.add_argument("--remove-oversampling", dest="remove_os", action="store_true", default=False)
    parser.add_argument("--flip-phase", dest="flip_phase", action="store_true", default=None)
    parser.add_argument("--no-flip-phase", dest="flip_phase", action="store_false")
    parser.add_argument("--dicom", action="store_true")
    parser.add_argument("--phase_reorder_override", action="store_true", default=False,
                        help="No phase reordering in recon, for GRE.")
    parser.add_argument("--sim_T2star", action="store_true", default=False, 
                        help="Include additional simulation for T2* effects")
    parser.add_argument("--spin_axes", default="xy",
                        help="Distribute the simulated spins along these axes, default xy")

    args = parser.parse_args()

    if args.output is None:
        args.output = tempfile.mkdtemp(prefix="mri_pipeline_")
        print(f"[pipeline] No --output specified; using temp dir: {args.output}")

    if args.isocenter is None:
        rho_img = sitk.ReadImage(args.rho)
        size = np.array(rho_img.GetSize())
        origin = np.array(rho_img.GetOrigin())
        spacing = np.array(rho_img.GetSpacing())
        direction = np.array(rho_img.GetDirection()).reshape(3, 3)
        isocenter = origin + direction @ ((size - 1) / 2.0 * spacing)
        print(f"Auto isocenter: {isocenter}")
    else:
        isocenter = np.array(args.isocenter)

    if args.use_hdf5 and not HDF5_AVAILABLE:
        print("WARNING: h5py not installed — falling back to JSON")
        args.use_hdf5 = False

    run_pipeline(
        rho_path=args.rho, t1_path=args.t1, t2_path=args.t2,
        sequence_file=args.sequence, output_dir=args.output,
        isocenter_mm=isocenter, slice_normal=np.array(args.normal),
        num_slices=args.num_slices, slice_thickness_mm=args.slice_thickness,
        slice_gap_mm=args.slice_gap,
        fov_mm=tuple(args.fov) if args.fov else None,
        seq_fov_mm=tuple(args.seq_fov) if args.seq_fov else None,
        matrix=tuple(args.matrix) if args.matrix else None,
        spin_factor=args.spin_factor, b0=args.b0,
        use_gpu=args.gpu, n_threads=args.threads, parallel_slices=args.parallel_slices,
        apply_hamming=args.hamming, remove_os=args.remove_os, dc_correction=args.dc_correction,
        export_dicom=args.dicom, slice_padding=args.slice_padding,
        isotropic_spin_mm=not args.legacy_z_spin_spacing,
        rho_scale=args.rho_scale, t1_scale=args.t1_scale, t2_scale=args.t2_scale,
        use_hdf5=args.use_hdf5, spins_per_voxel=args.spins_per_voxel,
        spin_method=normalize_spin_methods(args.spin_method),
        flip_phase_override=args.flip_phase, final_nifti_path=args.final_nifti,
        debug=args.debug, t2star_factor=args.t2star_factor,
        phase_reorder_override=args.phase_reorder_override,
        simT2s=args.sim_T2star,spin_axes=args.spin_axes)

if __name__ == "__main__":
    main()
