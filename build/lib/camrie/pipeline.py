#!/usr/bin/env python3
"""
camrie.pipeline - MRI simulation pipeline.

Complete workflow for simulating oblique multi-slice MRI:
1. Load body model (defines physical space)
2. Set isocenter, slice normal, and number of slices
3. Extract phantom data in sequence space coordinates
4. Run KomaMRI simulation (Julia) with rotation info
5. Reconstruct images from k-space
6. Merge slices into volume in body space
7. Optionally export DICOM

The sequence parameters (nF, nP, FOV) can be read directly from the 
Pulseq .seq file if not specified by the user.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Pulseq Sequence Reader
# =============================================================================

def read_pulseq_params(seq_path: str) -> Dict[str, Any]:
    """
    Read sequence parameters from a Pulseq .seq file.
    
    Extracts:
    - nF: Number of frequency encoding samples (from ADC)
    - nP: Number of phase encoding steps (from block count or gradient table)
    - FOV: Field of view in mm (computed from gradient moments)
    - Oversampling factor (detected when nF = 2 * nP)
    
    Parameters
    ----------
    seq_path : str
        Path to .seq file.
        
    Returns
    -------
    dict
        Sequence parameters including:
        - nF: Final freq encode samples
        - nF_raw: Raw samples before oversampling removal
        - nP: Phase encode steps
        - oversampling: Readout oversampling factor
        - fov_mm: [fov_x, fov_y] in mm
        - source: 'pypulseq' or 'fallback'
    """
    try:
        import pypulseq as pp
        seq = pp.Sequence()
        seq.read(seq_path)
        
        # Get sequence definitions
        definitions = seq.definitions
        
        # Try to get FOV from definitions
        fov_x = definitions.get("FOV", [0.3, 0.3, 0.005])
        if isinstance(fov_x, (list, tuple)):
            fov_mm = [f * 1000 for f in fov_x[:2]]  # Convert to mm
        else:
            fov_mm = [fov_x * 1000, fov_x * 1000]
        
        # Count ADC events to get readout samples
        nF = 0
        nP = 0
        adc_blocks = []
        
        for block_idx in range(1, len(seq.block_events) + 1):
            try:
                block = seq.get_block(block_idx)
                if hasattr(block, 'adc') and block.adc is not None:
                    adc_blocks.append(block_idx)
                    nF = max(nF, int(block.adc.num_samples))
            except:
                continue
        
        nP = len(adc_blocks)
        
        # Detect oversampling: if nF = 2 * nP, assume 2x readout oversampling
        nF_raw = nF
        if nF == 2 * nP:
            oversampling = 2
            nF_final = nF // 2
        else:
            oversampling = 1
            nF_final = nF
        
        duration = seq.duration()[0] if hasattr(seq.duration(), '__iter__') else seq.duration()
        
        params = {
            "nF": nF_final,
            "nF_raw": nF_raw,
            "nP": nP,
            "oversampling": oversampling,
            "fov_mm": fov_mm,
            "duration_s": duration,
            "n_blocks": len(seq.block_events),
            "source": "pypulseq",
        }
        
        return params
        
    except ImportError:
        return _read_pulseq_params_fallback(seq_path)
    except Exception as e:
        return _read_pulseq_params_fallback(seq_path)


def _read_pulseq_params_fallback(seq_path: str) -> Dict[str, Any]:
    """Fallback sequence reader - parse .seq file manually."""
    params = {
        "nF": 256,
        "nF_raw": 256,
        "nP": 128,
        "oversampling": 1,
        "fov_mm": [300.0, 300.0],
        "source": "fallback",
    }
    
    with open(seq_path, 'r') as f:
        content = f.read()
    
    # Try to extract from [DEFINITIONS] section
    in_definitions = False
    for line in content.split('\n'):
        line = line.strip()
        
        if line.startswith('[DEFINITIONS]'):
            in_definitions = True
            continue
        if line.startswith('[') and in_definitions:
            in_definitions = False
            
        if in_definitions and line:
            parts = line.split()
            if len(parts) >= 2:
                key, *values = parts
                if key.upper() == 'FOV':
                    try:
                        fov_m = [float(v) for v in values[:2]]
                        params["fov_mm"] = [f * 1000 for f in fov_m]
                    except:
                        pass
    
    # Parse ADC section for nF_raw
    in_adc = False
    nF_raw = 0
    for line in content.split('\n'):
        if line.startswith('[ADC]'):
            in_adc = True
            continue
        if line.startswith('[') and in_adc:
            in_adc = False
            
        if in_adc and line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    n_samples = int(parts[1])
                    nF_raw = max(nF_raw, n_samples)
                except:
                    pass
    
    if nF_raw > 0:
        params["nF_raw"] = nF_raw
    
    # Parse BLOCKS for nP
    in_blocks = False
    adc_count = 0
    for line in content.split('\n'):
        if line.startswith('[BLOCKS]'):
            in_blocks = True
            continue
        if line.startswith('[') and in_blocks:
            in_blocks = False
            
        if in_blocks and line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) == 8:
                try:
                    adc_event = int(parts[6])
                    if adc_event > 0:
                        adc_count += 1
                except:
                    pass
    
    if adc_count > 0:
        params["nP"] = adc_count
        
    # Detect oversampling
    if params["nF_raw"] == 2 * params["nP"]:
        params["oversampling"] = 2
        params["nF"] = params["nF_raw"] // 2
    else:
        params["nF"] = params["nF_raw"]
        params["oversampling"] = 1
    
    return params


# =============================================================================
# Geometry Utilities
# =============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return v / norm


def build_rotation_matrix(slice_normal: np.ndarray) -> np.ndarray:
    """
    Build rotation matrix from body space to sequence space.
    
    The z-axis of sequence space aligns with slice_normal.
    
    Parameters
    ----------
    slice_normal : np.ndarray (3,)
        Slice plane normal in body space.
    
    Returns
    -------
    R : np.ndarray (3, 3)
        Rotation matrix: R @ body_coords = seq_coords
    """
    z_seq = normalize(slice_normal.astype(np.float64))
    
    # Choose x_seq (readout) orthogonal to z_seq
    candidates = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    for c in candidates:
        if abs(np.dot(c, z_seq)) < 0.99:
            x_candidate = c
            break
    
    # Project onto plane perpendicular to z_seq  
    x_seq = x_candidate - np.dot(x_candidate, z_seq) * z_seq
    x_seq = normalize(x_seq)
    
    # y_seq completes right-handed system
    y_seq = np.cross(z_seq, x_seq)
    y_seq = normalize(y_seq)
    
    # R: rows are sequence axes in body coordinates
    R = np.stack([x_seq, y_seq, z_seq], axis=0)
    
    return R


@dataclass
class SliceSpec:
    """Specification for a single imaging slice."""
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
    """Specification for a multi-slice series."""
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
    isocenter_mm: np.ndarray,
    slice_normal: np.ndarray,
    num_slices: int,
    slice_thickness_mm: float,
    slice_gap_mm: float = 0.0,
    fov_mm: Tuple[float, float] = (200.0, 200.0),
    seq_fov_mm: Tuple[float, float] = (300.0, 300.0),
) -> SeriesSpec:
    """
    Compute geometry for a multi-slice series centered on isocenter.
    
    Parameters
    ----------
    isocenter_mm : np.ndarray (3,)
        Center of the series in body space (mm).
    slice_normal : np.ndarray (3,)
        Slice plane normal (will be normalized).
    num_slices : int
        Number of slices.
    slice_thickness_mm : float
        Slice thickness.
    slice_gap_mm : float
        Gap between slices.
    fov_mm : tuple
        In-plane FOV for phantom extraction.
    seq_fov_mm : tuple
        Sequence FOV for reconstruction.
    
    Returns
    -------
    SeriesSpec
        Series geometry specification.
    """
    slice_normal = normalize(np.array(slice_normal, dtype=np.float64))
    isocenter_mm = np.array(isocenter_mm, dtype=np.float64)
    
    R = build_rotation_matrix(slice_normal)
    
    spacing = slice_thickness_mm + slice_gap_mm
    slices = []
    
    for i in range(num_slices):
        offset = (i - (num_slices - 1) / 2.0) * spacing
        center_mm = isocenter_mm + offset * slice_normal
        pos_along_normal = float(np.dot(center_mm, slice_normal))
        
        slices.append(SliceSpec(
            normal=slice_normal,
            center_mm=center_mm,
            position_along_normal=pos_along_normal,
            R_body_to_seq=R,
            index=i,
        ))
    
    return SeriesSpec(
        isocenter_mm=isocenter_mm,
        slice_normal=slice_normal,
        R_body_to_seq=R,
        slices=slices,
        fov_mm=fov_mm,
        seq_fov_mm=seq_fov_mm,
        slice_thickness_mm=slice_thickness_mm,
    )


# =============================================================================
# Phantom Extraction
# =============================================================================

def extract_phantom_for_slice(
    rho_img: sitk.Image,
    t1_img: sitk.Image,
    t2_img: Optional[sitk.Image],
    slice_spec: SliceSpec,
    fov_mm: Tuple[float, float],
    resolution_mm: Tuple[float, float],
    spin_factor: int = 1,
) -> Tuple[Dict[str, Any], int]:
    """
    Extract phantom data for a slice in SEQUENCE SPACE coordinates.
    
    The phantom x, y coordinates are in the imaging plane (readout, phase).
    z = 0 (on-slice).
    
    Parameters
    ----------
    rho_img : sitk.Image
        Proton density image.
    t1_img : sitk.Image
        T1 map image.
    t2_img : sitk.Image or None
        T2 map image.
    slice_spec : SliceSpec
        Slice geometry specification.
    fov_mm : tuple
        Field of view (x, y) in mm.
    resolution_mm : tuple
        Base image resolution (readout_res, phase_res).
    spin_factor : int
        Spin density multiplier (1=native resolution, 2=2x spins per voxel).
    
    Returns
    -------
    phantom : dict
        Phantom data with coordinates in sequence space.
    n_spins : int
        Number of spins extracted.
    """
    R = slice_spec.R_body_to_seq
    center_mm = slice_spec.center_mm
    
    # Spin spacing = image resolution / spin_factor
    spin_spacing_x = resolution_mm[0] / spin_factor
    spin_spacing_y = resolution_mm[1] / spin_factor
    
    # Grid in sequence space
    nx = int(np.ceil(fov_mm[0] / spin_spacing_x))
    ny = int(np.ceil(fov_mm[1] / spin_spacing_y))
    
    x_seq = (np.arange(nx) - (nx - 1) / 2.0) * spin_spacing_x
    y_seq = (np.arange(ny) - (ny - 1) / 2.0) * spin_spacing_y
    
    phantom = {
        "name": f"slice_{slice_spec.index}",
        "x": [], "y": [], "z": [],
        "rho": [], "t1": [], "t2": [], "t2s": [], "dw": [],
    }
    
    # Pre-load arrays
    rho_arr = sitk.GetArrayFromImage(rho_img)
    t1_arr = sitk.GetArrayFromImage(t1_img)
    t2_arr = sitk.GetArrayFromImage(t2_img) if t2_img else np.zeros_like(rho_arr)
    size = rho_img.GetSize()
    
    for yj in y_seq:
        for xi in x_seq:
            p_seq = np.array([xi, yj, 0.0])
            p_body = R.T @ p_seq + center_mm
            
            try:
                idx = rho_img.TransformPhysicalPointToContinuousIndex(tuple(p_body))
                
                if any(idx[d] < 0 or idx[d] >= size[d] - 1 for d in range(3)):
                    continue
                
                # Trilinear interpolation
                ix, iy, iz = idx
                ix0, iy0, iz0 = int(np.floor(ix)), int(np.floor(iy)), int(np.floor(iz))
                ix1 = min(ix0 + 1, size[0] - 1)
                iy1 = min(iy0 + 1, size[1] - 1)
                iz1 = min(iz0 + 1, size[2] - 1)
                fx, fy, fz = ix - ix0, iy - iy0, iz - iz0
                
                def interp3d(arr):
                    c000 = arr[iz0, iy0, ix0]
                    c001 = arr[iz0, iy0, ix1]
                    c010 = arr[iz0, iy1, ix0]
                    c011 = arr[iz0, iy1, ix1]
                    c100 = arr[iz1, iy0, ix0]
                    c101 = arr[iz1, iy0, ix1]
                    c110 = arr[iz1, iy1, ix0]
                    c111 = arr[iz1, iy1, ix1]
                    
                    c00 = c000 * (1 - fx) + c001 * fx
                    c01 = c010 * (1 - fx) + c011 * fx
                    c10 = c100 * (1 - fx) + c101 * fx
                    c11 = c110 * (1 - fx) + c111 * fx
                    
                    c0 = c00 * (1 - fy) + c01 * fy
                    c1 = c10 * (1 - fy) + c11 * fy
                    
                    return c0 * (1 - fz) + c1 * fz
                
                rho_v = interp3d(rho_arr)
                
                if rho_v > 0:
                    t1_v = interp3d(t1_arr)
                    t2_v = interp3d(t2_arr)
                    
                    # Store in sequence space (meters for KomaMRI)
                    phantom["x"].append(float(xi * 1e-3))
                    phantom["y"].append(float(yj * 1e-3))
                    phantom["z"].append(0.0)
                    phantom["rho"].append(float(rho_v))
                    phantom["t1"].append(float(t1_v * 1e-3))  # ms -> s
                    phantom["t2"].append(float(t2_v * 1e-3))
                    phantom["t2s"].append(0.0)
                    phantom["dw"].append(0.0)
                    
            except Exception:
                continue
    
    n_spins = len(phantom["x"])
    phantom["slice_geometry"] = slice_spec.to_dict()
    
    return phantom, n_spins


# =============================================================================
# K-space Reconstruction
# =============================================================================

def reconstruct_image_2d(
    kspace: np.ndarray,
    expected_shape: Tuple[int, int] = (128, 128),
    oversampling: int = 2,
) -> np.ndarray:
    """
    Reconstruct image from 2D k-space data.
    
    Parameters
    ----------
    kspace : np.ndarray
        K-space data (nP x nF_raw).
    expected_shape : tuple
        Expected final image shape (nP, nF) after oversampling removal.
    oversampling : int
        Readout oversampling factor (typically 2). 
        When nF_raw = oversampling * nF, crop center after FFT.
    
    Returns
    -------
    np.ndarray
        Reconstructed magnitude image.
    """
    # 2D IFFT with proper shifts
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
    
    nP, nF_raw = img.shape
    nP_exp, nF_exp = expected_shape
    
    # Handle readout oversampling
    if oversampling > 1 and nF_raw == oversampling * nF_exp:
        start_f = (nF_raw - nF_exp) // 2
        img = img[:, start_f:start_f + nF_exp]
    elif nF_raw == 2 * nP:  # Auto-detect
        nF_final = nP
        start_f = (nF_raw - nF_final) // 2
        img = img[:, start_f:start_f + nF_final]
    
    # Handle phase oversampling
    if nP == 2 * nP_exp:
        start_p = nP // 4
        img = img[start_p:start_p + nP_exp, :]
    
    # Standard orientation adjustment
    img = np.fliplr(np.rot90(img))
    
    return np.abs(img).astype(np.float32)


# =============================================================================
# Julia Simulation
# =============================================================================

def _get_julia_script_path() -> str:
    """Get path to Julia simulation script."""
    # Try package data first
    try:
        import importlib.resources as pkg_resources
        from camrie import julia
        with pkg_resources.as_file(pkg_resources.files(julia) / "simulate.jl") as p:
            return str(p)
    except Exception:
        pass
    
    # Fallback to relative path
    module_dir = Path(__file__).parent
    return str(module_dir / "julia" / "simulate.jl")


def run_simulation_julia(
    phantom_json: str,
    sequence_file: str,
    output_dir: str,
    rotation_json: Optional[str] = None,
    b0: float = 3.0,
    use_gpu: bool = False,
    n_threads: int = 4,
) -> Tuple[np.ndarray, Dict]:
    """
    Run KomaMRI simulation via Julia subprocess.
    
    Parameters
    ----------
    phantom_json : str
        Path to phantom JSON file.
    sequence_file : str
        Path to Pulseq .seq file.
    output_dir : str
        Output directory.
    rotation_json : str, optional
        Path to rotation matrix JSON.
    b0 : float
        Main magnetic field strength (T).
    use_gpu : bool
        Use GPU acceleration.
    n_threads : int
        Number of CPU threads.
        
    Returns
    -------
    kspace : np.ndarray
        K-space data.
    info : dict
        Simulation metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    julia_script = _get_julia_script_path()
    
    cmd = [
        "julia", "--threads=auto", julia_script,
        str(b0),
        sequence_file,
        phantom_json,
        output_dir,
        "true" if use_gpu else "false",
        str(n_threads),
    ]
    
    if rotation_json:
        cmd.append(rotation_json)
    
    result = subprocess.run(cmd, check=True, capture_output=False)
    
    # Load results
    info_path = os.path.join(output_dir, "info.json")
    with open(info_path) as f:
        info = json.load(f)
    
    kspace = np.load(info["KS"])
    if hasattr(kspace, 'keys'):
        kspace = kspace[list(kspace.keys())[0]]
    kspace = np.transpose(kspace)
    
    return kspace, info


# =============================================================================
# Volume Assembly
# =============================================================================

def place_slice_in_body(
    recon: np.ndarray,
    slice_spec: SliceSpec,
    seq_fov_mm: Tuple[float, float],
    slice_thickness_mm: float,
    body_ref: sitk.Image,
) -> sitk.Image:
    """
    Place a reconstructed 2D image into body space.
    
    The reconstruction is in sequence space. We transform it to body space
    using the inverse of R_body_to_seq.
    """
    arr_3d = recon[None, :, :].astype(np.float32)
    img = sitk.GetImageFromArray(arr_3d)
    
    nx, ny = recon.shape[1], recon.shape[0]
    spacing = (
        seq_fov_mm[0] / nx,
        seq_fov_mm[1] / ny,
        slice_thickness_mm,
    )
    img.SetSpacing(spacing)
    
    R = slice_spec.R_body_to_seq
    R_inv = R.T
    
    direction = tuple(R_inv.flatten(order='C').tolist())
    img.SetDirection(direction)
    
    corner_seq = np.array([
        -seq_fov_mm[0] / 2.0,
        -seq_fov_mm[1] / 2.0,
        0.0,
    ])
    corner_body = R_inv @ corner_seq + slice_spec.center_mm
    img.SetOrigin(tuple(corner_body.tolist()))
    
    resampled = sitk.Resample(
        img,
        body_ref,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    
    return resampled


def assemble_volume(
    slices: List[sitk.Image],
    body_ref: sitk.Image,
) -> sitk.Image:
    """Combine multiple slices into a single volume."""
    assembled = sitk.Image(body_ref.GetSize(), sitk.sitkFloat32)
    assembled.CopyInformation(body_ref)
    
    weight = sitk.Image(body_ref.GetSize(), sitk.sitkFloat32)
    weight.CopyInformation(body_ref)
    
    for sl in slices:
        assembled = assembled + sl
        weight = weight + sitk.Cast(sl > 0, sitk.sitkFloat32)
    
    assembled = sitk.Divide(assembled, sitk.Maximum(weight, 1e-6))
    
    return assembled


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(
    config,  # SimulationConfig
    use_gpu: bool = False,
    save_previews: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Complete MRI simulation pipeline.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    use_gpu : bool
        Use GPU acceleration if available.
    save_previews : bool
        Save PNG previews of reconstructed slices.
    
    Returns
    -------
    recon_images : list of np.ndarray
        Reconstructed 2D images.
    kspace_list : list of np.ndarray
        Raw k-space data per slice.
    """
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Read sequence parameters
    seq_params = read_pulseq_params(config.sequence_path)
    seq_fov_mm = tuple(config.seq_fov_mm or seq_params["fov_mm"])
    matrix = (seq_params["nP"], seq_params["nF"])
    oversampling = seq_params.get("oversampling", 1)
    fov_mm = tuple(config.fov_mm or seq_fov_mm)
    
    # Load body model
    rho_img = sitk.ReadImage(config.rho_path)
    t1_img = sitk.ReadImage(config.t1_path)
    t2_img = sitk.ReadImage(config.t2_path) if config.t2_path and os.path.exists(config.t2_path) else None
    
    # Normalize parameters
    slice_normal = normalize(np.array(config.slice_normal, dtype=np.float64))
    isocenter_mm = np.array(config.isocenter_mm, dtype=np.float64)
    
    # Compute series geometry
    series_spec = compute_series_geometry(
        isocenter_mm=isocenter_mm,
        slice_normal=slice_normal,
        num_slices=config.num_slices,
        slice_thickness_mm=config.slice_thickness_mm,
        slice_gap_mm=config.slice_gap_mm,
        fov_mm=fov_mm,
        seq_fov_mm=seq_fov_mm,
    )
    
    # Save rotation matrix for Julia
    rotation_json = os.path.join(output_dir, "rotation.json")
    with open(rotation_json, "w") as f:
        json.dump({
            "R_body_to_seq": series_spec.R_body_to_seq.tolist(),
            "slice_normal": series_spec.slice_normal.tolist(),
        }, f, indent=2)
    
    # Compute image resolution
    resolution_mm = (seq_fov_mm[0] / matrix[1], seq_fov_mm[1] / matrix[0])
    
    # Process each slice
    recon_images = []
    kspace_list = []
    
    for slice_spec in series_spec.slices:
        # Extract phantom
        phantom, n_spins = extract_phantom_for_slice(
            rho_img, t1_img, t2_img,
            slice_spec, fov_mm, resolution_mm, config.spin_factor,
        )
        
        if n_spins == 0:
            continue
        
        # Save phantom
        phantom_path = os.path.join(output_dir, f"phantom_{slice_spec.index:03d}.json")
        with open(phantom_path, "w") as f:
            json.dump(phantom, f)
        
        # Run Julia simulation
        sim_dir = os.path.join(output_dir, f"sim_{slice_spec.index:03d}")
        kspace, info = run_simulation_julia(
            phantom_path,
            config.sequence_path,
            sim_dir,
            rotation_json=rotation_json,
            b0=config.b0,
            use_gpu=use_gpu,
        )
        kspace_list.append(kspace)
        
        # Reconstruct
        recon = reconstruct_image_2d(kspace, matrix, oversampling)
        recon_images.append(recon)
        
        # Save preview
        if save_previews and HAS_MATPLOTLIB:
            plt.figure(figsize=(8, 8))
            plt.imshow(recon, cmap="gray")
            plt.title(f"Slice {slice_spec.index}")
            plt.colorbar()
            plt.savefig(os.path.join(output_dir, f"recon_{slice_spec.index:03d}.png"), dpi=100)
            plt.close()
    
    # Save series spec
    with open(os.path.join(output_dir, "series_spec.json"), "w") as f:
        json.dump(series_spec.to_dict(), f, indent=2)
    
    return recon_images, kspace_list
