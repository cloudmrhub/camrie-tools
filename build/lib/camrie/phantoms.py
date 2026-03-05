"""
camrie.phantoms - Phantom loading and generation utilities.

Provides:
- load_nifti_maps: Load multi-map phantoms from NIfTI files
- create_shepp_logan_phantom: Generate Shepp-Logan phantom
- get_builtin_phantom: Access built-in phantoms (elephant, brain)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


# =============================================================================
# NIfTI Loading
# =============================================================================

def load_nifti_maps(
    rho_path: str,
    t1_path: str,
    t2_path: str = None,
    t2s_path: str = None,
) -> Dict[str, np.ndarray]:
    """
    Load MRI parameter maps from NIfTI files.
    
    Parameters
    ----------
    rho_path : str
        Path to proton density NIfTI file.
    t1_path : str
        Path to T1 map NIfTI file (in seconds).
    t2_path : str, optional
        Path to T2 map NIfTI file (in seconds).
        If None, uses T1 * 1.1 as approximation.
    t2s_path : str, optional
        Path to T2* map NIfTI file (in seconds).
        If None, uses T2 * 0.8 as approximation.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'rho': 3D proton density array
        - 't1': 3D T1 array (seconds)
        - 't2': 3D T2 array (seconds)
        - 't2s': 3D T2* array (seconds)
        - 'spacing': voxel spacing in mm
        - 'origin': physical origin in mm
        - 'direction': direction cosines (3x3)
        - 'sitk_image': SimpleITK image object (rho)
        
    Raises
    ------
    ImportError
        If SimpleITK is not installed.
    FileNotFoundError
        If any required file is missing.
    """
    if sitk is None:
        raise ImportError(
            "SimpleITK is required for NIfTI loading. "
            "Install with: pip install SimpleITK"
        )
    
    # Load rho
    if not os.path.exists(rho_path):
        raise FileNotFoundError(f"Rho file not found: {rho_path}")
    rho_img = sitk.ReadImage(rho_path)
    rho = sitk.GetArrayFromImage(rho_img).astype(np.float32)
    
    # Load T1
    if not os.path.exists(t1_path):
        raise FileNotFoundError(f"T1 file not found: {t1_path}")
    t1_img = sitk.ReadImage(t1_path)
    t1 = sitk.GetArrayFromImage(t1_img).astype(np.float32)
    
    # Load or derive T2
    if t2_path and os.path.exists(t2_path):
        t2_img = sitk.ReadImage(t2_path)
        t2 = sitk.GetArrayFromImage(t2_img).astype(np.float32)
    else:
        t2 = t1 * 1.1  # Rough approximation
    
    # Load or derive T2*
    if t2s_path and os.path.exists(t2s_path):
        t2s_img = sitk.ReadImage(t2s_path)
        t2s = sitk.GetArrayFromImage(t2s_img).astype(np.float32)
    else:
        t2s = t2 * 0.8  # Rough approximation
    
    return {
        'rho': rho,
        't1': t1,
        't2': t2,
        't2s': t2s,
        'spacing': rho_img.GetSpacing(),
        'origin': rho_img.GetOrigin(),
        'direction': np.array(rho_img.GetDirection()).reshape(3, 3),
        'sitk_image': rho_img,
    }


# =============================================================================
# Shepp-Logan Phantom
# =============================================================================

def create_shepp_logan_phantom(
    size: int = 256,
    dtype: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """
    Create a 3D modified Shepp-Logan phantom with realistic MRI parameters.
    
    Parameters
    ----------
    size : int
        Size of the cubic phantom (size x size x size). Default 256.
    dtype : np.dtype
        Data type for arrays. Default float32.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'rho': 3D proton density array (0-1)
        - 't1': 3D T1 array (seconds)
        - 't2': 3D T2 array (seconds)
        - 'spacing': voxel spacing in mm (1, 1, 1)
        - 'origin': physical origin in mm
        
    Notes
    -----
    The phantom contains ellipsoids with different tissue properties:
    - White matter: T1=0.8s, T2=0.08s, ρ=0.8
    - Gray matter: T1=1.0s, T2=0.10s, ρ=0.9
    - CSF: T1=4.0s, T2=0.5s, ρ=1.0
    - Fat: T1=0.35s, T2=0.07s, ρ=0.6
    """
    # Initialize arrays
    rho = np.zeros((size, size, size), dtype=dtype)
    t1 = np.zeros((size, size, size), dtype=dtype)
    t2 = np.zeros((size, size, size), dtype=dtype)
    
    # Create coordinate grids
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Define ellipsoids: (a, b, c, x0, y0, z0, rho_val, t1_val, t2_val)
    # a, b, c are semi-axes, (x0, y0, z0) is center
    ellipsoids = [
        # Main brain ellipsoid (gray matter)
        (0.69, 0.92, 0.81, 0, 0, 0, 0.9, 1.0, 0.10),
        # Inner darker region (white matter)
        (0.66, 0.87, 0.77, 0, -0.01, 0, 0.8, 0.8, 0.08),
        # Ventricles (CSF)
        (0.11, 0.31, 0.22, 0.22, 0, 0, 1.0, 4.0, 0.5),
        (0.11, 0.31, 0.22, -0.22, 0, 0, 1.0, 4.0, 0.5),
        # Small lesions
        (0.21, 0.25, 0.25, 0, 0.35, -0.15, 0.5, 0.5, 0.05),
        (0.046, 0.046, 0.05, 0, 0.1, 0.25, 0.6, 0.35, 0.07),  # Fat
        (0.046, 0.046, 0.05, 0, -0.1, 0.25, 0.6, 0.35, 0.07),  # Fat
        (0.023, 0.023, 0.023, -0.06, -0.65, 0, 0.95, 1.2, 0.12),
        (0.023, 0.023, 0.023, 0.06, -0.65, 0, 0.95, 1.2, 0.12),
    ]
    
    for (a, b, c, x0, y0, z0, rho_val, t1_val, t2_val) in ellipsoids:
        # Check if point is inside ellipsoid
        inside = ((X - x0) / a) ** 2 + ((Y - y0) / b) ** 2 + ((Z - z0) / c) ** 2 <= 1
        rho[inside] = rho_val
        t1[inside] = t1_val
        t2[inside] = t2_val
    
    # Spacing: 1mm isotropic, centered
    spacing = (1.0, 1.0, 1.0)
    origin = (-size / 2, -size / 2, -size / 2)
    
    return {
        'rho': rho,
        't1': t1,
        't2': t2,
        'spacing': spacing,
        'origin': origin,
    }


def save_phantom_nifti(
    phantom: Dict[str, np.ndarray],
    output_dir: Union[str, Path],
    name: str = "phantom",
) -> Dict[str, str]:
    """
    Save phantom arrays to NIfTI files.
    
    Parameters
    ----------
    phantom : dict
        Phantom dictionary from create_shepp_logan_phantom or similar.
    output_dir : str or Path
        Output directory.
    name : str
        Base name for output files. Default "phantom".
        
    Returns
    -------
    dict
        Dictionary mapping property names to file paths.
    """
    if sitk is None:
        raise ImportError("SimpleITK required for saving NIfTI files")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    spacing = phantom.get('spacing', (1.0, 1.0, 1.0))
    origin = phantom.get('origin', (0, 0, 0))
    
    for prop in ['rho', 't1', 't2', 't2s']:
        if prop in phantom:
            img = sitk.GetImageFromArray(phantom[prop])
            img.SetSpacing(spacing)
            img.SetOrigin(origin)
            
            path = output_dir / f"{name}_{prop}.nii.gz"
            sitk.WriteImage(img, str(path))
            paths[prop] = str(path)
    
    return paths


# =============================================================================
# Built-in Phantoms
# =============================================================================

def get_builtin_phantom(name: str = "elephant") -> Dict[str, str]:
    """
    Get paths to built-in phantom files.
    
    Parameters
    ----------
    name : str
        Phantom name. Available: 'elephant', 'shepp_logan'.
        Default 'elephant'.
        
    Returns
    -------
    dict
        Dictionary with paths:
        - 'rho_path': Path to proton density file
        - 't1_path': Path to T1 map file
        - 't2_path': Path to T2 map file (may be None)
        
    Raises
    ------
    ValueError
        If phantom name is unknown.
    """
    try:
        import importlib.resources as pkg_resources
        from camrie import data
    except ImportError:
        # Fallback for older Python
        data_dir = Path(__file__).parent / "data"
        pkg_resources = None
    
    if name == "elephant":
        if pkg_resources:
            try:
                # Python 3.9+ style
                rho_path = pkg_resources.files(data) / "elephant_rho.nii.gz"
                t1_path = pkg_resources.files(data) / "elephant_t1.nii.gz"
                t2_path = pkg_resources.files(data) / "elephant_t2.nii.gz"
                
                return {
                    'rho_path': str(rho_path),
                    't1_path': str(t1_path),
                    't2_path': str(t2_path),
                    'name': 'elephant',
                }
            except (TypeError, AttributeError):
                pass
        
        # Fallback to direct path
        data_dir = Path(__file__).parent / "data"
        return {
            'rho_path': str(data_dir / "elephant_rho.nii.gz"),
            't1_path': str(data_dir / "elephant_t1.nii.gz"),
            't2_path': str(data_dir / "elephant_t2.nii.gz"),
            'name': 'elephant',
        }
    
    elif name == "shepp_logan":
        # Generate Shepp-Logan on demand
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="camrie_shepp_")
        phantom = create_shepp_logan_phantom(size=128)
        paths = save_phantom_nifti(phantom, temp_dir, "shepp_logan")
        
        return {
            'rho_path': paths.get('rho'),
            't1_path': paths.get('t1'),
            't2_path': paths.get('t2'),
            'name': 'shepp_logan',
            'temp_dir': temp_dir,
        }
    
    else:
        available = ['elephant', 'shepp_logan']
        raise ValueError(
            f"Unknown phantom '{name}'. Available: {available}"
        )


def list_builtin_phantoms() -> list:
    """List available built-in phantoms."""
    return ['elephant', 'shepp_logan']


# =============================================================================
# Phantom Extraction
# =============================================================================

def extract_slab(
    phantom: Dict[str, np.ndarray],
    center_mm: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    thickness_mm: float,
    fov_mm: Tuple[float, float],
    matrix_size: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """
    Extract a 2D slab from a 3D phantom at arbitrary orientation.
    
    Parameters
    ----------
    phantom : dict
        3D phantom data with 'rho', 't1', 't2' arrays.
    center_mm : tuple
        (x, y, z) center of slab in mm.
    normal : tuple
        (nx, ny, nz) slab normal direction.
    thickness_mm : float
        Slab thickness in mm.
    fov_mm : tuple
        (fov_x, fov_y) field of view in mm.
    matrix_size : tuple
        (nx, ny) output matrix size.
        
    Returns
    -------
    dict
        Extracted slab with same keys as input phantom.
    """
    from scipy.ndimage import map_coordinates
    
    # Normalize the normal vector
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    
    # Build orthonormal basis for the slab
    # Find a vector not parallel to normal
    if abs(normal[2]) < 0.9:
        ref = np.array([0, 0, 1])
    else:
        ref = np.array([1, 0, 0])
    
    # In-plane vectors
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Create sampling grid
    fx, fy = fov_mm
    nx, ny = matrix_size
    
    x = np.linspace(-fx/2, fx/2, nx)
    y = np.linspace(-fy/2, fy/2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Convert to 3D coordinates
    center = np.array(center_mm)
    points = (
        center[np.newaxis, np.newaxis, :] +
        X[:, :, np.newaxis] * u +
        Y[:, :, np.newaxis] * v
    )
    
    # Convert to voxel coordinates
    spacing = np.array(phantom.get('spacing', (1, 1, 1)))
    origin = np.array(phantom.get('origin', (0, 0, 0)))
    
    voxel_coords = (points - origin) / spacing
    
    # Rearrange for map_coordinates (3, nx, ny)
    coords = np.stack([
        voxel_coords[:, :, 2],  # z
        voxel_coords[:, :, 1],  # y
        voxel_coords[:, :, 0],  # x
    ])
    
    # Sample each property
    result = {}
    for key in ['rho', 't1', 't2', 't2s']:
        if key in phantom:
            result[key] = map_coordinates(
                phantom[key], coords,
                order=1, mode='constant', cval=0.0
            )
    
    return result
