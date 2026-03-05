"""
camrie.config - Configuration dataclasses and utilities.

Provides:
- SimulationConfig: Main configuration for simulations
- GEOMETRY_PRESETS: Common slice orientations
- ConfigManager: Load/save configuration files
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


# =============================================================================
# Geometry Presets
# =============================================================================

GEOMETRY_PRESETS = {
    "AXIAL": [0.0, 0.0, 1.0],      # Superior-Inferior (transverse)
    "CORONAL": [0.0, 1.0, 0.0],    # Anterior-Posterior
    "SAGITTAL": [1.0, 0.0, 0.0],   # Left-Right
    "OBLIQUE_45_AP": [0.0, 0.707, 0.707],   # 45° off axial toward coronal
    "OBLIQUE_45_LR": [0.707, 0.0, 0.707],   # 45° off axial toward sagittal
}


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    """
    Configuration for MRI simulation.
    
    Attributes
    ----------
    rho_path : str
        Path to proton density NIfTI file.
    t1_path : str
        Path to T1 map NIfTI file.
    t2_path : str
        Path to T2 map NIfTI file.
    sequence_path : str
        Path to Pulseq .seq file.
    isocenter_mm : List[float]
        [x, y, z] isocenter position in mm (physical coordinates).
    slice_normal : List[float]
        [nx, ny, nz] unit vector defining slice normal direction.
    num_slices : int
        Number of slices to simulate.
    slice_thickness_mm : float
        Thickness of each slice in mm.
    slice_gap_mm : float
        Gap between consecutive slices in mm.
    fov_mm : List[float]
        [fov_x, fov_y] field of view in mm.
    seq_fov_mm : List[float]
        FOV from sequence file in mm.
    b0 : float
        Main magnetic field strength in Tesla. Default 3.0.
    spin_factor : int
        Spin density multiplier (1-4). Higher values create more
        spin isochromats per voxel for better accuracy. Default 1.
    output_dir : str
        Directory for output files.
    """
    
    # Body model paths
    rho_path: str
    t1_path: str
    t2_path: str
    
    # Sequence
    sequence_path: str
    
    # Geometry
    isocenter_mm: List[float]
    slice_normal: List[float]
    num_slices: int
    slice_thickness_mm: float
    slice_gap_mm: float
    
    # FOV
    fov_mm: List[float]
    seq_fov_mm: List[float]
    
    # Simulation parameters
    b0: float = 3.0
    spin_factor: int = 1
    
    # Output
    output_dir: str = "camrie_output"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Normalize slice_normal
        norm = np.linalg.norm(self.slice_normal)
        if norm > 0:
            self.slice_normal = [x / norm for x in self.slice_normal]
        
        # Ensure spin_factor is valid
        if self.spin_factor < 1:
            self.spin_factor = 1
        elif self.spin_factor > 4:
            self.spin_factor = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to: {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "SimulationConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_preset(
        cls,
        preset: str,
        rho_path: str,
        t1_path: str,
        t2_path: str,
        sequence_path: str,
        isocenter_mm: List[float] = None,
        **kwargs,
    ) -> "SimulationConfig":
        """
        Create configuration from a geometry preset.
        
        Parameters
        ----------
        preset : str
            One of: 'AXIAL', 'CORONAL', 'SAGITTAL', 'OBLIQUE_45_AP', 'OBLIQUE_45_LR'
        rho_path : str
            Path to proton density map.
        t1_path : str
            Path to T1 map.
        t2_path : str
            Path to T2 map.
        sequence_path : str
            Path to Pulseq sequence.
        isocenter_mm : List[float], optional
            Isocenter position. Auto-computed from image if None.
        **kwargs
            Additional SimulationConfig parameters.
            
        Returns
        -------
        SimulationConfig
            Configuration with preset geometry.
        """
        import SimpleITK as sitk
        
        if preset not in GEOMETRY_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {list(GEOMETRY_PRESETS.keys())}"
            )
        
        slice_normal = GEOMETRY_PRESETS[preset]
        
        # Auto-detect isocenter from image
        if isocenter_mm is None:
            img = sitk.ReadImage(rho_path)
            size = img.GetSize()
            center_index = [s // 2 for s in size]
            isocenter_mm = list(img.TransformIndexToPhysicalPoint(center_index))
        
        # Default values
        defaults = {
            "num_slices": 10,
            "slice_thickness_mm": 5.0,
            "slice_gap_mm": 1.0,
            "fov_mm": [300.0, 300.0],
            "seq_fov_mm": [300.0, 300.0],
            "b0": 3.0,
            "spin_factor": 1,
            "output_dir": "camrie_output",
        }
        defaults.update(kwargs)
        
        return cls(
            rho_path=rho_path,
            t1_path=t1_path,
            t2_path=t2_path,
            sequence_path=sequence_path,
            isocenter_mm=isocenter_mm,
            slice_normal=slice_normal,
            **defaults,
        )
    
    def get_rotation_matrix(self) -> np.ndarray:
        """
        Compute 3x3 rotation matrix from body to sequence space.
        
        The rotation transforms body coordinates to sequence coordinates
        where the slice normal aligns with the Z-axis.
        
        Returns
        -------
        np.ndarray
            3x3 rotation matrix R such that R @ slice_normal = [0, 0, 1]
        """
        from camrie.pipeline import build_rotation_matrix
        return build_rotation_matrix(self.slice_normal)
    
    def get_slice_positions(self) -> List[np.ndarray]:
        """
        Compute physical positions of all slice centers.
        
        Returns
        -------
        List[np.ndarray]
            List of [x, y, z] positions for each slice center.
        """
        isocenter = np.array(self.isocenter_mm)
        normal = np.array(self.slice_normal)
        
        # Total span
        pitch = self.slice_thickness_mm + self.slice_gap_mm
        total_thickness = (self.num_slices - 1) * pitch
        
        # Start from one end
        start = isocenter - (total_thickness / 2) * normal
        
        positions = []
        for i in range(self.num_slices):
            pos = start + i * pitch * normal
            positions.append(pos)
        
        return positions
    
    def summary(self) -> str:
        """Return human-readable summary of configuration."""
        lines = [
            "=" * 50,
            "CAMRIE Simulation Configuration",
            "=" * 50,
            f"Body Model:",
            f"  ρ (rho): {self.rho_path}",
            f"  T1:      {self.t1_path}",
            f"  T2:      {self.t2_path}",
            f"",
            f"Sequence: {self.sequence_path}",
            f"",
            f"Geometry:",
            f"  Isocenter:  ({self.isocenter_mm[0]:.1f}, {self.isocenter_mm[1]:.1f}, {self.isocenter_mm[2]:.1f}) mm",
            f"  Normal:     ({self.slice_normal[0]:.3f}, {self.slice_normal[1]:.3f}, {self.slice_normal[2]:.3f})",
            f"  Slices:     {self.num_slices}",
            f"  Thickness:  {self.slice_thickness_mm} mm",
            f"  Gap:        {self.slice_gap_mm} mm",
            f"  FOV:        {self.fov_mm[0]:.1f} x {self.fov_mm[1]:.1f} mm",
            f"",
            f"Simulation:",
            f"  B0:          {self.b0} T",
            f"  Spin Factor: {self.spin_factor}",
            f"",
            f"Output: {self.output_dir}",
            "=" * 50,
        ]
        return "\n".join(lines)


# =============================================================================
# Config Manager
# =============================================================================

class ConfigManager:
    """
    Manage multiple simulation configurations.
    
    Provides methods to:
    - Save/load configurations
    - List available configs
    - Create configs from presets
    """
    
    def __init__(self, config_dir: Union[str, Path] = None):
        """
        Initialize config manager.
        
        Parameters
        ----------
        config_dir : str or Path, optional
            Directory for storing configs. Uses ~/.camrie/configs if None.
        """
        if config_dir is None:
            config_dir = Path.home() / ".camrie" / "configs"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, name: str, config: SimulationConfig) -> Path:
        """Save configuration with given name."""
        path = self.config_dir / f"{name}.json"
        config.save(path)
        return path
    
    def load(self, name: str) -> SimulationConfig:
        """Load configuration by name."""
        path = self.config_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"No configuration named '{name}'")
        return SimulationConfig.load(path)
    
    def list(self) -> List[str]:
        """List available configuration names."""
        return [p.stem for p in self.config_dir.glob("*.json")]
    
    def delete(self, name: str) -> None:
        """Delete configuration by name."""
        path = self.config_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            print(f"Deleted configuration: {name}")
        else:
            print(f"Configuration not found: {name}")
