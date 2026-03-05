#!/usr/bin/env python3
"""
camrie.planning - Interactive MRI simulation planning tools.

3D visualization for planning oblique multi-slice MRI acquisitions:
1. Load and render body model surface
2. Interactively position imaging volume
3. Set slice direction, count, and spacing
4. Preview slice planes in 3D
5. Export configuration
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from camrie.pipeline import normalize, build_rotation_matrix
from camrie.config import SimulationConfig


# =============================================================================
# Body Model Loading
# =============================================================================

def load_body_model(rho_path: str) -> Tuple["pv.ImageData", "sitk.Image"]:
    """
    Load body model and convert to PyVista format.
    
    Parameters
    ----------
    rho_path : str
        Path to proton density NIfTI file.
    
    Returns
    -------
    pv_image : pv.ImageData
        PyVista image data for rendering.
    sitk_image : sitk.Image
        Original SimpleITK image with proper geometry.
        
    Raises
    ------
    ImportError
        If pyvista or SimpleITK is not installed.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista is required. Install with: pip install pyvista")
    if sitk is None:
        raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")
    
    # Load with SimpleITK to preserve geometry
    sitk_img = sitk.ReadImage(rho_path)
    
    # Get array (note: SimpleITK uses z,y,x ordering)
    arr = sitk.GetArrayFromImage(sitk_img)
    
    # Get geometry
    origin = sitk_img.GetOrigin()
    spacing = sitk_img.GetSpacing()
    
    # Create PyVista image with correct dimension handling
    pv_img = pv.ImageData()
    
    # arr.shape is (nz, ny, nx) - these are cell counts
    # PyVista dimensions = point counts = cell_counts + 1
    nx, ny, nz = arr.shape[2], arr.shape[1], arr.shape[0]
    pv_img.dimensions = [nx + 1, ny + 1, nz + 1]
    pv_img.origin = origin
    pv_img.spacing = spacing
    
    # Transpose array from (z,y,x) to (x,y,z) and add as cell data
    arr_xyz = np.transpose(arr, (2, 1, 0))
    pv_img.cell_data["values"] = arr_xyz.flatten(order='F')
    
    return pv_img, sitk_img


def get_image_center(pv_image: "pv.ImageData", sitk_image: "sitk.Image") -> np.ndarray:
    """
    Get the center of the image using SimpleITK's coordinate system.
    
    Parameters
    ----------
    pv_image : pv.ImageData
        PyVista image data.
    sitk_image : sitk.Image
        SimpleITK image.
    
    Returns
    -------
    center : np.ndarray
        Center point in physical coordinates (x, y, z).
    """
    size = np.array(sitk_image.GetSize())
    origin = np.array(sitk_image.GetOrigin())
    spacing = np.array(sitk_image.GetSpacing())
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
    
    center_idx = (size - 1) / 2.0
    center_physical = origin + direction @ (center_idx * spacing)
    
    return center_physical


def create_body_surface(
    pv_image: "pv.ImageData",
    sitk_image: "sitk.Image" = None,
) -> "pv.PolyData":
    """
    Create isosurface from body model for all voxels > 0.
    
    Parameters
    ----------
    pv_image : pv.ImageData
        PyVista image data.
    sitk_image : sitk.Image, optional
        SimpleITK image with direction information.
        
    Returns
    -------
    surface : pv.PolyData
        Surface mesh in SimpleITK coordinate space.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista is required")
    
    # Convert cell data to point data for better contour quality
    pv_image = pv_image.cell_data_to_point_data()
    
    iso_value = 1e-6  # Just above zero
    
    # Create isosurface
    surface = pv_image.contour([iso_value])
    
    # Smooth the surface if it has points
    if surface.n_points > 0:
        surface = surface.smooth(n_iter=50, relaxation_factor=0.1)
        
        # Transform points to SimpleITK coordinates if direction matrix provided
        if sitk_image is not None:
            direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
            if not np.allclose(direction, np.eye(3)):
                origin = np.array(sitk_image.GetOrigin())
                pts = surface.points.copy()
                offset = pts - origin
                pts_transformed = origin + (direction @ offset.T).T
                surface.points = pts_transformed
    
    return surface


# =============================================================================
# Slice Visualization
# =============================================================================

def create_slice_planes(
    center: np.ndarray,
    normal: np.ndarray,
    num_slices: int,
    slice_thickness: float,
    slice_gap: float,
    fov: Tuple[float, float],
) -> List["pv.PolyData"]:
    """
    Create plane meshes representing slice positions.
    
    Parameters
    ----------
    center : np.ndarray
        Isocenter position.
    normal : np.ndarray  
        Slice plane normal vector.
    num_slices : int
        Number of slices.
    slice_thickness : float
        Slice thickness in mm.
    slice_gap : float
        Gap between slices in mm.
    fov : tuple
        In-plane field of view (width, height) in mm.
    
    Returns
    -------
    list of pv.PolyData
        Plane meshes for each slice.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista is required")
    
    normal = normalize(np.array(normal, dtype=np.float64))
    center = np.array(center, dtype=np.float64)
    
    planes = []
    spacing = slice_thickness + slice_gap
    
    for i in range(num_slices):
        offset = (i - (num_slices - 1) / 2.0) * spacing
        slice_center = center + offset * normal
        
        plane = pv.Plane(
            center=slice_center,
            direction=normal,
            i_size=fov[0],
            j_size=fov[1],
            i_resolution=1,
            j_resolution=1,
        )
        
        planes.append(plane)
    
    return planes


def create_imaging_volume_box(
    center: np.ndarray,
    normal: np.ndarray,
    num_slices: int,
    slice_thickness: float,
    slice_gap: float,
    fov: Tuple[float, float],
) -> "pv.PolyData":
    """
    Create a box representing the total imaging volume.
    
    Parameters
    ----------
    center : np.ndarray
        Isocenter position.
    normal : np.ndarray
        Slice normal direction.
    num_slices : int
        Number of slices.
    slice_thickness : float
        Slice thickness in mm.
    slice_gap : float
        Gap between slices in mm.
    fov : tuple
        In-plane FOV (width, height) in mm.
        
    Returns
    -------
    pv.PolyData
        Box mesh.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista is required")
    
    normal = normalize(np.array(normal, dtype=np.float64))
    center = np.array(center, dtype=np.float64)
    
    R = build_rotation_matrix(normal)
    x_dir = R[0, :]
    y_dir = R[1, :]
    z_dir = R[2, :]  # = normal
    
    # Total depth
    spacing = slice_thickness + slice_gap
    total_depth = num_slices * spacing
    
    half_w = fov[0] / 2
    half_h = fov[1] / 2
    half_d = total_depth / 2
    
    # 8 corners in local coordinates
    corners_local = np.array([
        [-half_w, -half_h, -half_d],
        [+half_w, -half_h, -half_d],
        [+half_w, +half_h, -half_d],
        [-half_w, +half_h, -half_d],
        [-half_w, -half_h, +half_d],
        [+half_w, -half_h, +half_d],
        [+half_w, +half_h, +half_d],
        [-half_w, +half_h, +half_d],
    ])
    
    # Transform to world coordinates
    corners_world = np.zeros_like(corners_local)
    for i, c in enumerate(corners_local):
        corners_world[i] = center + c[0] * x_dir + c[1] * y_dir + c[2] * z_dir
    
    # Create box mesh
    faces = np.array([
        [4, 0, 1, 2, 3],  # bottom
        [4, 4, 5, 6, 7],  # top
        [4, 0, 1, 5, 4],  # front
        [4, 2, 3, 7, 6],  # back
        [4, 0, 3, 7, 4],  # left
        [4, 1, 2, 6, 5],  # right
    ])
    
    box = pv.PolyData(corners_world, faces)
    return box


# =============================================================================
# Interactive Planner
# =============================================================================

class MRISimulatorPlanner:
    """
    Interactive 3D visualization for MRI simulation planning.
    
    Parameters
    ----------
    rho_path : str
        Path to proton density NIfTI file.
    t1_path : str
        Path to T1 map NIfTI file.
    t2_path : str, optional
        Path to T2 map NIfTI file.
    sequence_path : str, optional
        Path to Pulseq sequence file.
        
    Examples
    --------
    >>> planner = MRISimulatorPlanner("rho.nii.gz", "t1.nii.gz")
    >>> planner.show()  # Opens interactive 3D window
    >>> config = planner.get_config()
    >>> config.save("simulation.json")
    """
    
    def __init__(
        self,
        rho_path: str,
        t1_path: str,
        t2_path: str = None,
        sequence_path: str = None,
    ):
        if not HAS_PYVISTA:
            raise ImportError(
                "PyVista is required for interactive planning. "
                "Install with: pip install camrie-tools[interactive]"
            )
        
        self.rho_path = rho_path
        self.t1_path = t1_path
        self.t2_path = t2_path or t1_path.replace("t1", "t2")
        self.sequence_path = sequence_path or ""
        
        # Load body model
        self.pv_image, self.sitk_image = load_body_model(rho_path)
        self.surface = create_body_surface(self.pv_image, self.sitk_image)
        self.body_center = get_image_center(self.pv_image, self.sitk_image)
        
        # Default parameters
        self.isocenter = self.body_center.copy()
        self.slice_normal = np.array([0.0, 0.0, 1.0])  # Axial
        self.num_slices = 5
        self.slice_thickness = 5.0
        self.slice_gap = 0.0
        self.fov = (200.0, 200.0)
        self.seq_fov = (300.0, 300.0)
        self.b0 = 3.0
        self.spin_factor = 1
        
        # Try to read sequence FOV
        if sequence_path:
            try:
                from camrie.pipeline import read_pulseq_params
                seq_params = read_pulseq_params(sequence_path)
                self.seq_fov = tuple(seq_params["fov_mm"])
            except:
                pass
    
    def set_orientation(self, preset: str) -> None:
        """
        Set slice orientation from preset.
        
        Parameters
        ----------
        preset : str
            One of: 'axial', 'coronal', 'sagittal'
        """
        from camrie.config import GEOMETRY_PRESETS
        preset = preset.upper()
        if preset in GEOMETRY_PRESETS:
            self.slice_normal = np.array(GEOMETRY_PRESETS[preset])
        else:
            raise ValueError(f"Unknown preset: {preset}")
    
    def get_config(self) -> SimulationConfig:
        """
        Get current configuration as SimulationConfig.
        
        Returns
        -------
        SimulationConfig
            Current simulation configuration.
        """
        return SimulationConfig(
            rho_path=self.rho_path,
            t1_path=self.t1_path,
            t2_path=self.t2_path,
            sequence_path=self.sequence_path,
            isocenter_mm=self.isocenter.tolist(),
            slice_normal=self.slice_normal.tolist(),
            num_slices=self.num_slices,
            slice_thickness_mm=self.slice_thickness,
            slice_gap_mm=self.slice_gap,
            fov_mm=list(self.fov),
            seq_fov_mm=list(self.seq_fov),
            b0=self.b0,
            spin_factor=self.spin_factor,
        )
    
    def show(self) -> None:
        """
        Open interactive 3D planning window.
        
        Uses PyVista for 3D visualization with interactive box widget
        for positioning the imaging volume.
        """
        plotter = pv.Plotter(title="CAMRIE MRI Planner")
        
        # Add body surface
        if self.surface.n_points > 0:
            plotter.add_mesh(
                self.surface,
                color='pink',
                opacity=0.5,
                smooth_shading=True,
                name='body',
            )
        
        # Create slice planes
        planes = create_slice_planes(
            self.isocenter,
            self.slice_normal,
            self.num_slices,
            self.slice_thickness,
            self.slice_gap,
            self.fov,
        )
        
        colors = ['cyan', 'yellow', 'magenta', 'lime', 'orange']
        for i, plane in enumerate(planes):
            plotter.add_mesh(
                plane,
                color=colors[i % len(colors)],
                opacity=0.7,
                name=f'slice_{i}',
            )
        
        # Create imaging volume box
        box = create_imaging_volume_box(
            self.isocenter,
            self.slice_normal,
            self.num_slices,
            self.slice_thickness,
            self.slice_gap,
            self.fov,
        )
        plotter.add_mesh(box, style='wireframe', color='white', line_width=2, name='box')
        
        # Add isocenter marker
        plotter.add_mesh(
            pv.Sphere(radius=5, center=self.isocenter),
            color='red',
            name='isocenter',
        )
        
        # Add axes
        plotter.add_axes()
        plotter.add_text(
            "CAMRIE MRI Planner\nRotate: Left-click drag | Zoom: Scroll | Pan: Shift+drag",
            position='upper_left',
            font_size=10,
        )
        
        plotter.show()


def create_plotly_visualization(
    surface: "pv.PolyData",
    slice_planes: List["pv.PolyData"],
    box_mesh: "pv.PolyData",
    isocenter: np.ndarray,
) -> "go.Figure":
    """
    Create 3D Plotly visualization for web interfaces.
    
    Parameters
    ----------
    surface : pv.PolyData
        Body surface mesh.
    slice_planes : list
        List of slice plane meshes.
    box_mesh : pv.PolyData
        Imaging volume box mesh.
    isocenter : np.ndarray
        Isocenter position.
        
    Returns
    -------
    go.Figure
        Plotly 3D figure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly is required. Install with: pip install plotly")
    
    fig = go.Figure()
    all_points = []
    
    # Add body surface
    if surface.n_points > 0:
        pts = surface.points
        all_points.append(pts)
        
        bounds = np.array([pts.min(axis=0), pts.max(axis=0)])
        extent = bounds[1] - bounds[0]
        marker_size = max(extent) / 200.0
        
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=marker_size, color='pink', opacity=0.6),
            name='Body Surface',
            hoverinfo='skip',
        ))
    
    # Add slice planes
    colors = ['cyan', 'yellow', 'magenta', 'lime', 'orange']
    for i, plane in enumerate(slice_planes):
        pts = plane.points
        if len(pts) > 0:
            all_points.append(pts)
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=3, color=colors[i % len(colors)]),
                name=f'Slice {i}',
                hoverinfo='skip',
            ))
    
    # Add box wireframe
    if box_mesh.n_points > 0:
        pts = box_mesh.points
        all_points.append(pts)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
        for edge in edges:
            x = [pts[edge[0], 0], pts[edge[1], 0]]
            y = [pts[edge[0], 1], pts[edge[1], 1]]
            z = [pts[edge[0], 2], pts[edge[1], 2]]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='white', width=3),
                name='Imaging Box',
                hoverinfo='skip',
                showlegend=False,
            ))
    
    # Add isocenter marker
    fig.add_trace(go.Scatter3d(
        x=[isocenter[0]], y=[isocenter[1]], z=[isocenter[2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Isocenter',
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        height=700,
        hovermode='closest',
        showlegend=True,
    )
    
    return fig
