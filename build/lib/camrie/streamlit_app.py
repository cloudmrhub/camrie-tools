#!/usr/bin/env python3
"""
camrie.streamlit_app - Interactive MRI Simulation Planning with Streamlit.

Web-based UI for planning oblique multi-slice MRI acquisitions with:
- 3D visualization of body surface and slice planes
- Interactive sliders for all parameters
- Configuration save/load
- Direct simulation launch

Run with:
    camrie-streamlit
    
Or directly:
    streamlit run camrie/streamlit_app.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly is required. Install with: pip install plotly")
    st.stop()

try:
    import SimpleITK as sitk
except ImportError:
    st.error("SimpleITK is required. Install with: pip install SimpleITK")
    st.stop()

from camrie.pipeline import normalize, build_rotation_matrix, read_pulseq_params
from camrie.planning import (
    load_body_model,
    get_image_center,
    create_body_surface,
    create_slice_planes,
    create_imaging_volume_box,
    create_plotly_visualization,
)
from camrie.config import SimulationConfig, GEOMETRY_PRESETS


# =============================================================================
# Streamlit Configuration
# =============================================================================

st.set_page_config(
    page_title="CAMRIE MRI Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧲 CAMRIE MRI Simulation Planner")
st.markdown("Interactive planning for oblique multi-slice MRI acquisitions")


# =============================================================================
# Session State
# =============================================================================

if "pv_image" not in st.session_state:
    st.session_state.pv_image = None
    st.session_state.sitk_image = None
    st.session_state.surface = None
    st.session_state.body_center = None


# =============================================================================
# Cached Loading
# =============================================================================

@st.cache_resource
def cached_load_body_model(rho_path: str):
    """Load body model (cached)."""
    with st.spinner("Loading body model..."):
        pv_img, sitk_img = load_body_model(rho_path)
        surface = create_body_surface(pv_img, sitk_img)
        body_center = get_image_center(pv_img, sitk_img)
        return pv_img, sitk_img, surface, body_center


# =============================================================================
# Sidebar - File Selection
# =============================================================================

st.sidebar.header("📁 Data Files")

# Default to package data if available
default_rho = "data/rhoh.nii.gz" if os.path.exists("data/rhoh.nii.gz") else ""
default_t1 = "data/t1.nii.gz" if os.path.exists("data/t1.nii.gz") else ""
default_t2 = "data/t2.nii.gz" if os.path.exists("data/t2.nii.gz") else ""

rho_path = st.sidebar.text_input("Rho (ρ) path", value=default_rho)
t1_path = st.sidebar.text_input("T1 path", value=default_t1)
t2_path = st.sidebar.text_input("T2 path", value=default_t2)
seq_path = st.sidebar.text_input("Sequence path", value="")

# Load body model
if rho_path and os.path.exists(rho_path):
    pv_img, sitk_img, surface, body_center = cached_load_body_model(rho_path)
    st.session_state.pv_image = pv_img
    st.session_state.sitk_image = sitk_img
    st.session_state.surface = surface
    st.session_state.body_center = body_center
else:
    st.sidebar.warning("⚠️ Enter path to rho NIfTI file")
    st.info("Enter the path to your proton density (rho) NIfTI file in the sidebar to begin.")
    st.stop()

# Read sequence FOV
default_seq_fov = (300.0, 300.0)
if seq_path and os.path.exists(seq_path):
    try:
        seq_params = read_pulseq_params(seq_path)
        default_seq_fov = tuple(seq_params["fov_mm"])
    except:
        pass


# =============================================================================
# Sidebar - Geometry Parameters
# =============================================================================

st.sidebar.header("⚙️ Geometry")

# Isocenter
col1, col2, col3 = st.sidebar.columns(3)
iso_x = col1.number_input("Iso X (mm)", value=float(body_center[0]), step=5.0)
iso_y = col2.number_input("Iso Y (mm)", value=float(body_center[1]), step=5.0)
iso_z = col3.number_input("Iso Z (mm)", value=float(body_center[2]), step=5.0)
isocenter = np.array([iso_x, iso_y, iso_z])

# Slice orientation - preset buttons
st.sidebar.subheader("Orientation")

col1, col2, col3 = st.sidebar.columns(3)
if col1.button("🔵 Axial"):
    st.session_state.slice_normal = np.array([0.0, 0.0, 1.0])

if col2.button("🟡 Coronal"):
    st.session_state.slice_normal = np.array([0.0, 1.0, 0.0])

if col3.button("🟢 Sagittal"):
    st.session_state.slice_normal = np.array([1.0, 0.0, 0.0])

# Custom normal vector
if "slice_normal" not in st.session_state:
    st.session_state.slice_normal = np.array([0.0, 0.0, 1.0])

col1, col2, col3 = st.sidebar.columns(3)
nx = col1.number_input("Normal X", value=float(st.session_state.slice_normal[0]), step=0.1, format="%.3f")
ny = col2.number_input("Normal Y", value=float(st.session_state.slice_normal[1]), step=0.1, format="%.3f")
nz = col3.number_input("Normal Z", value=float(st.session_state.slice_normal[2]), step=0.1, format="%.3f")

# Normalize
raw_normal = np.array([nx, ny, nz])
norm = np.linalg.norm(raw_normal)
if norm > 1e-6:
    slice_normal = raw_normal / norm
else:
    slice_normal = np.array([0.0, 0.0, 1.0])
    st.sidebar.warning("Normal cannot be zero")

st.sidebar.caption(f"‖n‖ = {norm:.3f} → normalized")
st.session_state.slice_normal = slice_normal

# Slices
st.sidebar.subheader("Slices")
num_slices = st.sidebar.slider("Number of slices", 1, 50, value=5)
slice_thickness = st.sidebar.slider("Slice thickness (mm)", 1.0, 20.0, value=5.0, step=0.5)
slice_gap = st.sidebar.slider("Slice gap (mm)", 0.0, 20.0, value=0.0, step=0.5)

# FOV
st.sidebar.subheader("Field of View")
col1, col2 = st.sidebar.columns(2)
fov_x = col1.slider("FOV X (mm)", 50.0, 500.0, value=200.0, step=10.0)
fov_y = col2.slider("FOV Y (mm)", 50.0, 500.0, value=200.0, step=10.0)
fov = (fov_x, fov_y)

# Sequence FOV
col1, col2 = st.sidebar.columns(2)
seq_fov_x = col1.number_input("Seq FOV X", value=default_seq_fov[0])
seq_fov_y = col2.number_input("Seq FOV Y", value=default_seq_fov[1])
seq_fov = (seq_fov_x, seq_fov_y)

# B0
b0 = st.sidebar.number_input("B0 (T)", value=3.0, step=0.1)

# Spin factor
st.sidebar.subheader("Simulation")
spin_factor = st.sidebar.slider(
    "Spin factor", 
    min_value=1, max_value=4, value=1, step=1,
    help="Spin density multiplier (1=native res, 2=2x spins/voxel)"
)

# Output directory
output_dir = st.sidebar.text_input("Output directory", value="camrie_output")


# =============================================================================
# Main Content - 3D Visualization
# =============================================================================

# Create slice planes and box
slice_planes = create_slice_planes(
    isocenter,
    slice_normal,
    num_slices,
    slice_thickness,
    slice_gap,
    fov,
)

box_mesh = create_imaging_volume_box(
    isocenter,
    slice_normal,
    num_slices,
    slice_thickness,
    slice_gap,
    fov,
)

# Create and display 3D visualization
fig = create_plotly_visualization(surface, slice_planes, box_mesh, isocenter)
st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Configuration Display & Actions
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Configuration Summary")
    config = SimulationConfig(
        rho_path=rho_path,
        t1_path=t1_path,
        t2_path=t2_path,
        sequence_path=seq_path,
        isocenter_mm=isocenter.tolist(),
        slice_normal=slice_normal.tolist(),
        num_slices=num_slices,
        slice_thickness_mm=slice_thickness,
        slice_gap_mm=slice_gap,
        fov_mm=list(fov),
        seq_fov_mm=list(seq_fov),
        b0=b0,
        spin_factor=spin_factor,
        output_dir=output_dir,
    )
    
    st.json(config.to_dict())

with col2:
    st.subheader("💾 Save & Simulate")
    
    # Save config
    if st.button("💾 Save Configuration", key="save_config"):
        config.save("simulation_config.json")
        st.success("✅ Saved to `simulation_config.json`")
    
    # Load config
    if st.button("📂 Load Configuration", key="load_config"):
        try:
            loaded_config = SimulationConfig.load("simulation_config.json")
            st.success("✅ Configuration loaded")
            st.json(loaded_config.to_dict())
        except FileNotFoundError:
            st.error("❌ No `simulation_config.json` found")
    
    st.divider()
    
    # Run simulation
    if st.button("🚀 Run Simulation", key="run_sim", type="primary"):
        st.info("🔄 Starting simulation... (this may take several minutes)")
        
        cmd = [
            sys.executable, "-m", "camrie.cli",
            "--rho", rho_path,
            "--t1", t1_path,
            "--t2", t2_path,
            "--output", output_dir,
            "--normal", str(slice_normal[0]), str(slice_normal[1]), str(slice_normal[2]),
            "--isocenter", str(isocenter[0]), str(isocenter[1]), str(isocenter[2]),
            "--num-slices", str(num_slices),
            "--slice-thickness", str(slice_thickness),
            "--slice-gap", str(slice_gap),
            "--fov", str(fov[0]), str(fov[1]),
            "--seq-fov", str(seq_fov[0]), str(seq_fov[1]),
            "--spin-factor", str(spin_factor),
        ]
        
        if seq_path:
            cmd.extend(["--sequence", seq_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                st.success("✅ Simulation completed!")
                st.info(f"Results in: `{output_dir}`")
            else:
                st.error(f"❌ Simulation failed:\n{result.stderr}")
        except subprocess.TimeoutExpired:
            st.error("❌ Simulation timed out (> 1 hour)")
        except Exception as e:
            st.error(f"❌ Error: {e}")


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.markdown("""
**Controls:**
- Drag to rotate the 3D view
- Scroll to zoom
- Click legend items to toggle visibility

**Quick Start:**
1. Enter paths to phantom files (rho, T1, T2)
2. Select orientation (Axial/Coronal/Sagittal) or set custom normal
3. Adjust slice count, thickness, and FOV
4. Click "Run Simulation" to start
""")
