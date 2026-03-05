#!/usr/bin/env python3
"""
camrie.cli - Command-line interface for CAMRIE tools.

Entry points:
- camrie-sim: Run MRI simulation
- camrie-streamlit: Launch Streamlit web interface
- camrie-setup: Setup Julia environment
- camrie-notebook: Open Jupyter notebook with example
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main_simulation():
    """Main entry point for camrie-sim command."""
    import numpy as np
    import SimpleITK as sitk
    
    from camrie.config import SimulationConfig, GEOMETRY_PRESETS
    from camrie.pipeline import run_pipeline, normalize
    
    parser = argparse.ArgumentParser(
        description="CAMRIE MRI Simulation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Axial acquisition with 5 slices
  camrie-sim --rho phantom_rho.nii.gz --t1 phantom_t1.nii.gz --normal 0 0 1 --num-slices 5

  # Coronal with custom FOV
  camrie-sim --rho rho.nii.gz --t1 t1.nii.gz --normal 0 1 0 --fov 200 200

  # Use geometry preset
  camrie-sim --rho rho.nii.gz --t1 t1.nii.gz --preset SAGITTAL

  # Load config file
  camrie-sim --config simulation.json
""",
    )
    
    # Config file (alternative to CLI args)
    parser.add_argument("--config", "-c", type=str,
                        help="Load configuration from JSON file")
    
    # Input files
    parser.add_argument("--rho", type=str, help="Path to proton density NIfTI")
    parser.add_argument("--t1", type=str, help="Path to T1 map NIfTI")
    parser.add_argument("--t2", type=str, help="Path to T2 map NIfTI")
    parser.add_argument("--sequence", "-s", type=str, help="Path to Pulseq .seq file")
    
    # Output
    parser.add_argument("--output", "-o", default="camrie_output",
                        help="Output directory")
    
    # Geometry - either preset or manual
    parser.add_argument("--preset", choices=list(GEOMETRY_PRESETS.keys()),
                        help="Use geometry preset (AXIAL, CORONAL, SAGITTAL, etc.)")
    parser.add_argument("--normal", "-n", type=float, nargs=3,
                        metavar=("NX", "NY", "NZ"),
                        help="Slice plane normal vector (overrides --preset)")
    
    # Isocenter
    parser.add_argument("--isocenter", type=float, nargs=3,
                        metavar=("X", "Y", "Z"),
                        help="Isocenter in body space (mm). Default: body center")
    
    # Slice parameters
    parser.add_argument("--num-slices", type=int, default=5)
    parser.add_argument("--slice-thickness", type=float, default=5.0)
    parser.add_argument("--slice-gap", type=float, default=0.0)
    
    # FOV and matrix
    parser.add_argument("--fov", type=float, nargs=2,
                        help="Phantom FOV (mm). Auto-read from seq if not specified")
    parser.add_argument("--seq-fov", type=float, nargs=2,
                        help="Sequence FOV (mm). Auto-read from seq if not specified")
    
    # Simulation parameters
    parser.add_argument("--spin-factor", type=int, default=1,
                        help="Spin density multiplier (1-4)")
    parser.add_argument("--b0", type=float, default=3.0,
                        help="Main magnetic field strength (Tesla)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration")
    
    # Export
    parser.add_argument("--nifti", action="store_true", default=True,
                        help="Save assembled NIfTI volume (default: True)")
    parser.add_argument("--no-nifti", action="store_false", dest="nifti",
                        help="Skip NIfTI volume assembly")
    parser.add_argument("--dicom", action="store_true",
                        help="Export DICOM files")
    
    args = parser.parse_args()
    
    # Load config from file or build from args
    if args.config:
        config = SimulationConfig.load(args.config)
    else:
        # Validate required args
        if not args.rho or not args.t1:
            parser.error("--rho and --t1 are required unless using --config")
        
        # Determine slice normal
        if args.normal:
            slice_normal = args.normal
        elif args.preset:
            slice_normal = GEOMETRY_PRESETS[args.preset]
        else:
            slice_normal = [0.0, 0.0, 1.0]  # Default: axial
        
        # Determine isocenter
        if args.isocenter:
            isocenter_mm = args.isocenter
        else:
            # Auto-compute from image center
            rho_img = sitk.ReadImage(args.rho)
            size = np.array(rho_img.GetSize())
            origin = np.array(rho_img.GetOrigin())
            spacing = np.array(rho_img.GetSpacing())
            direction = np.array(rho_img.GetDirection()).reshape(3, 3)
            center_idx = (size - 1) / 2.0
            isocenter_mm = origin + direction @ (center_idx * spacing)
            isocenter_mm = isocenter_mm.tolist()
        
        # Determine FOV from sequence if not specified
        fov_mm = args.fov or [300.0, 300.0]
        seq_fov_mm = args.seq_fov or fov_mm
        
        config = SimulationConfig(
            rho_path=args.rho,
            t1_path=args.t1,
            t2_path=args.t2 or args.t1,
            sequence_path=args.sequence or "",
            isocenter_mm=isocenter_mm,
            slice_normal=slice_normal,
            num_slices=args.num_slices,
            slice_thickness_mm=args.slice_thickness,
            slice_gap_mm=args.slice_gap,
            fov_mm=fov_mm,
            seq_fov_mm=seq_fov_mm,
            b0=args.b0,
            spin_factor=args.spin_factor,
            output_dir=args.output,
        )
    
    # Print config summary
    print(config.summary())
    
    # Run pipeline
    recon_images, kspace_list, volume_path = run_pipeline(
        config, 
        use_gpu=args.gpu,
        save_nifti=args.nifti,
        save_dicom=args.dicom,
    )
    
    print(f"\n✓ Simulation complete!")
    print(f"  Output: {config.output_dir}")
    print(f"  Slices: {len(recon_images)}")
    if volume_path:
        print(f"  Volume: {volume_path}")


def main_streamlit():
    """Launch Streamlit web interface."""
    try:
        import streamlit
    except ImportError:
        print("ERROR: Streamlit is required. Install with:")
        print("  pip install camrie-tools[interactive]")
        sys.exit(1)
    
    # Get path to streamlit app without importing it (to avoid early execution)
    app_path = str(Path(__file__).parent / "streamlit_app.py")
    
    print(f"Launching CAMRIE Streamlit app...")
    subprocess.run(["streamlit", "run", app_path], check=True)


def main_setup():
    """Setup Julia environment for KomaMRI."""
    print("=" * 60)
    print("CAMRIE Julia Environment Setup")
    print("=" * 60)
    
    # Check Julia installation
    try:
        result = subprocess.run(["julia", "--version"], capture_output=True, text=True)
        print(f"✓ Julia found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("✗ Julia not found!")
        print("\nPlease install Julia from: https://julialang.org/downloads/")
        print("After installation, add Julia to your PATH and run this command again.")
        sys.exit(1)
    
    # Install Julia packages
    print("\nInstalling Julia packages...")
    
    julia_code = """
    import Pkg
    Pkg.add("KomaMRI")
    Pkg.add("JSON")
    Pkg.add("NPZ")
    Pkg.add("LinearAlgebra")
    println("\\n✓ Julia packages installed successfully!")
    """
    
    result = subprocess.run(
        ["julia", "-e", julia_code],
        check=True,
    )
    
    print("\n" + "=" * 60)
    print("Setup complete! You can now run simulations with:")
    print("  camrie-sim --help")
    print("=" * 60)


def main_notebook():
    """Open Jupyter notebook with examples."""
    try:
        import jupyter
    except ImportError:
        print("ERROR: Jupyter is required. Install with:")
        print("  pip install jupyter")
        sys.exit(1)
    
    # Get example notebook path
    try:
        import importlib.resources as pkg_resources
        from camrie import examples
        with pkg_resources.as_file(pkg_resources.files(examples) / "getting_started.ipynb") as p:
            notebook_path = str(p)
    except Exception:
        # Create a simple notebook
        notebook_path = _create_example_notebook()
    
    print(f"Opening example notebook: {notebook_path}")
    subprocess.run(["jupyter", "notebook", notebook_path], check=True)


def _create_example_notebook() -> str:
    """Create an example Jupyter notebook."""
    import tempfile
    import json
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# CAMRIE MRI Simulation - Getting Started\n",
                    "\n",
                    "This notebook demonstrates how to run MRI simulations with CAMRIE."
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Import CAMRIE\n",
                    "from camrie import quick_sim, SimulationConfig, GEOMETRY_PRESETS\n",
                    "from camrie.phantoms import get_builtin_phantom, create_shepp_logan_phantom"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Quick Simulation\n",
                    "\n",
                    "The simplest way to run a simulation:"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Get built-in phantom\n",
                    "phantom = get_builtin_phantom('elephant')\n",
                    "\n",
                    "# Run quick simulation\n",
                    "result = quick_sim(\n",
                    "    rho_path=phantom['rho_path'],\n",
                    "    t1_path=phantom['t1_path'],\n",
                    "    t2_path=phantom['t2_path'],\n",
                    "    num_slices=3,\n",
                    "    slice_normal=[0, 0, 1],  # Axial\n",
                    ")"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Display results\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "fig, axes = plt.subplots(1, len(result['images']), figsize=(15, 5))\n",
                    "for i, img in enumerate(result['images']):\n",
                    "    axes[i].imshow(img, cmap='gray')\n",
                    "    axes[i].set_title(f'Slice {i+1}')\n",
                    "    axes[i].axis('off')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Interactive Interface\n",
                    "\n",
                    "For interactive simulation planning:"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from camrie import notebook_interface\n",
                    "notebook_interface()"
                ],
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    path = os.path.join(tempfile.gettempdir(), "camrie_getting_started.ipynb")
    with open(path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    return path


if __name__ == "__main__":
    main_simulation()
