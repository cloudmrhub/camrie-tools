"""
camrie-tools: Cloud-Accelerated MRI Environment

A Python package for MRI simulation with arbitrary slice orientations,
supporting local execution, cloud deployment, and interactive planning.

Quick Start:
    >>> from camrie import quick_sim
    >>> result = quick_sim(
    ...     rho_path="phantom_rho.nii.gz",
    ...     t1_path="phantom_t1.nii.gz",
    ...     t2_path="phantom_t2.nii.gz",
    ...     sequence_path="sequence.seq",
    ... )
    
For interactive planning:
    >>> from camrie import notebook_interface
    >>> notebook_interface()  # Jupyter widget interface

Author: CAMRIE Team
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "CAMRIE Team"

# Core exports
from camrie.config import SimulationConfig, GEOMETRY_PRESETS
from camrie.pipeline import (
    run_pipeline,
    read_pulseq_params,
    reconstruct_image_2d,
    normalize,
    build_rotation_matrix,
    compute_series_geometry,
)

# Phantom utilities
from camrie.phantoms import (
    load_nifti_maps,
    create_shepp_logan_phantom,
    get_builtin_phantom,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "SimulationConfig",
    "GEOMETRY_PRESETS",
    # Pipeline
    "run_pipeline",
    "read_pulseq_params",
    "reconstruct_image_2d",
    "normalize",
    "build_rotation_matrix",
    "compute_series_geometry",
    # Phantoms
    "load_nifti_maps",
    "create_shepp_logan_phantom",
    "get_builtin_phantom",
    # Convenience
    "quick_sim",
    "notebook_interface",
]


def quick_sim(
    rho_path: str,
    t1_path: str,
    t2_path: str = None,
    sequence_path: str = None,
    isocenter_mm: list = None,
    slice_normal: list = None,
    num_slices: int = 1,
    slice_thickness_mm: float = 5.0,
    slice_gap_mm: float = 1.0,
    fov_mm: list = None,
    b0: float = 3.0,
    spin_factor: int = 1,
    output_dir: str = None,
    use_gpu: bool = False,
) -> dict:
    """
    Quick simulation function for simple use cases.
    
    Runs a complete MRI simulation with sensible defaults.
    
    Parameters
    ----------
    rho_path : str
        Path to proton density NIfTI file.
    t1_path : str
        Path to T1 map NIfTI file.
    t2_path : str, optional
        Path to T2 map NIfTI file. If None, uses T1*1.1 as T2.
    sequence_path : str, optional
        Path to Pulseq .seq file. Uses built-in GRE if None.
    isocenter_mm : list, optional
        [x, y, z] isocenter in mm. Uses image center if None.
    slice_normal : list, optional
        [nx, ny, nz] slice normal direction. Default [0, 0, 1] (axial).
    num_slices : int
        Number of slices. Default 1.
    slice_thickness_mm : float
        Slice thickness in mm. Default 5.0.
    slice_gap_mm : float
        Gap between slices in mm. Default 1.0.
    fov_mm : list, optional
        [fov_x, fov_y] field of view in mm. Auto-detected from sequence.
    b0 : float
        Main magnetic field strength (Tesla). Default 3.0.
    spin_factor : int
        Spin density multiplier (1-4). Higher = more spins per voxel. Default 1.
    output_dir : str, optional
        Output directory. Uses temp dir if None.
    use_gpu : bool
        Use GPU acceleration if available. Default False.
        
    Returns
    -------
    dict
        Simulation results including:
        - 'images': List of reconstructed 2D images
        - 'kspace': Raw k-space data per slice
        - 'config': SimulationConfig used
        - 'output_dir': Path to output files
        
    Examples
    --------
    >>> # Basic axial acquisition
    >>> result = quick_sim("rho.nii.gz", "t1.nii.gz")
    
    >>> # Oblique coronal with multiple slices
    >>> result = quick_sim(
    ...     "rho.nii.gz", "t1.nii.gz", "t2.nii.gz",
    ...     slice_normal=[0, 1, 0],  # Coronal
    ...     num_slices=10,
    ...     spin_factor=2,
    ... )
    """
    import tempfile
    from pathlib import Path
    import SimpleITK as sitk
    
    # Generate temp output dir if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="camrie_sim_")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Auto-detect isocenter from image
    if isocenter_mm is None:
        img = sitk.ReadImage(rho_path)
        size = img.GetSize()
        center_index = [s // 2 for s in size]
        isocenter_mm = list(img.TransformIndexToPhysicalPoint(center_index))
    
    # Default slice normal (axial)
    if slice_normal is None:
        slice_normal = [0.0, 0.0, 1.0]
    
    # Get sequence params or use defaults
    if sequence_path is None:
        # Use built-in sequence
        import importlib.resources as pkg_resources
        from camrie import data
        with pkg_resources.as_file(pkg_resources.files(data) / "ge.seq") as p:
            sequence_path = str(p)
    
    seq_params = read_pulseq_params(sequence_path)
    if fov_mm is None:
        fov_mm = seq_params.get("fov_mm", [300.0, 300.0])
    
    # Create config
    config = SimulationConfig(
        rho_path=rho_path,
        t1_path=t1_path,
        t2_path=t2_path or t1_path,  # Fallback
        sequence_path=sequence_path,
        isocenter_mm=isocenter_mm,
        slice_normal=slice_normal,
        num_slices=num_slices,
        slice_thickness_mm=slice_thickness_mm,
        slice_gap_mm=slice_gap_mm,
        fov_mm=fov_mm,
        seq_fov_mm=seq_params.get("fov_mm", fov_mm),
        b0=b0,
        spin_factor=spin_factor,
        output_dir=output_dir,
    )
    
    # Run simulation
    images, kspace_list = run_pipeline(config, use_gpu=use_gpu)
    
    return {
        "images": images,
        "kspace": kspace_list,
        "config": config,
        "output_dir": output_dir,
    }


def notebook_interface(config: SimulationConfig = None):
    """
    Create an interactive Jupyter notebook interface for simulation planning.
    
    Displays widgets for:
    - Loading body model
    - Selecting geometry presets
    - Adjusting slice parameters
    - Running simulation
    - Viewing results
    
    Parameters
    ----------
    config : SimulationConfig, optional
        Pre-filled configuration. Creates empty config if None.
        
    Returns
    -------
    ipywidgets container
        Interactive widget for simulation planning.
        
    Note
    ----
    Requires ipywidgets, ipyvolume, and plotly for full functionality.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        raise ImportError(
            "notebook_interface requires ipywidgets. "
            "Install with: pip install ipywidgets"
        )
    
    # Create basic widgets
    output = widgets.Output()
    
    rho_input = widgets.Text(
        description='Rho path:',
        placeholder='path/to/rho.nii.gz',
        value=config.rho_path if config else '',
    )
    t1_input = widgets.Text(
        description='T1 path:',
        placeholder='path/to/t1.nii.gz',
        value=config.t1_path if config else '',
    )
    t2_input = widgets.Text(
        description='T2 path:',
        placeholder='path/to/t2.nii.gz',
        value=config.t2_path if config else '',
    )
    
    sequence_input = widgets.Text(
        description='Sequence:',
        placeholder='path/to/sequence.seq',
        value=config.sequence_path if config else '',
    )
    
    # Geometry presets dropdown
    preset_dropdown = widgets.Dropdown(
        options=['AXIAL', 'CORONAL', 'SAGITTAL', 'OBLIQUE'],
        value='AXIAL',
        description='Preset:',
    )
    
    # Sliders
    num_slices_slider = widgets.IntSlider(
        value=config.num_slices if config else 10,
        min=1, max=50, step=1,
        description='Slices:',
    )
    thickness_slider = widgets.FloatSlider(
        value=config.slice_thickness_mm if config else 5.0,
        min=1.0, max=20.0, step=0.5,
        description='Thickness:',
    )
    gap_slider = widgets.FloatSlider(
        value=config.slice_gap_mm if config else 1.0,
        min=0, max=10.0, step=0.1,
        description='Gap:',
    )
    spin_factor_slider = widgets.IntSlider(
        value=config.spin_factor if config else 1,
        min=1, max=4, step=1,
        description='Spin Factor:',
    )
    
    # Run button
    run_button = widgets.Button(
        description='Run Simulation',
        button_style='primary',
        icon='play',
    )
    
    def on_run_click(b):
        with output:
            output.clear_output()
            print("Starting simulation...")
            try:
                result = quick_sim(
                    rho_path=rho_input.value,
                    t1_path=t1_input.value,
                    t2_path=t2_input.value,
                    sequence_path=sequence_input.value if sequence_input.value else None,
                    slice_normal=GEOMETRY_PRESETS.get(preset_dropdown.value, [0, 0, 1]),
                    num_slices=num_slices_slider.value,
                    slice_thickness_mm=thickness_slider.value,
                    slice_gap_mm=gap_slider.value,
                    spin_factor=spin_factor_slider.value,
                )
                print(f"Simulation complete! Output: {result['output_dir']}")
                
                # Display first image
                import matplotlib.pyplot as plt
                if result['images']:
                    plt.figure(figsize=(8, 8))
                    plt.imshow(result['images'][0], cmap='gray')
                    plt.title("Reconstructed Image (Slice 1)")
                    plt.colorbar()
                    plt.show()
            except Exception as e:
                print(f"Error: {e}")
    
    run_button.on_click(on_run_click)
    
    # Layout
    file_box = widgets.VBox([
        widgets.Label("File Inputs:"),
        rho_input, t1_input, t2_input, sequence_input,
    ])
    
    param_box = widgets.VBox([
        widgets.Label("Geometry:"),
        preset_dropdown,
        num_slices_slider,
        thickness_slider,
        gap_slider,
        spin_factor_slider,
    ])
    
    ui = widgets.VBox([
        widgets.HTML("<h2>CAMRIE MRI Simulator</h2>"),
        widgets.HBox([file_box, param_box]),
        run_button,
        output,
    ])
    
    display(ui)
    return ui
