# CAMRIE-Tools

**Cloud-Accelerated MRI Environment** - A Python package for MRI simulation with arbitrary slice orientations.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Arbitrary Slice Orientations**: Simulate axial, coronal, sagittal, or any oblique slice orientation
- **Multi-Slice Imaging**: Configure multiple slices with customizable thickness and gap
- **Pulseq Integration**: Read sequence parameters directly from .seq files
- **Interactive Planning**: 3D visualization for slice positioning
- **Multiple Interfaces**: CLI, Streamlit web app, and Jupyter notebooks
- **Cloud Ready**: Docker support for AWS Lambda, Fargate, and other cloud platforms

## Installation

### Basic Installation

```bash
pip install camrie-tools
```

### With Interactive Features (PyVista, Streamlit, Plotly)

```bash
pip install camrie-tools[interactive]
```

### For Cloud Deployment

```bash
pip install camrie-tools[cloud]
```

### Development Installation

```bash
git clone https://github.com/yourusername/camrie-tools.git
cd camrie-tools
pip install -e ".[dev]"
```

## Prerequisites

### Julia and KomaMRI

CAMRIE uses Julia and KomaMRI for Bloch simulation. Install Julia first:

```bash
# Linux
curl -fsSL https://install.julialang.org | sh

# Or download from: https://julialang.org/downloads/
```

Then run the setup command to install Julia packages:

```bash
camrie-setup
```

## Quick Start

### Python API

```python
from camrie import quick_sim, SimulationConfig

# Quick simulation with defaults
result = quick_sim(
    rho_path="phantom_rho.nii.gz",
    t1_path="phantom_t1.nii.gz",
    t2_path="phantom_t2.nii.gz",
    slice_normal=[0, 0, 1],  # Axial
    num_slices=5,
)

# Display results
import matplotlib.pyplot as plt
for i, img in enumerate(result['images']):
    plt.subplot(1, len(result['images']), i+1)
    plt.imshow(img, cmap='gray')
plt.show()
```

### Command Line

```bash
# Run simulation
camrie-sim --rho phantom_rho.nii.gz --t1 phantom_t1.nii.gz --normal 0 0 1 --num-slices 5

# Use geometry preset
camrie-sim --rho rho.nii.gz --t1 t1.nii.gz --preset CORONAL

# Launch web interface
camrie-streamlit

# Setup Julia environment
camrie-setup
```

### Streamlit Web Interface

```bash
camrie-streamlit
```

This opens an interactive web interface for:
- Loading phantom files
- Selecting slice orientations with presets
- Adjusting parameters with sliders
- 3D preview of slice positions
- Running simulations

### Jupyter Notebooks

```python
from camrie import notebook_interface
notebook_interface()  # Interactive widget interface
```

## Configuration

### SimulationConfig

The main configuration class for simulations:

```python
from camrie import SimulationConfig, GEOMETRY_PRESETS

config = SimulationConfig(
    rho_path="phantom_rho.nii.gz",
    t1_path="phantom_t1.nii.gz",
    t2_path="phantom_t2.nii.gz",
    sequence_path="sequence.seq",
    isocenter_mm=[0, 0, 0],  # Center of imaging volume
    slice_normal=[0, 0, 1],  # Slice direction (axial)
    num_slices=10,
    slice_thickness_mm=5.0,
    slice_gap_mm=1.0,
    fov_mm=[200, 200],
    b0=3.0,  # Tesla
    spin_factor=1,  # 1-4, higher = more accurate but slower
)

# Save/load configuration
config.save("my_simulation.json")
config = SimulationConfig.load("my_simulation.json")

# Use presets
config = SimulationConfig.from_preset(
    "CORONAL",
    rho_path="rho.nii.gz",
    t1_path="t1.nii.gz",
    t2_path="t2.nii.gz",
    sequence_path="seq.seq",
)
```

### Geometry Presets

```python
from camrie import GEOMETRY_PRESETS

# Available presets:
# - AXIAL: [0, 0, 1] - Superior-Inferior
# - CORONAL: [0, 1, 0] - Anterior-Posterior
# - SAGITTAL: [1, 0, 0] - Left-Right
# - OBLIQUE_45_AP: [0, 0.707, 0.707]
# - OBLIQUE_45_LR: [0.707, 0, 0.707]
```

## Built-in Phantoms

```python
from camrie.phantoms import get_builtin_phantom, create_shepp_logan_phantom

# Use built-in elephant phantom
phantom = get_builtin_phantom('elephant')
result = quick_sim(
    rho_path=phantom['rho_path'],
    t1_path=phantom['t1_path'],
    t2_path=phantom['t2_path'],
)

# Generate Shepp-Logan phantom
phantom = get_builtin_phantom('shepp_logan')
```

## Interactive Planning

### PyVista (Desktop)

```python
from camrie.planning import MRISimulatorPlanner

planner = MRISimulatorPlanner("rho.nii.gz", "t1.nii.gz")
planner.set_orientation("CORONAL")
planner.show()  # Opens 3D visualization

config = planner.get_config()
config.save("planned_simulation.json")
```

### Plotly (Web)

```python
from camrie.planning import (
    load_body_model,
    create_body_surface,
    create_slice_planes,
    create_plotly_visualization,
)

pv_img, sitk_img = load_body_model("rho.nii.gz")
surface = create_body_surface(pv_img, sitk_img)
planes = create_slice_planes(center, normal, num_slices, thickness, gap, fov)
fig = create_plotly_visualization(surface, planes, box_mesh, isocenter)
fig.show()
```

## Docker Deployment

### Build Docker Image

```bash
cd docker
docker build -f Dockerfile.fargate -t camrie:latest .
```

### Run in Docker

```bash
docker run -v /path/to/data:/data camrie:latest \
    camrie-sim --rho /data/rho.nii.gz --t1 /data/t1.nii.gz --normal 0 0 1
```

### AWS Fargate

See [docker/README.md](docker/README.md) for detailed cloud deployment instructions.

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `quick_sim()` | Run simulation with minimal configuration |
| `run_pipeline()` | Full pipeline with all options |
| `read_pulseq_params()` | Extract parameters from .seq files |
| `reconstruct_image_2d()` | 2D FFT reconstruction from k-space |

### Configuration

| Class | Description |
|-------|-------------|
| `SimulationConfig` | Main configuration dataclass |
| `GEOMETRY_PRESETS` | Standard orientation presets |
| `ConfigManager` | Save/load multiple configurations |

### Phantoms

| Function | Description |
|----------|-------------|
| `load_nifti_maps()` | Load phantom from NIfTI files |
| `create_shepp_logan_phantom()` | Generate 3D Shepp-Logan |
| `get_builtin_phantom()` | Access built-in phantoms |

### Planning

| Class/Function | Description |
|----------------|-------------|
| `MRISimulatorPlanner` | Interactive 3D planner |
| `load_body_model()` | Load NIfTI as PyVista |
| `create_slice_planes()` | Generate slice plane meshes |
| `create_plotly_visualization()` | Web-based 3D visualization |

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

```bash
# Run tests
pytest

# Run linters
ruff check camrie/
black camrie/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use CAMRIE in your research, please cite:

```bibtex
@software{camrie2024,
  title={CAMRIE-Tools: Cloud-Accelerated MRI Environment},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/camrie-tools}
}
```

## Acknowledgments

- [KomaMRI](https://github.com/cncastillo/KomaMRI) - Bloch simulation engine
- [pypulseq](https://github.com/imr-framework/pypulseq) - Pulseq sequence parsing
- [PyVista](https://github.com/pyvista/pyvista) - 3D visualization
- [SimpleITK](https://simpleitk.org/) - Medical image I/O
