# camrie-tools

Python tools for CAMRIE MRI reconstruction workflows using KomaMRI through
`KomaInterface.jl`.

## Citation

This project is prepared for Zenodo archiving. After the Zenodo release is
published, cite the archived release DOI shown by Zenodo.

Authors:

- Eros Montin, NYU Grossman School of Medicine
- José E. Cruz Seralles, NYU Grossman School of Medicine
- Steven Baete, NYU Grossman School of Medicine
- Giuseppe Carluccio, University of Napoli Federico II
- Riccardo Lattanzi, NYU Grossman School of Medicine

Citation metadata is provided in [`CITATION.cff`](CITATION.cff) and
[`.zenodo.json`](.zenodo.json).

## Requirements

- Python 3.9 or newer
- Julia installed and available on `PATH`
- Git access to `cloudmrhub/KomaInterface.jl`

CPU installation is supported and is the safest option for Colab or machines
without an NVIDIA GPU. GPU-capable installation additionally installs
`CUDA.jl`; `CUDA.functional()` reports whether GPU execution is actually
available at runtime.

## Install

From the repository root:

```bash
python -m pip install -e .
hash -r
```

Install the Julia dependency stack. For CPU-only systems or Colab runtimes
without a GPU, use:

```bash
camrie-install-julia --cpu
```

For GPU-capable systems with NVIDIA CUDA available, use:

```bash
camrie-install-julia
```

Both modes install:

- `KomaInterface.jl`

The GPU-capable mode also installs:

- `CUDA.jl`

To override the KomaInterface repository or branch:

```bash
camrie-install-julia \
  --repository-url https://github.com/cloudmrhub/KomaInterface.jl.git \
  --branch master
```

SSH-style GitHub URLs are also supported and normalized automatically for Julia.

To deliberately refresh the CAMRIE Julia project to the latest compatible
Julia package versions, use `--update`:

```bash
camrie-install-julia --cpu --update
```

On GPU-capable systems:

```bash
camrie-install-julia --update
```

This runs `Pkg.update()` followed by `Pkg.precompile()` inside the dedicated
CAMRIE Julia project. It is useful during release maintenance, but normal users
should usually omit `--update` for a more stable install.

To install Julia packages into a dedicated depot:

```bash
camrie-install-julia \
  --install-dir ~/.julia-camrie
```

Use that depot later with:

```bash
JULIA_DEPOT_PATH=~/.julia-camrie camrie-test-installation --cpu
```

## Test The Installation

Run the packaged installation check:

```bash
camrie-test-installation --cpu
```

This verifies:

- required Python dependencies import
- bundled Julia simulation script is present
- Julia is available on `PATH`
- `KomaInterface.jl` imports successfully
- `CUDA.jl` imports successfully unless `--cpu` is used

On a GPU-capable install without a usable GPU, the check can still pass. The
output may include:

```text
CUDA functional: false
```

That means CUDA is installed but no usable CUDA GPU/runtime is available.

## Example

Run the packaged example:

```bash
camrie-example
```

This checks that `camrie_tools` imports, locates the bundled Julia simulation
script, writes a tiny temporary Pulseq-style sequence, and parses it with the
pipeline reader.

Expected output includes:

```text
camrie_tools 0.1.0
Bundled Julia script: ...
Parsed example sequence:
  nF: 64
  nP: 4
  FOV mm: [220.0, 180.0]
```

## Reconstruction Smoke Test

Run a small end-to-end reconstruction smoke test:

```bash
camrie-reconstruction-smoke
```

This creates the same two-compartment concentric phantom used by the local
CAMRIE smoke workflow, simulates the bundled `PD-Weighted_Spin_Echo.seq`
sequence with KomaInterface/KomaMRI, reconstructs the k-space, and writes
outputs to a temporary directory. Expected output
includes:

```text
CAMRIE reconstruction smoke test passed.
k-space shape: [128, 256]
reconstruction shape: [128, 128]
Preview PNG: /tmp/.../reconstruction_preview.png
```

Open the printed `Preview PNG` path to inspect the reconstructed concentric
phantom image.

The default concentric phantom has two compartments:

```text
inner core: PD=1.0, T1=800 ms,  T2=60 ms
outer ring: PD=0.8, T1=1200 ms, T2=80 ms
```

You can change the tissue values from the command line:

```bash
camrie-reconstruction-smoke \
  --inner-pd 1.0 \
  --inner-t1-ms 900 \
  --inner-t2-ms 70 \
  --outer-pd 0.7 \
  --outer-t1-ms 1300 \
  --outer-t2-ms 90
```

The CLI and Python API accept T1/T2 in milliseconds to match the phantom
generator used by the CAMRIE application. The generated Koma phantom stores
those values in seconds internally.

The same values can be set from Python:

```python
from camrie_tools._reconstruction_smoke import run_reconstruction_smoke

summary = run_reconstruction_smoke(
    output_dir="/tmp/camrie_reconstruction_smoke",
    inner_pd=1.0,
    inner_t1_ms=900.0,
    inner_t2_ms=70.0,
    outer_pd=0.7,
    outer_t1_ms=1300.0,
    outer_t2_ms=90.0,
)
```

For a faster low-level debugging phantom, use:

```bash
camrie-reconstruction-smoke --phantom circle --grid-size 31
```

To keep automated test runs quick, the real Julia reconstruction smoke test is
opt-in:

```bash
CAMRIE_RUN_RECON_SMOKE=1 PYTHONPATH=src python -m unittest \
  tests.test_installation_smoke.InstallationSmokeTests.test_concentric_phantom_reconstruction_smoke
```

## Google Colab

The notebook entry point is:

```text
camrie_tools.ipynb
```

In Colab, first install Julia because the base Python runtime usually does not
include it:

```bash
!curl -fsSL https://install.julialang.org | sh -s -- -y
```

Then add Julia to the notebook process `PATH`:

```python
import os

os.environ["PATH"] = f"{os.path.expanduser('~')}/.juliaup/bin:" + os.environ["PATH"]
```

Install `camrie-tools` and the full Julia dependency stack:

```python
%pip install git+https://github.com/cloudmrhub/camrie-tools@v1
```

```bash
!camrie-install-julia --cpu
```

Then verify the installation and run the lightweight example:

```bash
!camrie-test-installation --cpu
!camrie-example
```

The notebook then runs the reconstruction smoke test from Python, loads
`reconstruction_magnitude.npy`, and displays the concentric phantom
reconstruction inline with Matplotlib.

## Developer Smoke Tests

Run the lightweight package tests:

```bash
PYTHONPATH=src python -m unittest tests.test_package_installation
```

Run the broader installation smoke tests:

```bash
PYTHONPATH=src python -m unittest tests.test_installation_smoke
```

Some functional tests are disabled by default because they require local sample
data and heavier runtime dependencies. Enable them with:

```bash
MAKEITKOMA_SMOKE_FULL=1 PYTHONPATH=src python -m unittest tests.test_installation_smoke
```
