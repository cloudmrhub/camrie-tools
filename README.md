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

This creates a circular spin phantom, simulates the bundled
`T1-Weighted_Spin_Echo.seq` sequence with KomaInterface/KomaMRI, reconstructs
the k-space, and writes outputs to a temporary directory. Expected output
includes:

```text
CAMRIE reconstruction smoke test passed.
k-space shape: [128, 256]
reconstruction shape: [128, 128]
```

To keep automated test runs quick, the real Julia reconstruction smoke test is
opt-in:

```bash
CAMRIE_RUN_RECON_SMOKE=1 PYTHONPATH=src python -m unittest \
  tests.test_installation_smoke.InstallationSmokeTests.test_circular_phantom_reconstruction_smoke
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
`reconstruction_magnitude.npy`, and displays the circular phantom
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
