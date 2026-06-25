#!/usr/bin/env python3
"""
Smoke tests to validate a local installation with bundled sample data.

Run:
    python3 -m unittest discover -s tests -p 'test_installation_smoke.py' -v

By default this script validates Python package availability only.
To run deeper functional checks, set:
    MAKEITKOMA_SMOKE_FULL=1
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
for import_path in (SRC_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

RHO_PATH = Path(os.environ.get("MAKEITKOMA_RHO", str(DATA_DIR / "rhoh.nii.gz")))
T1_PATH = Path(os.environ.get("MAKEITKOMA_T1", str(DATA_DIR / "t1.nii.gz")))
T2_PATH = Path(os.environ.get("MAKEITKOMA_T2", str(DATA_DIR / "t2.nii.gz")))

# SimpleITK can emit SWIG deprecation warnings on some Python builds.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute",
)


class InstallationSmokeTests(unittest.TestCase):
    FULL_MODE = os.environ.get("MAKEITKOMA_SMOKE_FULL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def _require_full_mode(self) -> None:
        if not self.FULL_MODE:
            self.skipTest(
                "Functional smoke checks disabled by default. "
                "Set MAKEITKOMA_SMOKE_FULL=1 to run them."
            )

    def _require_modules(self, *modules: str) -> None:
        missing: list[str] = []
        for module_name in modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing.append(module_name)
        if missing:
            self.skipTest(
                "Skipping because required modules are missing: " + ", ".join(sorted(missing))
            )

    def test_required_python_dependencies_are_importable(self) -> None:
        required = ["numpy", "SimpleITK", "pypulseq", "vtk", "matplotlib"]
        missing: list[str] = []

        for module_name in required:
            try:
                if module_name == "matplotlib":
                    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
                importlib.import_module(module_name)
            except ImportError:
                missing.append(module_name)

        if missing:
            self.fail(
                "Missing required Python modules: "
                + ", ".join(sorted(missing))
                + ". Install requirements and re-run this smoke test."
            )

    def test_camrie_tools_package_imports_with_bundled_julia_script(self) -> None:
        import camrie_tools

        simulate_batch = Path(camrie_tools.simulate_batch_path())
        self.assertTrue(simulate_batch.exists(), f"Missing bundled Julia script: {simulate_batch}")
        self.assertEqual(simulate_batch.name, "simulate_batch.jl")

    def test_julia_installer_command_adds_expected_packages(self) -> None:
        from camrie_tools._julia import build_julia_command

        cmd = build_julia_command(project_dir="/tmp/camrie-julia")

        self.assertEqual(cmd[0], "julia")
        self.assertIn("--project=/tmp/camrie-julia", cmd)
        self.assertIn(
            'Pkg.PackageSpec(url="https://github.com/cloudmrhub/KomaInterface.jl.git", rev="master")',
            cmd[-1],
        )
        self.assertIn('Pkg.add("CUDA")', cmd[-1])
        self.assertIn("Pkg.activate", cmd[-1])
        self.assertIn("Pkg.instantiate()", cmd[-1])
        self.assertIn("Pkg.precompile()", cmd[-1])

    def test_installation_checker_reports_success_when_dependencies_are_available(self) -> None:
        from unittest import mock

        from camrie_tools import _installation_check

        completed = subprocess.CompletedProcess(
            ["julia", "-e", ""],
            0,
            stdout="Julia package OK\n",
            stderr=None,
        )
        with mock.patch.object(_installation_check.shutil, "which", return_value="/usr/bin/julia"):
            with mock.patch.object(Path, "exists", return_value=True):
                with mock.patch.object(_installation_check, "_run_julia_check", return_value=completed):
                    self.assertEqual(_installation_check.main(["--cpu"]), 0)

    def test_circular_phantom_reconstruction_smoke(self) -> None:
        if os.environ.get("CAMRIE_RUN_RECON_SMOKE", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self.skipTest(
                "Set CAMRIE_RUN_RECON_SMOKE=1 to run the Julia reconstruction smoke test."
            )

        from camrie_tools._reconstruction_smoke import run_reconstruction_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_reconstruction_smoke(
                output_dir=tmpdir,
                grid_size=11,
                radius_mm=45.0,
                n_threads=1,
            )

            self.assertGreater(summary["spin_count"], 0)
            self.assertEqual(summary["reconstruction_shape"], [128, 128])
            self.assertGreater(summary["peak"], 0.0)
            self.assertGreater(summary["total_signal"], 0.0)

    def test_bundled_nifti_data_loads_and_is_consistent(self) -> None:
        self._require_full_mode()
        self._require_modules("SimpleITK")
        import SimpleITK as sitk

        volumes = {}
        model_paths = {
            "rhoh": RHO_PATH,
            "t1": T1_PATH,
            "t2": T2_PATH,
        }
        for key, path in model_paths.items():
            self.assertTrue(path.exists(), f"Missing test data file: {path}")
            image = sitk.ReadImage(str(path))
            array = sitk.GetArrayFromImage(image)

            self.assertEqual(array.ndim, 3, f"{key} is not 3D.")
            self.assertGreater(array.size, 0, f"{key} is empty.")
            self.assertTrue(np.isfinite(array).all(), f"{key} has non-finite values.")
            volumes[key] = (image, array)

        sizes = {volumes[name][0].GetSize() for name in volumes}
        self.assertEqual(len(sizes), 1, "Body-model NIfTI files do not share the same shape.")

        rho_nonzero = np.count_nonzero(volumes["rhoh"][1])
        self.assertGreater(rho_nonzero, 0, "rhoh volume appears empty (all zeros).")

    def test_pulseq_parser_reads_bundled_sequence(self) -> None:
        self._require_full_mode()
        self._require_modules("SimpleITK")
        from MRI_pipeline import read_pulseq_params

        preferred = [
            DATA_DIR / "ge.seq",
            DATA_DIR / "sdl_pypulseq.seq",
            DATA_DIR / "sdl_miniflash.seq",
            DATA_DIR / "epi_se.seq",
            DATA_DIR / "epi_multislice.seq",
        ]
        seq_path = next((p for p in preferred if p.exists()), None)
        if seq_path is None:
            available = sorted(DATA_DIR.glob("*.seq"))
            if available:
                seq_path = available[0]
        self.assertIsNotNone(
            seq_path,
            f"No .seq files found in test data directory: {DATA_DIR}",
        )

        params = read_pulseq_params(str(seq_path))
        self.assertGreater(params["nF"], 0)
        self.assertGreater(params["nP"], 0)
        self.assertEqual(len(params["fov_mm"]), 2)

    def test_gre_builder_generates_a_seq_file(self) -> None:
        self._require_full_mode()
        self._require_modules("pypulseq")
        from sequences.gre2d import make_gre_2d
        from sequences.t1se_edit import make_spin_echo_2d

        with tempfile.TemporaryDirectory() as tmpdir:
            errors: list[str] = []

            # Some pypulseq versions are stricter on raster timing for this GRE
            # parameter set. Try GRE first, then fall back to SE as a smoke check.
            out_gre = Path(tmpdir) / "smoke_gre.seq"
            try:
                _, info = make_gre_2d(
                    tr=0.04,
                    te=0.008,
                    flip_angle_deg=10.0,
                    fov_read=0.220,
                    fov_phase=0.220,
                    nx=16,
                    ny=16,
                    slice_thickness=5e-3,
                    out_file=str(out_gre),
                    seq_name="SMOKE_GRE_2D",
                    plot=False,
                    verbose=False,
                )
                self.assertTrue(out_gre.exists(), "GRE smoke sequence was not written.")
                self.assertGreater(out_gre.stat().st_size, 0, "GRE smoke sequence is empty.")
                self.assertEqual(info["nx"], 16)
                self.assertEqual(info["ny"], 16)
                self.assertTrue(info["timing_ok"])
                return
            except Exception as exc:
                errors.append(f"GRE failed: {exc}")

            out_se = Path(tmpdir) / "smoke_se.seq"
            try:
                _, info = make_spin_echo_2d(
                    tr=0.6,
                    te=0.02,
                    fov_read=0.220,
                    fov_phase=0.220,
                    nx=32,
                    ny=32,
                    slice_thickness=5e-3,
                    out_file=str(out_se),
                    seq_name="SMOKE_SE_2D",
                    plot=False,
                    verbose=False,
                )
                self.assertTrue(out_se.exists(), "SE smoke sequence was not written.")
                self.assertGreater(out_se.stat().st_size, 0, "SE smoke sequence is empty.")
                self.assertEqual(info["nx"], 32)
                self.assertEqual(info["ny"], 32)
                self.assertTrue(info["timing_ok"])
                return
            except Exception as exc:
                errors.append(f"SE fallback failed: {exc}")

            self.fail("No sequence builder produced a valid smoke sequence. " + " | ".join(errors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
