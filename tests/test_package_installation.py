"""Minimal package installation smoke tests.

These checks intentionally avoid the heavier MRI runtime dependencies so they
can run immediately after an editable install.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class CamrieToolsInstallationTests(unittest.TestCase):
    def test_import_and_package_data(self) -> None:
        import camrie_tools

        simulate_batch = Path(camrie_tools.simulate_batch_path())
        self.assertTrue(simulate_batch.exists())
        self.assertEqual(simulate_batch.name, "simulate_batch.jl")

        sequence = Path(camrie_tools.sequence_path())
        self.assertTrue(sequence.exists())
        self.assertEqual(sequence.suffix, ".seq")

    def test_root_and_packaged_julia_entrypoints_match(self) -> None:
        root_script = REPO_ROOT / "src" / "simulate_batch.jl"
        packaged_script = REPO_ROOT / "src" / "camrie_tools" / "simulate_batch.jl"

        self.assertTrue(root_script.exists())
        self.assertTrue(packaged_script.exists())
        self.assertEqual(root_script.read_text(), packaged_script.read_text())

    def test_julia_installer_command_shape(self) -> None:
        from camrie_tools._julia import build_julia_command

        command = build_julia_command(project_dir="/tmp/camrie-julia")

        self.assertEqual(command[0], "julia")
        self.assertIn("--project=/tmp/camrie-julia", command)
        self.assertIn("--startup-file=no", command)
        self.assertIn(
            'Pkg.PackageSpec(url="https://github.com/cloudmrhub/KomaInterface.jl.git", rev="master")',
            command[-1],
        )
        self.assertIn('Pkg.add("CUDA")', command[-1])
        self.assertIn("Pkg.activate", command[-1])
        self.assertIn("Pkg.instantiate()", command[-1])
        self.assertIn("Pkg.precompile()", command[-1])
        self.assertNotIn("Pkg.update()", command[-1])

    def test_julia_installer_normalizes_ssh_repository_urls(self) -> None:
        from camrie_tools._julia import build_julia_command

        command = build_julia_command(
            project_dir="/tmp/camrie-julia",
            repository_url="git@github.com:cloudmrhub/KomaInterface.jl.git",
            branch="master",
        )

        self.assertIn(
            'Pkg.PackageSpec(url="ssh://git@github.com/cloudmrhub/KomaInterface.jl.git", rev="master")',
            command[-1],
        )
        self.assertIn('Pkg.add("CUDA")', command[-1])

    def test_julia_installer_cpu_mode_skips_cuda(self) -> None:
        from camrie_tools._julia import build_julia_command

        command = build_julia_command(project_dir="/tmp/camrie-julia", cpu=True)

        self.assertIn("KomaInterface.jl.git", command[-1])
        self.assertNotIn("CUDA", command[-1])

    def test_julia_installer_update_mode_runs_pkg_update(self) -> None:
        from camrie_tools._julia import build_julia_command

        command = build_julia_command(project_dir="/tmp/camrie-julia", update=True)

        self.assertIn("Pkg.update()", command[-1])
        self.assertIn("Pkg.precompile()", command[-1])

    def test_packaged_example_sequence_is_parseable(self) -> None:
        from camrie_tools._example import EXAMPLE_SEQUENCE

        self.assertIn("[DEFINITIONS]", EXAMPLE_SEQUENCE)
        self.assertIn("[ADC]", EXAMPLE_SEQUENCE)
        self.assertIn("FOV 0.220 0.180 0.005", EXAMPLE_SEQUENCE)

    def test_circular_reconstruction_smoke_phantom_shape(self) -> None:
        from camrie_tools._reconstruction_smoke import create_circular_phantom

        phantom = create_circular_phantom(grid_size=11, radius_mm=45.0)

        self.assertGreater(len(phantom["rho"]), 0)
        self.assertEqual(len(phantom["x"]), len(phantom["rho"]))
        self.assertEqual(len(phantom["y"]), len(phantom["rho"]))
        self.assertEqual(len(phantom["z"]), len(phantom["rho"]))
        self.assertEqual(phantom["name"], "circular_smoke_phantom")

    def test_default_reconstruction_smoke_phantom_matches_local_test(self) -> None:
        import numpy as np

        from camrie_tools._reconstruction_smoke import create_concentric_cylinder_phantom

        phantom = create_concentric_cylinder_phantom(
            grid_size=31,
            inner_radius_mm=30.0,
            outer_radius_mm=60.0,
        )

        self.assertGreater(len(phantom["rho"]), 0)
        self.assertEqual(phantom["name"], "concentric_cylinder_smoke_phantom")
        self.assertTrue(np.isclose(phantom["rho"], 1.0).any())
        self.assertTrue(np.isclose(phantom["rho"], 0.8).any())
        self.assertTrue(np.isclose(phantom["t1"], 0.8).any())
        self.assertTrue(np.isclose(phantom["t1"], 1.2).any())
        self.assertTrue(np.isclose(phantom["t2"], 0.06).any())
        self.assertTrue(np.isclose(phantom["t2"], 0.08).any())
        self.assertEqual(phantom["slice_geometry"]["inner_radius_mm"], 30.0)
        self.assertEqual(phantom["slice_geometry"]["outer_radius_mm"], 60.0)
        self.assertEqual(phantom["slice_geometry"]["inner"]["t1_ms"], 800.0)
        self.assertEqual(phantom["slice_geometry"]["outer"]["t1_ms"], 1200.0)

    def test_reconstruction_preview_writer_creates_png(self) -> None:
        import tempfile

        import numpy as np
        from camrie_tools._reconstruction_smoke import save_reconstruction_preview

        image = np.eye(8, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = Path(tmpdir) / "preview.png"
            saved = Path(save_reconstruction_preview(image, preview_path))

            self.assertTrue(saved.exists())
            self.assertEqual(saved.suffix, ".png")
            self.assertGreater(saved.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
