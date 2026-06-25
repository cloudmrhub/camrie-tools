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

    def test_julia_installer_command_shape(self) -> None:
        from camrie_tools._julia import build_julia_command

        command = build_julia_command()

        self.assertEqual(command[:2], ["julia", "-e"])
        self.assertIn(
            'Pkg.PackageSpec(url="https://github.com/cloudmrhub/KomaInterface.jl.git", rev="master")',
            command[-1],
        )
        self.assertIn('Pkg.add(["CUDA"])', command[-1])
        self.assertIn("Pkg.update()", command[-1])
        self.assertIn("Pkg.precompile()", command[-1])

    def test_julia_installer_normalizes_ssh_repository_urls(self) -> None:
        from camrie_tools._julia import build_julia_command

        command = build_julia_command(
            repository_url="git@github.com:cloudmrhub/KomaInterface.jl.git",
            branch="master",
        )

        self.assertIn(
            'Pkg.PackageSpec(url="ssh://git@github.com/cloudmrhub/KomaInterface.jl.git", rev="master")',
            command[-1],
        )
        self.assertIn('Pkg.add(["CUDA"])', command[-1])

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
