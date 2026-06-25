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

    def test_julia_installer_command_shape(self) -> None:
        from camrie_tools._julia import build_julia_command

        self.assertIn('Pkg.add(["KomaInterface"])', build_julia_command(gpu=False)[-1])
        self.assertIn('Pkg.add(["KomaInterface", "CUDA"])', build_julia_command(gpu=True)[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
