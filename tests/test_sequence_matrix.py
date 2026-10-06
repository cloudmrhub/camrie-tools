"""Tests for the sequence-as-template path (configure_sequence).

The configurer round-trip test needs a Pulseq spin-echo template in the
IXIguy layout; point CAMRIE_TEST_SE_SEQ at one to enable it.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import camrie_tools  # noqa: E402
from camrie_tools.MRI_pipeline import _affine_for_matrix, _configure_sequence  # noqa: E402


class AffineForMatrixTests(unittest.TestCase):
    def test_keeps_centre_direction_and_sets_fov_over_matrix(self):
        r = np.deg2rad(30)
        d = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        A = np.eye(4)
        A[:3, :3] = d @ np.diag([1.0, 1.0, 2.0])
        A[:3, 3] = [10, -20, 30]
        presc, actual, fov = (311, 611), (192, 192), (611.0, 311.0)  # (Ny, Nx)
        B = _affine_for_matrix(A, presc, actual, fov)
        np.testing.assert_allclose(np.linalg.norm(B[:3, :3], axis=0), [611 / 192, 311 / 192, 2.0])
        np.testing.assert_allclose(B[:3, :3] / np.linalg.norm(B[:3, :3], axis=0), d, atol=1e-12)
        ca = A @ [(611 - 1) / 2, (311 - 1) / 2, 0, 1]
        cb = B @ [(192 - 1) / 2, (192 - 1) / 2, 0, 1]
        np.testing.assert_allclose(ca[:3], cb[:3], atol=1e-9)


class ConfigureSequenceTests(unittest.TestCase):
    def test_unsupported_template_falls_back_with_reason(self):
        seq = Path(camrie_tools.sequence_path("T1-Weighted_Spin_Echo.seq"))
        with tempfile.TemporaryDirectory() as tmp:
            out, rep = _configure_sequence(str(seq), (64, 96), (240.0, 160.0), tmp)
        self.assertEqual(out, str(seq))
        self.assertEqual(rep["mode"], "as_is")
        self.assertTrue(rep["reason"])

    def test_mtrk_falls_back(self):
        seq = Path(camrie_tools.sequence_path("T1-Weighted_Spin_Echo.mtrk"))
        with tempfile.TemporaryDirectory() as tmp:
            out, rep = _configure_sequence(str(seq), (64, 96), (240.0, 160.0), tmp)
        self.assertEqual(rep["mode"], "as_is")

    @unittest.skipUnless(os.environ.get("CAMRIE_TEST_SE_SEQ"), "set CAMRIE_TEST_SE_SEQ to a Pulseq SE template")
    def test_configured_sequence_has_requested_matrix_and_same_timing(self):
        import pypulseq as pp
        from camrie_tools.MRI_pipeline import read_sequence_params

        src = os.environ["CAMRIE_TEST_SE_SEQ"]
        p0 = read_sequence_params(src)
        with tempfile.TemporaryDirectory() as tmp:
            out, rep = _configure_sequence(src, (p0["nP"], p0["nF"]), tuple(p0["fov_mm"]), tmp)
            a = pp.Sequence(); a.read(src)
            b = pp.Sequence(); b.read(out)
            self.assertEqual(rep["mode"], "configured")
            self.assertAlmostEqual(a.duration()[0], b.duration()[0], places=6)  # identity keeps TR/TE
            out2, rep2 = _configure_sequence(src, (128, 100), (200.0, 150.0), tmp)
            p = read_sequence_params(out2)
            self.assertEqual((p["nP"], p["nF"]), (128, 100))
            self.assertEqual(list(p["fov_mm"]), [200.0, 150.0])
            self.assertAlmostEqual(float(pp.Sequence().definitions.get("TR", 0) or 0), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
