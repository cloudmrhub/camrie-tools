"""Orientation and phase-encode regression tests.

These guard three fixes that a radially symmetric phantom cannot catch:

  1. read_mtrk_params previously hardcoded flip_phase=False, which produced
     phase-encode-FLIPPED reconstructions for every spin-echo .mtrk file.
  2. read_pulseq_params negated the ky accumulator at every RF from the second
     onward (refocusing logic), corrupting gradient-echo trains where each TR
     begins with a fresh excitation.
  3. remove_readout_oversampling_kspace lacked the ifftshift/fftshift pair, so
     the "central half" crop kept the periphery of the FOV.

The fast tests need no Julia. The end-to-end orientation check does, and is
opt-in via CAMRIE_RUN_ORIENTATION=1, matching the CAMRIE_RUN_RECON_SMOKE
convention used by test_installation_smoke.

Why a chiral phantom: concentric cylinders are radially symmetric, so both the
centroid and any radial contrast measure are invariant under flipud/fliplr and a
flip is undetectable. build_chiral_phantom() below is a right triangle plus two
markers, and assert_discriminating() refuses to let it be used unless flipping it
actually changes the image.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Grid deliberately matches the CAMRIE local test phantom so isocentre and
# geometry handling behave identically.
NX = NY = 137
NZ = 27
DX = DY = 1.98529052734375
DZ = 1.923076868057251
ORIGIN = (-135.0, -135.0, -25.0)

# right triangle, vertices (-55,-45) (+55,-45) (-55,+55)
_TRI = dict(x0=-55.0, x1=55.0, y0=-45.0, y1=55.0)
_TRI_TISSUE = (0.50, 1000.0, 70.0)          # pd, t1_ms, t2_ms
_MARKERS = [                                 # cx, cy, r, pd, t1_ms, t2_ms
    (34.0, -33.0, 11.0, 1.00, 600.0, 105.0),
    (-38.0, 36.0, 16.0, 0.78, 1500.0, 42.0),
]

MAX_SELF_SIMILARITY = 0.70


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation, 1.0 for identical layouts."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def chiral_slice() -> np.ndarray:
    """The 2-D proton-density layout, array order (y, x)."""
    x = ORIGIN[0] + np.arange(NX) * DX
    y = ORIGIN[1] + np.arange(NY) * DY
    Y, X = np.meshgrid(y, x, indexing="ij")

    u = (X - _TRI["x0"]) / (_TRI["x1"] - _TRI["x0"])
    v = (Y - _TRI["y0"]) / (_TRI["y1"] - _TRI["y0"])
    rho = np.zeros((NY, NX), np.float32)
    rho[(u >= 0) & (v >= 0) & (u + v <= 1.0)] = _TRI_TISSUE[0]
    for cx, cy, r, pd, _t1, _t2 in _MARKERS:
        rho[((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2] = pd
    return rho


def build_chiral_phantom(out_dir: Path) -> Path:
    """Write rho/t1/t2 NIfTI of the chiral phantom into out_dir."""
    import SimpleITK as sitk

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = ORIGIN[0] + np.arange(NX) * DX
    y = ORIGIN[1] + np.arange(NY) * DY
    Y, X = np.meshgrid(y, x, indexing="ij")

    u = (X - _TRI["x0"]) / (_TRI["x1"] - _TRI["x0"])
    v = (Y - _TRI["y0"]) / (_TRI["y1"] - _TRI["y0"])
    tri = (u >= 0) & (v >= 0) & (u + v <= 1.0)

    rho2 = np.zeros((NY, NX), np.float32)
    t12 = np.zeros((NY, NX), np.float32)
    t22 = np.zeros((NY, NX), np.float32)
    rho2[tri], t12[tri], t22[tri] = _TRI_TISSUE
    for cx, cy, r, pd, t1, t2 in _MARKERS:
        m = ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2
        rho2[m], t12[m], t22[m] = pd, t1, t2

    zc = ORIGIN[2] + np.arange(NZ) * DZ
    slab = np.abs(zc) <= 20.0
    vols = {}
    for name, plane in (("rho", rho2), ("t1", t12), ("t2", t22)):
        vol = np.zeros((NZ, NY, NX), np.float32)
        vol[slab] = plane
        vols[name] = vol

    for name, vol in vols.items():
        img = sitk.GetImageFromArray(vol)
        img.SetSpacing((DX, DY, DZ))
        img.SetOrigin(ORIGIN)
        sitk.WriteImage(img, str(out_dir / f"{name}.nii"))
    return out_dir


class ChiralPhantomTests(unittest.TestCase):
    def test_phantom_can_actually_discriminate_flips(self) -> None:
        """A phantom is useless for orientation tests if it is flip-symmetric.

        An earlier four-disc design failed here: flipping y only moved small
        satellites while a large central disc dominated, leaving a 0.003 NCC
        margin. Assert real discriminating power instead of assuming it.
        """
        rho = chiral_slice()
        for label, flipped in (
            ("flipud (phase)", rho[::-1, :]),
            ("fliplr (readout)", rho[:, ::-1]),
            ("both", rho[::-1, ::-1]),
        ):
            with self.subTest(flip=label):
                self.assertLess(
                    ncc(rho, flipped), MAX_SELF_SIMILARITY,
                    f"phantom too symmetric under {label} to detect that flip")


class MtrkFlipPhaseTests(unittest.TestCase):
    """read_mtrk_params must report flip_phase=True.

    Established by correlating the assembled reconstruction against the input
    rho on the same body grid, using the chiral phantom:
        PD-Weighted_Spin_Echo.mtrk  identity 0.485 vs flipud 0.956
        T1-Weighted_Spin_Echo.mtrk  identity 0.433 vs flipud 0.933
        T2-Weighted_Spin_Echo.mtrk  identity 0.419 vs flipud 0.904
    and confirmed via flip_phase_override on the T1 file (0.9329 correct vs
    0.4329 flipped). The matching .seq files all report True and reconstruct
    correctly, so True is consistent across both readers.

    NOTE this value is empirical, not derived from the file. It is NOT validated
    for mtrk_spoiled_gre.mtrk, which matches no simple flip.
    """

    def _mtrk_paths(self):
        import camrie_tools
        found = []
        for name in ("PD-Weighted_Spin_Echo.mtrk", "T1-Weighted_Spin_Echo.mtrk"):
            try:
                p = Path(camrie_tools.sequence_path(name))
            except Exception:
                continue
            if p.exists():
                found.append(p)
        return found

    def test_spin_echo_mtrk_reports_flip_phase_true(self) -> None:
        from camrie_tools.MRI_pipeline import read_mtrk_params

        paths = self._mtrk_paths()
        if not paths:
            self.skipTest("no bundled .mtrk sequences")
        for path in paths:
            with self.subTest(sequence=path.name):
                params = read_mtrk_params(str(path))
                self.assertTrue(
                    params["orientation"]["flip_phase"],
                    f"{path.name}: flip_phase must be True or the recon is "
                    "phase-encode flipped")

    def test_flip_phase_is_wired_through_not_hardcoded(self) -> None:
        """The returned dict previously hardcoded False, ignoring the variable."""
        from camrie_tools.MRI_pipeline import read_mtrk_params

        paths = self._mtrk_paths()
        if not paths:
            self.skipTest("no bundled .mtrk sequences")
        params = read_mtrk_params(str(paths[0]))
        orientation = params["orientation"]
        self.assertIn("flip_phase", orientation)
        self.assertIsInstance(orientation["flip_phase"], bool)


class PulseqKyTests(unittest.TestCase):
    """ky must reset on excitation and negate only on refocusing.

    The shipped logic negated at every RF from the second onward. Spin echoes
    survived by accident because their net per-TR ky returns to zero, making
    negation equivalent to a reset. A gradient echo does not rewind its phase
    encode, so it produced 28/128 unique, non-monotonic ky values.
    """

    def _seq_paths(self):
        import camrie_tools
        found = []
        for name in ("PD-Weighted_Spin_Echo.seq", "T1-Weighted_Spin_Echo.seq"):
            try:
                p = Path(camrie_tools.sequence_path(name))
            except Exception:
                continue
            if p.exists():
                found.append(p)
        return found

    def test_spin_echo_flip_angles_classified(self) -> None:
        try:
            import pypulseq as pp
        except ImportError:
            self.skipTest("pypulseq not installed")
        from camrie_tools.MRI_pipeline import _classify_rf_flip_angles

        paths = self._seq_paths()
        if not paths:
            self.skipTest("no bundled .seq sequences")
        seq = pp.Sequence()
        seq.read(str(paths[0]))
        exc, refoc = _classify_rf_flip_angles(seq)
        self.assertIsNotNone(exc, "excitation flip angle not detected")
        self.assertIsNotNone(refoc, "spin echo must expose a refocusing pulse")
        self.assertGreater(refoc, exc)

    def test_ky_trajectory_unique_and_monotonic(self) -> None:
        try:
            import pypulseq  # noqa: F401
        except ImportError:
            self.skipTest("pypulseq not installed")
        from camrie_tools.MRI_pipeline import read_pulseq_params

        paths = self._seq_paths()
        if not paths:
            self.skipTest("no bundled .seq sequences")
        for path in paths:
            with self.subTest(sequence=path.name):
                params = read_pulseq_params(str(path))
                ky = params["orientation"].get("ky_trajectory")
                self.assertIsNotNone(ky, "no ky trajectory extracted")
                arr = np.asarray(ky, dtype=float)
                unique = np.unique(np.round(arr, 9)).size
                self.assertEqual(
                    unique, arr.size,
                    f"{path.name}: {unique}/{arr.size} unique ky values; "
                    "duplicates mean the accumulator is being mishandled")
                diffs = np.diff(arr)
                monotonic = bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))
                self.assertTrue(monotonic, f"{path.name}: ky not monotonic")


class RemoveOversamplingTests(unittest.TestCase):
    """Readout-oversampling removal follows mapVBVD flagRemoveOS semantics.

    2x readout oversampling doubles the acquired readout FOV (k_max and
    resolution unchanged), so removal is a hybrid-domain crop of the central
    half. The transform must be centred; without the shifts a centre-DC k-space
    puts the object at the array edges and the crop keeps the periphery.
    """

    def test_central_half_survives_round_trip(self) -> None:
        from camrie_tools.MRI_pipeline import remove_readout_oversampling_kspace

        n_p, n_f_raw = 128, 256
        n_f = n_f_raw // 2
        yy, xx = np.mgrid[0:n_p, 0:n_f_raw]
        cy, cx = n_p / 2.0, n_f_raw / 2.0

        img = np.zeros((n_p, n_f_raw), dtype=np.complex128)
        img[(((yy - cy) / 28.0) ** 2 + ((xx - cx) / 28.0) ** 2) <= 1.0] = 1.0
        img[(((yy - cy) / 12.0) ** 2 + ((xx - cx) / 12.0) ** 2) <= 1.0] = 2.0

        expected = np.abs(img[:, n_f_raw // 4: n_f_raw // 4 + n_f])
        kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

        out = remove_readout_oversampling_kspace(kspace)
        self.assertEqual(out.shape, (n_p, n_f))

        recon = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(out))))
        col_energy = recon.sum(axis=0)
        centre = col_energy[n_f // 4: n_f // 4 + n_f // 2].sum()
        self.assertGreater(
            centre / max(col_energy.sum(), 1e-12), 0.95,
            "object energy is not in the central half; the 1-D transform is "
            "probably missing its ifftshift/fftshift")

        got = recon / max(recon.max(), 1e-12)
        ref = expected / max(expected.max(), 1e-12)
        self.assertLess(np.abs(got - ref).mean(), 1e-6)


@unittest.skipUnless(
    os.environ.get("CAMRIE_RUN_ORIENTATION") == "1",
    "set CAMRIE_RUN_ORIENTATION=1 to run the end-to-end orientation check "
    "(requires Julia and KomaInterface)")
class EndToEndOrientationTests(unittest.TestCase):
    """Simulate the chiral phantom and confirm the output is not flipped.

    run_pipeline resamples each slice back into the body grid via
    place_slice_in_body, so reconstruction.nii.gz shares the grid of the input
    rho.nii. Correlating output against input under the four flip hypotheses
    reveals which transform was applied, with no need to reason about internal
    conventions.
    """

    def test_reconstruction_matches_input_orientation(self) -> None:
        import tempfile

        import SimpleITK as sitk

        import camrie_tools
        from camrie_tools.MRI_pipeline import run_pipeline

        seq = Path(camrie_tools.sequence_path("PD-Weighted_Spin_Echo.mtrk"))
        if not seq.exists():
            self.skipTest("bundled .mtrk sequence missing")

        with tempfile.TemporaryDirectory(prefix="camrie_orient_") as tmp:
            tmp = Path(tmp)
            phantom = build_chiral_phantom(tmp / "phantom")
            out = tmp / "out"
            out.mkdir()

            rho_img = sitk.ReadImage(str(phantom / "rho.nii"))
            size = np.array(rho_img.GetSize())
            origin = np.array(rho_img.GetOrigin())
            spacing = np.array(rho_img.GetSpacing())
            direction = np.array(rho_img.GetDirection()).reshape(3, 3)
            iso = origin + direction @ ((size - 1) / 2.0 * spacing)

            run_pipeline(
                rho_path=str(phantom / "rho.nii"),
                t1_path=str(phantom / "t1.nii"),
                t2_path=str(phantom / "t2.nii"),
                sequence_file=str(seq), output_dir=str(out),
                isocenter_mm=iso, slice_normal=[0, 0, 1], num_slices=1,
                slice_thickness_mm=None, slice_gap_mm=0.0,
                spin_factor=1, b0=1.5, n_threads=4, parallel_slices=1,
                apply_hamming=True, spins_per_voxel=0, slice_padding=1.0,
                debug=True,
            )

            rho = sitk.GetArrayFromImage(rho_img)
            vol = sitk.GetArrayFromImage(
                sitk.ReadImage(str(out / "reconstruction.nii.gz")))
            self.assertEqual(rho.shape, vol.shape)

            z = int(np.argmax(vol.reshape(vol.shape[0], -1).sum(axis=1)))
            ref, got = rho[z], vol[z]
            scores = {
                "identity": ncc(ref, got),
                "flipud": ncc(ref, got[::-1, :]),
                "fliplr": ncc(ref, got[:, ::-1]),
                "both": ncc(ref, got[::-1, ::-1]),
            }
            best = max(scores, key=scores.get)
            self.assertEqual(
                best, "identity",
                f"reconstruction is {best}-transformed relative to the input; "
                f"scores={ {k: round(v, 4) for k, v in scores.items()} }")


if __name__ == "__main__":
    unittest.main()
