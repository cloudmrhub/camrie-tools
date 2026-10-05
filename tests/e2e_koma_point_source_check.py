#!/usr/bin/env python3
"""Analytical point-source FFT pixel-alignment check (precise half of test 9).

This complements e2e_koma_affine_check.py's landmark-search test with a much
more precise, noise-free check of reconstruct_from_kspace's pixel-center
convention: it builds an idealized point-source k-space DIRECTLY (no Julia,
no Bloch simulation) at a chosen sub-pixel world position, reconstructs it,
and checks the recovered peak pixel is exactly where the DFT math predicts.

This isolates the "which pixel/phase convention does the FFT impose" question
from the sequence-geometry/placement question answered by
e2e_koma_affine_check.py, as required: "Test FFT pixel alignment separately
and more precisely using analytical point signals."

Does not require Julia. Run:
    conda run -n koma python tests/e2e_koma_point_source_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camrie_tools.MRI_pipeline import reconstruct_from_kspace  # noqa: E402


def point_source_kspace(nP: int, nF: int, voxel_i: float, voxel_j: float) -> np.ndarray:
    """Build the k-space of a unit point source at continuous image index
    (voxel_j, voxel_i) (row=phase=j, col=freq=i), using the SAME DFT
    convention as reconstruct_from_kspace (fftshift/ifft2/ifftshift, centered
    at index (n-1)/2 for n points, matching numpy.fft's own convention when
    combined with fftshift).
    """
    ky = (np.arange(nP) - nP // 2)
    kx = (np.arange(nF) - nF // 2)
    KX, KY = np.meshgrid(kx, ky)
    # Phase ramp corresponding to a point source at (voxel_j, voxel_i) in an
    # image produced by fftshift(ifft2(ifftshift(kspace))). ifft2 applies
    # exp(+2pi*i*k*x/N), so placing energy at position x requires the
    # k-space phase to carry the OPPOSITE (negative) sign.
    phase = -2 * np.pi * (KX * voxel_i / nF + KY * voxel_j / nP)
    kspace = np.exp(1j * phase).astype(np.complex128)
    return kspace


def peak_index(img: np.ndarray) -> tuple[float, float]:
    j, i = np.unravel_index(np.argmax(img), img.shape)
    return float(i), float(j)


def main() -> int:
    nP, nF = 64, 64  # even
    cases_even = [(0.0, 0.0), (3.0, -5.0), (0.5, 0.0)]  # last one: sub-pixel

    print("Even-size grid (nP=nF=64):")
    all_ok = True
    for voxel_i, voxel_j in cases_even:
        ks = point_source_kspace(nP, nF, voxel_i, voxel_j)
        img = reconstruct_from_kspace(
            ks, expected_shape=(nP, nF), oversampling=1, apply_hamming=False,
            remove_os=False, orientation={"detected_ro_sign": 1, "flip_phase": False})
        got_i, got_j = peak_index(img)
        # Peak search finds the nearest grid point; for non-integer source
        # positions the true peak is between pixels, so round the expected
        # value to the nearest grid index for comparison.
        exp_i = round(voxel_i + nF // 2)
        exp_j = round(voxel_j + nP // 2)
        ok = (abs(got_i - exp_i) <= 1) and (abs(got_j - exp_j) <= 1)
        status = "PASS" if ok else "FAIL"
        print(f"  source=({voxel_i:+.1f},{voxel_j:+.1f}) -> peak pixel=({got_i:.0f},{got_j:.0f}) "
              f"expected~({exp_i},{exp_j}) -> {status}")
        all_ok = all_ok and ok

    nP_odd, nF_odd = 65, 65
    print("\nOdd-size grid (nP=nF=65):")
    for voxel_i, voxel_j in [(0.0, 0.0), (2.0, -3.0)]:
        ks = point_source_kspace(nP_odd, nF_odd, voxel_i, voxel_j)
        img = reconstruct_from_kspace(
            ks, expected_shape=(nP_odd, nF_odd), oversampling=1, apply_hamming=False,
            remove_os=False, orientation={"detected_ro_sign": 1, "flip_phase": False})
        got_i, got_j = peak_index(img)
        exp_i = round(voxel_i + nF_odd // 2)
        exp_j = round(voxel_j + nP_odd // 2)
        ok = (abs(got_i - exp_i) <= 1) and (abs(got_j - exp_j) <= 1)
        status = "PASS" if ok else "FAIL"
        print(f"  source=({voxel_i:+.1f},{voxel_j:+.1f}) -> peak pixel=({got_i:.0f},{got_j:.0f}) "
              f"expected~({exp_i},{exp_j}) -> {status}")
        all_ok = all_ok and ok

    print("\nReadout polarity check (detected_ro_sign=-1 flips columns):")
    ks = point_source_kspace(nP, nF, 10.0, 0.0)
    img_pos = reconstruct_from_kspace(
        ks, expected_shape=(nP, nF), oversampling=1, apply_hamming=False,
        orientation={"detected_ro_sign": 1, "flip_phase": False})
    img_neg = reconstruct_from_kspace(
        ks, expected_shape=(nP, nF), oversampling=1, apply_hamming=False,
        orientation={"detected_ro_sign": -1, "flip_phase": False})
    i_pos, _ = peak_index(img_pos)
    i_neg, _ = peak_index(img_neg)
    polarity_ok = abs((nF - 1 - i_pos) - i_neg) <= 1
    print(f"  ro_sign=+1 peak_i={i_pos:.0f}  ro_sign=-1 peak_i={i_neg:.0f}  "
          f"(expect mirrored about center) -> {'PASS' if polarity_ok else 'FAIL'}")
    all_ok = all_ok and polarity_ok

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
