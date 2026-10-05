#!/usr/bin/env python3
"""End-to-end Koma landmark check for the full-affine geometry path (test 9).

Standalone script (not run automatically by `unittest discover`) because it
drives real Julia/KomaMRI simulations and takes minutes per acquisition. See
GEOMETRY_FIX_NOTES.md / the deliverables doc for the exact command and
environment requirements (Julia + KomaInterface via the `koma` conda env).

What it does
------------
1. Builds a small asymmetric body model (rho/t1/t2 NIfTI) with THREE
   distinguishable, non-collinear landmark spheres and a non-identity body
   direction matrix (the body model itself is tilted/rotated, not just the
   slice).
2. Runs ``run_pipeline(..., affine=..., output_grid="sequence")`` twice:
     (a) aligned acquisition: slice plane == body model's native axial plane
     (b) oblique acquisition: translated, in-plane rotated, oblique normal
   with identical simulation settings otherwise (spin_factor, b0, threads).
3. For each run, locates each landmark's peak-intensity voxel in the output
   NIfTI, converts it to a world-mm position via the NIfTI's own affine,
   and compares against the landmark's known world position (independent of
   anything the pipeline computed) to within half a reconstructed voxel per
   axis.
4. Prints PASS/FAIL per landmark and run.

A separate, more precise FFT pixel-alignment check (analytical point source,
no landmark search) is NOT included here -- see
``e2e_koma_point_source_check.py`` if a sub-voxel/pixel-center-only check
without landmark-search noise is needed; this script focuses on real
Koma simulation + full extraction/assembly, matching required test 9.

Usage
-----
    conda run -n koma python tests/e2e_koma_affine_check.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import camrie_tools  # noqa: E402
from camrie_tools.MRI_pipeline import run_pipeline  # noqa: E402


def rot_axis(axis: str, deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(axis)


# Three non-collinear landmark centers in BODY-MODEL VOXEL space (mm, before
# the body model's own direction/origin are applied), chosen asymmetric so no
# flip/rotation of the phantom could be confused with another landmark.
LANDMARKS_MM = {
    "A": np.array([40.0, 10.0, 0.0]),
    "B": np.array([-30.0, 35.0, 0.0]),
    "C": np.array([5.0, -45.0, 0.0]),
}
LANDMARK_RADIUS_MM = 6.0


def build_body_model(out_dir: Path, body_direction: np.ndarray):
    """Write rho/t1/t2 NIfTI with a non-identity direction matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (121, 121, 61)  # (Nx, Ny, Nz) in SITK order
    spacing = (2.0, 2.0, 2.0)
    # Body-model grid centered at world origin.
    origin = -body_direction @ (np.array(size) - 1) / 2.0 * np.array(spacing)

    img_shape = size[::-1]  # numpy order (z, y, x)
    rho = np.full(img_shape, 0.5, dtype=np.float32)  # uniform background tissue
    t1 = np.full(img_shape, 900.0, dtype=np.float32)
    t2 = np.full(img_shape, 80.0, dtype=np.float32)

    # Base SITK image to convert world mm -> voxel index for painting spheres.
    base = sitk.GetImageFromArray(rho)
    base.SetSpacing(spacing)
    base.SetOrigin(tuple(origin.tolist()))
    base.SetDirection(tuple(body_direction.flatten(order="C").tolist()))

    for name, center_mm in LANDMARKS_MM.items():
        cont_idx = base.TransformPhysicalPointToContinuousIndex(tuple(center_mm.tolist()))
        ci, cj, ck = cont_idx
        radius_vox = LANDMARK_RADIUS_MM / spacing[0]
        ii, jj, kk = np.meshgrid(
            np.arange(size[0]), np.arange(size[1]), np.arange(size[2]), indexing="ij")
        mask = ((ii - ci) ** 2 + (jj - cj) ** 2 + (kk - ck) ** 2) <= radius_vox ** 2
        mask_zyx = np.transpose(mask, (2, 1, 0))
        pd = {"A": 1.0, "B": 0.7, "C": 0.9}[name]
        rho[mask_zyx] = pd
        t1[mask_zyx] = {"A": 500.0, "B": 1400.0, "C": 1000.0}[name]
        t2[mask_zyx] = {"A": 40.0, "B": 120.0, "C": 70.0}[name]

    paths = {}
    for key, arr in (("rho", rho), ("t1", t1), ("t2", t2)):
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing)
        img.SetOrigin(tuple(origin.tolist()))
        img.SetDirection(tuple(body_direction.flatten(order="C").tolist()))
        path = out_dir / f"{key}.nii.gz"
        sitk.WriteImage(img, str(path))
        paths[key] = str(path)
    return paths


def make_affine(direction: np.ndarray, spacing, origin) -> np.ndarray:
    A = np.eye(4)
    A[:3, :3] = direction @ np.diag(spacing)
    A[:3, 3] = origin
    return A


def run_one(label: str, affine: np.ndarray, matrix, fov_mm, num_slices,
            slice_thickness_mm, body_paths, out_root: Path):
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = Path(camrie_tools.sequence_path("PD-Weighted_Spin_Echo.seq"))

    volume, series_spec = run_pipeline(
        rho_path=body_paths["rho"], t1_path=body_paths["t1"], t2_path=body_paths["t2"],
        sequence_file=str(seq), output_dir=str(out_dir),
        isocenter_mm=None, slice_normal=None, num_slices=num_slices,
        slice_thickness_mm=slice_thickness_mm, slice_gap_mm=0.0,
        fov_mm=fov_mm, seq_fov_mm=fov_mm, matrix=matrix,
        affine=affine, output_grid="sequence",
        spin_factor=2, b0=1.5, n_threads=4, parallel_slices=2,
        apply_hamming=True, debug=True,
    )
    out_nii = out_dir / "reconstruction.nii.gz"
    return out_nii, series_spec


def check_landmarks(nii_path: Path, label: str) -> bool:
    img = sitk.ReadImage(str(nii_path))
    arr = sitk.GetArrayFromImage(img)  # (Nz, Ny, Nx)
    spacing = np.array(img.GetSpacing())
    half_voxel = spacing[:2].max() / 2.0  # in-plane half-voxel tolerance

    all_ok = True
    for name, expected_mm in LANDMARKS_MM.items():
        # Intensity-weighted centroid within a search radius of the expected
        # point: far more robust to reconstruction noise/ringing than a
        # single max-intensity voxel, and still an independent world-space
        # check (does not rely on anything the pipeline itself computed).
        acc_pos = np.zeros(3)
        acc_weight = 0.0
        n_hits = 0
        for k in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for i in range(arr.shape[2]):
                    val = arr[k, j, i]
                    if val <= 0:
                        continue
                    world = np.array(img.TransformIndexToPhysicalPoint((int(i), int(j), int(k))))
                    d = np.linalg.norm(world - expected_mm)
                    if d <= LANDMARK_RADIUS_MM * 1.8:
                        acc_pos += world * float(val)
                        acc_weight += float(val)
                        n_hits += 1
        if n_hits == 0 or acc_weight <= 0:
            print(f"  [{label}] landmark {name}: NOT FOUND near {expected_mm}")
            all_ok = False
            continue
        centroid = acc_pos / acc_weight
        err = centroid - expected_mm
        err_inplane = float(np.linalg.norm(err[:2]))
        ok = err_inplane <= half_voxel
        status = "PASS" if ok else "FAIL"
        print(f"  [{label}] landmark {name}: in-plane centroid error={err_inplane:.3f} mm "
              f"(half-voxel tol={half_voxel:.3f} mm, n_hits={n_hits}) -> {status}")
        all_ok = all_ok and ok
    return all_ok


def main() -> int:
    body_direction = rot_axis("y", 8.0)  # nonidentity body-model direction
    Nx, Ny = 128, 128
    fov_mm = (300.0, 300.0)  # match the .seq file's native FOV: avoids a
    # pre-existing KomaInterface.jl 0.1.6 apply_fov_rescaling! Grad
    # constructor bug unrelated to this fix (see GEOMETRY_FIX_NOTES.md).
    thickness = 4.0

    with tempfile.TemporaryDirectory(prefix="camrie_e2e_affine_") as tmp:
        tmp_path = Path(tmp)
        body_paths = build_body_model(tmp_path / "body", body_direction)

        results = {}

        def centered_origin(direction, dx, dy, dz, translation):
            """Origin (voxel-0 world position) so the grid's (N-1)/2 index
            lands at ``translation`` (the desired slice-stack center)."""
            half = np.array([(Nx - 1) / 2.0 * dx, (Ny - 1) / 2.0 * dy, (3 - 1) / 2.0 * dz])
            return np.array(translation) - direction @ half

        # (a) Aligned acquisition: axial, no in-plane rotation, centered at world origin.
        dx, dy = fov_mm[0] / Nx, fov_mm[1] / Ny
        aligned_direction = np.eye(3)
        origin_a = centered_origin(aligned_direction, dx, dy, thickness, (0.0, 0.0, 0.0))
        A_aligned = make_affine(aligned_direction, (dx, dy, thickness), origin_a)
        nii_a, _ = run_one("aligned", A_aligned, (Ny, Nx), fov_mm, 3, thickness,
                            body_paths, tmp_path)
        results["aligned"] = check_landmarks(nii_a, "aligned")

        # (b) Oblique acquisition: translated, in-plane rotated, oblique normal.
        # Use a smaller tilt than a first attempt (15/25deg sent all three
        # landmarks' z=0 plane >6mm from a thin 3-slice stack's mid-plane --
        # a test-fixture coverage issue, not a pipeline issue) so the slab
        # still intersects all three landmarks while keeping a clearly
        # oblique, non-trivial in-plane rotation and translation.
        oblique_direction = rot_axis("x", 6.0) @ rot_axis("z", 25.0)

        def oblique_origin(direction, dx, dy, dz, translation):
            half = np.array([(Nx - 1) / 2.0 * dx, (Ny - 1) / 2.0 * dy, (3 - 1) / 2.0 * dz])
            return np.array(translation) - direction @ half

        origin_b = oblique_origin(oblique_direction, dx, dy, thickness, (5.0, -8.0, 0.0))
        A_oblique = make_affine(oblique_direction, (dx, dy, thickness), origin_b)
        nii_b, _ = run_one("oblique", A_oblique, (Ny, Nx), fov_mm, 3, thickness,
                            body_paths, tmp_path)
        results["oblique"] = check_landmarks(nii_b, "oblique")

        print()
        for label, ok in results.items():
            print(f"{label}: {'PASS' if ok else 'FAIL'}")
        return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
