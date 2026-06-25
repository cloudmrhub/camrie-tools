#!/usr/bin/env python3
"""
inspect_kspace.py — visualise raw k-space from a debug simulation run.

Usage:
    python dev/inspect_kspace.py /tmp/tse_debug/kspace_000.npy
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def inspect(ks_path: str, out_dir: str = None):
    ks_path = Path(ks_path)
    out_dir = Path(out_dir) if out_dir else ks_path.parent

    kspace = np.load(ks_path)
    print(f"k-space shape : {kspace.shape}  dtype={kspace.dtype}")
    print(f"|kspace| max  : {np.abs(kspace).max():.4f}")
    print(f"|kspace| mean : {np.abs(kspace).mean():.4f}")

    nP, nF = kspace.shape

    # ── 1. Magnitude image (simple IFFT) ─────────────────────────────────────
    img = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace))
    )
    img_mag = np.abs(img)

    # ── 2. k-space profiles: magnitude along ky (column-averaged)  ───────────
    ky_profile   = np.abs(kspace).mean(axis=1)   # mean over frequency
    dc_col       = nF // 2
    ky_dc_column = np.abs(kspace[:, dc_col])      # DC column (kx=0) vs ky

    # ── 3. Detect periodicity in PE profile ──────────────────────────────────
    fft_ky = np.abs(np.fft.rfft(ky_profile - ky_profile.mean()))
    dominant_freq_idx = int(np.argmax(fft_ky[1:]) + 1)   # skip DC
    dominant_period   = nP / dominant_freq_idx if dominant_freq_idx else 0
    print(f"\nky-profile dominant period : {dominant_period:.1f} lines "
          f"(peak at frequency index {dominant_freq_idx})")
    if dominant_period > 1:
        print(f"  → If ETL={round(dominant_period)}, this is classic TSE T2-block modulation.")

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"k-space inspection: {ks_path.name}", fontsize=12)

    # (a) log-magnitude k-space
    ax = axes[0, 0]
    vmin, vmax = None, None
    ks_log = np.log1p(np.abs(np.fft.fftshift(kspace)))
    ax.imshow(ks_log, aspect="auto", cmap="gray")
    ax.set_title("log|k-space| (fftshifted)")
    ax.set_xlabel("kx (readout)"); ax.set_ylabel("ky (phase)")

    # (b) reconstructed image
    ax = axes[0, 1]
    ax.imshow(img_mag, cmap="gray",
              vmax=np.percentile(img_mag, 99))
    ax.set_title("Reconstructed image (raw IFFT)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # (c) ky magnitude profile — should be SMOOTH for SE, STEPPED for TSE
    ax = axes[1, 0]
    ax.plot(ky_profile, label="mean |ks| over kx")
    ax.plot(ky_dc_column, label="|ks[:, kx=0]|", alpha=0.7)
    ax.set_title("k-space magnitude profile (along ky)")
    ax.set_xlabel("ky row index"); ax.set_ylabel("|k-space|")
    ax.legend(fontsize=8)
    if dominant_period > 1:
        # Mark block boundaries
        period = int(round(dominant_period))
        for b in range(0, nP, period):
            ax.axvline(b, color="red", alpha=0.3, lw=0.8)
        ax.set_title(f"ky profile — period≈{period} (ETL blocks marked)")

    # (d) FFT of ky profile — spike at ETL frequency reveals block modulation
    ax = axes[1, 1]
    freqs = np.fft.rfftfreq(nP, d=1)
    ax.plot(freqs[1:], fft_ky[1:])
    ax.axvline(freqs[dominant_freq_idx], color="red",
               label=f"peak f={freqs[dominant_freq_idx]:.3f} → period={dominant_period:.1f}")
    ax.set_title("FFT of ky-profile (reveals periodic modulation)")
    ax.set_xlabel("spatial frequency (cycles / line)"); ax.set_ylabel("|FFT|")
    ax.legend(fontsize=8)

    out_png = out_dir / f"kspace_inspect_{ks_path.stem}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nPlot saved → {out_png}")
    return dominant_period


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tse_debug/kspace_000.npy"
    out  = sys.argv[2] if len(sys.argv) > 2 else None
    inspect(path, out)
