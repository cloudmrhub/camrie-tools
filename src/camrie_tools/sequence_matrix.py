"""Scanner-style prescription of a Pulseq spin-echo template.

The template file supplies everything except the sampling geometry (RF, TE, TR,
crushers, slice selection, timing). ``rebuild_pulseq_matrix`` re-prescribes
the number of readout samples (nx), phase-encode lines (ny) and the in-plane
FOV, as a scanner operator would, by:

* setting the ADC sample count to ``nx`` over the SAME acquisition window
  (dwell = window / nx), so block durations, TE and TR are unchanged;
* scaling the readout gradient so its area is ``nx / fov_x``, and adjusting the
  pre/rewinder x-areas by -(dA)/2 so the echo stays centred;
* regenerating linear phase-encode steps ``(i - ny/2) / fov_y`` with ``ny`` TRs.

Supported: Pulseq spin echo with one ADC per TR and linear phase encoding.
Anything else raises ``NotImplementedError`` (turbo spin echo is not supported yet).
"""
from __future__ import annotations

import numpy as np
import pypulseq as pp


class SequenceRebuildError(ValueError):
    """The template cannot be re-prescribed consistently."""


def _ev(blk, name):
    return getattr(blk, name, None)


def _split_units(src):
    """Split blocks into TR units; a unit starts at each excitation RF."""
    idx = sorted(src.block_events)
    exc_id = None
    starts = []
    for i in idx:
        rf_id = int(np.asarray(src.block_events[i]).ravel()[1])
        if rf_id and exc_id is None:
            exc_id = rf_id
        if rf_id and rf_id == exc_id:
            starts.append(max(i - 1, idx[0]))  # TR begins one block before the RF
    if not starts:
        raise SequenceRebuildError("no RF excitation found in template")
    ends = starts[1:] + [idx[-1] + 1]
    return [list(range(s, e)) for s, e in zip(starts, ends)]


def rebuild_pulseq_matrix(seq_in, seq_out, nx, ny, fov_x_mm, fov_y_mm):
    src = pp.Sequence()
    src.read(seq_in)
    sysm = src.system
    blocks = {i: src.get_block(i) for i in sorted(src.block_events)}
    units = _split_units(src)

    has_adc = [any(_ev(blocks[i], "adc") is not None for i in u) for u in units]
    img = [u for u, a in zip(units, has_adc) if a]
    if not img:
        raise SequenceRebuildError("template has no ADC")
    if any(sum(_ev(blocks[i], "adc") is not None for i in u) != 1 for u in img):
        raise NotImplementedError("only one ADC per TR (spin echo) is supported")
    if any(len(u) != len(img[0]) for u in units):
        raise NotImplementedError("TR units differ in block count")

    tpl = img[0]
    pos = next(k for k, i in enumerate(tpl) if _ev(blocks[i], "adc") is not None)
    if pos == 0 or pos == len(tpl) - 1:
        raise SequenceRebuildError("expected pre/rewinder blocks around the ADC block")

    adc0 = _ev(blocks[tpl[pos]], "adc")
    nx0, ny0 = int(adc0.num_samples), len(img)
    fov0 = np.asarray(src.definitions["FOV"], dtype=float)
    gro = _ev(blocks[tpl[pos]], "gx")
    a_ro0 = float(gro.area)
    if abs(a_ro0 - nx0 / fov0[0]) > 0.02 * a_ro0:
        raise SequenceRebuildError(
            f"readout area {a_ro0:.1f} != nx/FOV {nx0 / fov0[0]:.1f}; unsupported template")

    fx, fy = fov_x_mm * 1e-3, fov_y_mm * 1e-3
    a_ro1 = nx / fx
    s = a_ro1 / a_ro0
    da = a_ro1 - a_ro0

    # Spin echo: the x dephaser is the last gx event before the refocusing RF.
    # After the 180 its area flips sign, so echo centring requires
    #   -A_dephase + A_pre_crusher + A_readout/2 = 0   (crushers stay unchanged)
    rf_pos = [k for k, i in enumerate(tpl) if _ev(blocks[i], "rf") is not None]
    if len(rf_pos) != 2:
        raise NotImplementedError("expected exactly 2 RF pulses per TR (spin echo)")
    deph_k = max((k for k in range(rf_pos[1]) if _ev(blocks[tpl[k]], "gx") is not None), default=None)
    if deph_k is None:
        raise SequenceRebuildError("no x dephaser found before the refocusing pulse")
    a_d = float(_ev(blocks[tpl[deph_k]], "gx").area)
    a_pre = float(_ev(blocks[tpl[pos - 1]], "gx").area)
    if abs(-a_d + a_pre + a_ro0 / 2.0) > 0.02 * a_ro0:
        raise SequenceRebuildError("template x-gradient balance is not a standard spin echo; unsupported")
    if a_d + da / 2.0 <= 0:
        raise SequenceRebuildError("dephaser area would change sign; unsupported prescription")
    deph_scale = (a_d + da / 2.0) / a_d

    # check linear phase encoding in the template
    ky0 = [float(_ev(blocks[u[pos - 1]], "gy").area) for u in img]
    if not np.all(np.diff(ky0) > 0) or abs(ky0[1] - ky0[0] - 1.0 / fov0[1]) > 0.02 / fov0[1]:
        raise NotImplementedError("only linear, monotonic phase encoding is supported")
    gy_tpl = _ev(blocks[tpl[pos - 1]], "gy")

    # ADC over the same acquisition window, centred on the same point
    t_old = adc0.num_samples * adc0.dwell
    raster = getattr(sysm, "adc_raster_time", 1e-7)
    dwell = round(t_old / nx / raster) * raster
    t_new = nx * dwell
    delay = max(0.0, adc0.delay + t_old / 2.0 - t_new / 2.0)
    delay = round(delay / 1e-6) * 1e-6

    def emit(dst, unit, ky, with_adc):
        for k, i in enumerate(unit):
            b = blocks[i]
            rf, gx, gy, gz, adc = (_ev(b, n) for n in ("rf", "gx", "gy", "gz", "adc"))
            if k == deph_k:
                gx = pp.scale_grad(gx, deph_scale, system=sysm)
            if k == pos - 1 or k == pos + 1:
                if with_adc:
                    sgn = -1.0 if k == pos + 1 else 1.0
                    amp = sgn * ky / (gy_tpl.flat_time + 0.5 * (gy_tpl.rise_time + gy_tpl.fall_time))
                    gy = pp.make_trapezoid(channel="y", amplitude=amp, rise_time=gy_tpl.rise_time,
                                           flat_time=gy_tpl.flat_time, fall_time=gy_tpl.fall_time,
                                           delay=gy_tpl.delay, system=sysm)
            if k == pos:
                gx = pp.scale_grad(gx, s, system=sysm)
                adc = pp.make_adc(num_samples=nx, dwell=dwell, delay=delay,
                                  phase_offset=getattr(adc0, "phase_offset", 0.0),
                                  freq_offset=getattr(adc0, "freq_offset", 0.0), system=sysm) if with_adc else None
            evs = [e for e in (rf, gx, gy, gz, adc) if e is not None]
            evs.append(pp.make_delay(float(b.block_duration)))
            dst.add_block(*evs)

    dst = pp.Sequence(system=sysm)
    for key, val in src.definitions.items():
        dst.set_definition(key, val)
    dst.set_definition("FOV", [fx, fy, float(fov0[2])])
    dst.set_definition("Nx", int(nx))
    dst.set_definition("Ny", int(ny))

    done_img = False
    for unit, a in zip(units, has_adc):
        if not a:
            emit(dst, unit, 0.0, False)
        elif not done_img:
            for j in range(ny):
                emit(dst, unit, (j - ny / 2.0) / fy, True)
            done_img = True
    dst.write(seq_out)
    return {"scale_readout": s, "a_ro_old": a_ro0, "a_ro_new": a_ro1, "nx_old": nx0, "ny_old": ny0,
            "dwell_s": dwell, "adc_delay_s": delay, "n_units_out": len(dst.block_events)}
