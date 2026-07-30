"""Small runnable example for CAMRIE tools."""

from __future__ import annotations

import tempfile
import os
from pathlib import Path


EXAMPLE_SEQUENCE = """[DEFINITIONS]
FOV 0.220 0.180 0.005
SliceThickness 0.005
TE 0.012
TEeff 0.012
TurboFactor 1

[BLOCKS]
# id delay rf gx gy gz adc ext
1 0 0 1 0 0 1 0
2 0 0 1 0 0 1 0
3 0 0 1 0 0 1 0
4 0 0 1 0 0 1 0

[TRAP]
# id amplitude rise flat fall delay
1 -1.0 0.0001 0.001 0.0001 0

[ADC]
# id num dwell delay freq phase
1 64 0.000004 0 0 0
"""


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

    import camrie_tools
    from camrie_tools.MRI_pipeline import read_pulseq_params_fallback

    print(f"camrie_tools {camrie_tools.__version__}")
    print(f"Bundled Julia script: {camrie_tools.simulate_batch_path()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        seq_path = Path(tmpdir) / "camrie_example.seq"
        seq_path.write_text(EXAMPLE_SEQUENCE)
        params = read_pulseq_params_fallback(str(seq_path))

    print("Parsed example sequence:")
    print(f"  nF: {params['nF']}")
    print(f"  nP: {params['nP']}")
    print(f"  FOV mm: {params['fov_mm']}")
    print(f"  slice thickness mm: {params['slice_thickness_mm']}")
    print(f"  source: {params['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
