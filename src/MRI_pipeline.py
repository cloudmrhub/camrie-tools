#!/usr/bin/env python3
"""Compatibility wrapper for the packaged CAMRIE Koma pipeline.

Existing scripts can continue to run ``python src/MRI_pipeline.py`` or import
``MRI_pipeline`` while new installs can use ``camrie_tools.MRI_pipeline``.
"""

from __future__ import annotations

from camrie_tools.MRI_pipeline import *  # noqa: F401,F403
from camrie_tools.MRI_pipeline import main


if __name__ == "__main__":
    main()
