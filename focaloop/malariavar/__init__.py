"""Vendored MalariaVar production streaming & surveillance engine."""
from __future__ import annotations

import os
import pathlib
import tempfile

if "NUMBA_CACHE_DIR" not in os.environ:
    _nb_dir = pathlib.Path(tempfile.gettempdir()) / f"focaloop_numba_{os.getuid()}"
    _nb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_nb_dir)
