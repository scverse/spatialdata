from __future__ import annotations

import os
import sys

# Disable numba JIT compilation for dataloader tests. Datashader (used by rasterize) triggers
# numba JIT on first call, costing ~1.4s per worker. Python-mode gives identical results for
# the small test data here — unlike real data, there is no throughput advantage from JIT.
os.environ["NUMBA_DISABLE_JIT"] = "1"
if "numba.core.config" in sys.modules:
    sys.modules["numba.core.config"].NUMBA_DISABLE_JIT = 1
