r"""E1 headroom diagnostic: fusion rules, exhaustive subset oracles, noise floor.

The package implements the pre-registered E1 protocol from
``E:\Quant\docs\qubo\E1实现计划_headroom扫描_20260911.md``:

- :mod:`fusion` eight frozen fusion rules ``phi`` with train-only parameters;
- :mod:`subsets` bit-mask enumeration and scalar utility helpers;
- :mod:`metrics_fast` O(n) screening metrics numerically identical to
  :mod:`qubo_receptor_ensemble.screening`;
- :mod:`headroom` ``H_raw`` / ``H_nested`` / ``H_perm`` fold batteries;
- :mod:`bootstrap` scaffold-cluster bootstrap, noise floor and MDE;
- :mod:`gate` frozen G1 decision logic;
- :mod:`assets` D1 input verification, problem.json carriers, SHA-256 manifests.
"""

from __future__ import annotations

PACKAGE_SCHEMA = "e1_headroom_v1"

__all__ = ["PACKAGE_SCHEMA"]