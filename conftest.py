"""Pytest root configuration.

Adds the ``src`` layout package to ``sys.path`` so tests can import
``qubo_receptor_ensemble`` without requiring an editable install.
Equivalent to ``python -m pip install -e .`` for import resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


@pytest.fixture()
def headroom_workspace() -> Iterator[Path]:
    """Sandbox-friendly scratch directory for the E1 headroom tests.

    ``pytest``'s ``tmp_path`` uses ``tempfile.mkdtemp`` (mode 0o700) and the
    cleanup of such directories is blocked in some restricted Windows
    sandboxes, so E1 tests allocate a plain directory under
    ``.codex-tmp/headroom-tests`` instead.
    """
    import shutil
    import uuid

    base = Path(__file__).resolve().parent / ".codex-tmp" / "headroom-tests"
    path = base / uuid.uuid4().hex[:10]
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
