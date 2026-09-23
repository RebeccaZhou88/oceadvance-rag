"""pytest global fixtures: uniformly inject import paths, ensuring any test file, any collection order can import.

- backend/src: src-layout app package (pyproject's pythonpath only works for some invocation modes)
- backend/   : top-level ingestion / eval packages outside src
"""
import asyncio
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_BACKEND_ROOT / "src"), str(_BACKEND_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
