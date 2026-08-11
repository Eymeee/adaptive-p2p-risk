from __future__ import annotations

import sys


def test_python_version_matches_project_requirement() -> None:
    assert sys.version_info >= (3, 11)
