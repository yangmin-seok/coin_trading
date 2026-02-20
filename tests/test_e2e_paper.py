from __future__ import annotations

from pipelines.test import run


def test_paper_pipeline_scaffold():
    assert "scaffold" in run()
