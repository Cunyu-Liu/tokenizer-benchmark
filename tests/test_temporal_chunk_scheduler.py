"""CPU fixtures for the release-shift chunk scheduler."""
from __future__ import annotations

from data.run_temporal_chunks import _has_running


class _Proc:
    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_has_running_checks_process_values_not_chunk_ids():
    assert not _has_running({})
    assert _has_running({7: _Proc(None), 9: _Proc(0)})
    assert not _has_running({7: _Proc(0), 9: _Proc(1)})
