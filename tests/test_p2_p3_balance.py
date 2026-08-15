"""Unit fixtures for the P2/P3 balance audit (contract 3.2, 5.4).

These run on CPU using synthetic P2/P3 boundary providers (no GPU, no neural
entropy predictor) to verify the gate logic in p2_p3_balance.audit_p2_p3.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Stub the ConditionalRandomPatchPolicy so the audit runs on CPU.
from model.conditional_patch import ConditionalRandomPatchPolicy


class _P3Stub:
    def __init__(self, bnds): self.bnds = bnds
    def boundary(self, seq, length):
        return self.bnds.get(seq[:length], [1] + [0] * (length - 1))[:length]


class _P2Stub(ConditionalRandomPatchPolicy):
    """CPU stub: boundary = exact P3 replay for ALL positions (balanced)."""
    def __init__(self, bnds, edges=(8, 8)):
        super().__init__()
        self._bnds = bnds
        self.n_prefix_bins, self.n_entropy_bins = edges
        self.q_table = [[0.5] * edges[1] for _ in range(edges[0])]
        self.prefix_edges = [0, 100, 1000, 4096]
        self.ent_edges = [0.0, 0.5, 1.0, 1.5, 2.0]
        self.coverage_report = {
            "position_coverage": 1.0, "boundary_coverage": 1.0,
            "passes_support_coverage": True,
        }
    def _batch_entropy(self, canon):
        return np.full(len(canon), 0.5)
    def boundary(self, seq, length, seq_id=0):
        return self._bnds.get(seq[:length], [1] + [0] * (length - 1))[:length]


def _mk_seqs(n=100):
    import itertools, random
    rng = random.Random(0)
    out = []
    for _ in range(n):
        L = rng.randint(16, 300)
        out.append("".join(rng.choice("ACGU") for _ in range(L)))
    return out


def _balanced_bounds(seqs):
    """Same boundaries for P2 and P3 (perfect balance)."""
    bnds = {}
    for s in seqs:
        bnds[s] = [1] + [1 if i % 6 == 0 else 0 for i in range(1, len(s))]
    return bnds


def test_audit_perfect_balance_all_pass():
    """Identical P2/P3 boundaries -> all gates pass."""
    from p2_p3_balance import audit_p2_p3
    seqs = _mk_seqs(50)
    bnds = _balanced_bounds(seqs)
    p2 = _P2Stub(bnds)
    p3 = _P3Stub(bnds)
    rep = audit_p2_p3(p2, p3, seqs, d_model=448, n_layers=40)
    assert rep["total_patch_count"]["relative_error_pct"] <= 0.5
    assert rep["gate2_length_strata_pass"] is True
    assert rep["flops"]["difference_pct"] <= 5.0
    assert rep["all_pass"] is True


def test_audit_total_patch_gate_fail():
    """P2 with 2x patch count -> total patch error gate fails."""
    from p2_p3_balance import audit_p2_p3
    seqs = _mk_seqs(50)
    bnds = _balanced_bounds(seqs)
    # P2 doubles every boundary -> 2x patches
    bnds2 = {s: [1] * len(s) for s in seqs}
    p2 = _P2Stub(bnds2)
    p3 = _P3Stub(bnds)
    rep = audit_p2_p3(p2, p3, seqs, d_model=448, n_layers=40)
    assert rep["total_patch_count"]["relative_error_pct"] > 0.5
    assert rep["total_patch_count"]["pass"] is False
    assert rep["all_pass"] is False


def test_audit_flops_gate_fail():
    """P2 with 2x token count -> FLOPs diff gate fails."""
    from p2_p3_balance import audit_p2_p3
    seqs = _mk_seqs(50)
    bnds = _balanced_bounds(seqs)
    bnds2 = {s: [1] * len(s) for s in seqs}  # more tokens -> more FLOPs
    p2 = _P2Stub(bnds2)
    p3 = _P3Stub(bnds)
    rep = audit_p2_p3(p2, p3, seqs, d_model=448, n_layers=40)
    assert rep["flops"]["difference_pct"] > 5.0
    assert rep["flops"]["pass"] is False


def test_audit_length_stratum_report():
    """Length-stratum rows are produced and individually flagged."""
    from p2_p3_balance import audit_p2_p3
    seqs = _mk_seqs(100)
    bnds = _balanced_bounds(seqs)
    p2 = _P2Stub(bnds)
    p3 = _P3Stub(bnds)
    rep = audit_p2_p3(p2, p3, seqs, d_model=448, n_layers=40)
    assert len(rep["length_strata"]) >= 1
    for row in rep["length_strata"]:
        assert "p2_patch_count" in row and "p3_patch_count" in row
        assert "pass" in row


def test_audit_nonsupported_replay():
    """Non-supported strata replay P3 exactly -> gate passes."""
    from p2_p3_balance import audit_p2_p3
    seqs = _mk_seqs(50)
    bnds = _balanced_bounds(seqs)
    p2 = _P2Stub(bnds)
    p3 = _P3Stub(bnds)
    # Force q_table to non-supported everywhere (q=1.0 -> not supported)
    p2.q_table = [[1.0] * 8 for _ in range(8)]
    rep = audit_p2_p3(p2, p3, seqs, d_model=448, n_layers=40)
    # P2 replays P3 exactly, so replay check has 0 failures
    assert rep["non_supported_replay"]["failures"] == 0
    assert rep["non_supported_replay"]["checked"] > 0


def test_audit_boundary_rate_gate():
    """Per-stratum boundary-rate abs diff > 2pp flags gate4."""
    from p2_p3_balance import audit_p2_p3
    seqs = _mk_seqs(50)
    bnds3 = _balanced_bounds(seqs)
    # P2 has boundaries at every position (rate ~1.0) while P3 has sparse.
    bnds2 = {s: [1] * len(s) for s in seqs}
    p2 = _P2Stub(bnds2)
    p3 = _P3Stub(bnds3)
    rep = audit_p2_p3(p2, p3, seqs, d_model=448, n_layers=40)
    assert rep["stratum_boundary_rate"]["pass"] == False  # noqa: E712


def test_audit_nonsequences_empty():
    """Empty seq list -> audit still returns a report without crashing."""
    from p2_p3_balance import audit_p2_p3
    p2 = _P2Stub({})
    p3 = _P3Stub({})
    rep = audit_p2_p3(p2, p3, [], d_model=448, n_layers=40)
    assert rep["n_sequences"] == 0
    assert "all_pass" in rep