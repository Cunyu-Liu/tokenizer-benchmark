# Licensing and Third-Party Registry

This directory holds license records and the third-party manifest for all upstream code, public weights, and external baselines referenced by the TokBench-RNA benchmark.

- `third_party_manifest.json`: authoritative registry of upstream code, licenses, commits, and weights (see Goal §7 and §2.2).
- `UPSTREAM_LICENSES.md`: human-readable license summary for each upstream dependency.

Policy (Goal §2.2, §7):
- New code uses official BLT fixed commit, a public flat Transformer implementation, and public tokenizer tools as upstream references.
- Legacy candidate code is read-only historical evidence; no wholesale copy into the new benchmark.
- Any reused function is ported per-function, attributed, tested, and semantically audited.
- Upstream commit, license, and local modifications are recorded in `third_party/third_party_manifest.json`.
- External public weights are recorded with SHA-256, license, and model/revision in the external model registry (Phase 5).

Detailed upstream license audit is completed as code is ported in Phases 3 and 5; the manifest is updated incrementally and frozen at each delivery.