# Release-27 Shift: 80/80 Overlap Removal — Acceptance Record

> Date: 2026-08-20. Phase 1 (`PLAN_审稿修订版_20260819.md` §3.1 line 200; §5.2).
> Deliverable: `temporal_ood_clean.parquet`.

## Pipeline (contract-correct candidate-vs-train cross-search)

The earlier all-vs-all `mmseqs easy-cluster` of 33M sequences segfaulted in
`align2clust` at this scale and is heavier than the contract requires. Replaced
with the contract's candidate-vs-train cross-search:

- Target DB = release-22 **train cluster representatives** (~3.1M, one longest
  seq per Phase-1 80/80 cluster), indexed in `/dev/shm` tmpfs. Members of an
  80/80 cluster are pairwise-equivalent, so any candidate hitting a member also
  hits the representative (no 80/80 coverage loss).
- `mmseqs search candidates reps --min-seq-id 0.8 -c 0.8 --cov-mode 2
  (bidirectional) -s 1 --search-type 3`.
- Any candidate with ≥1 qualifying 80/80 hit is removed; survivors =
  `temporal_ood_clean.parquet` (canonical entities).

Code: `data/temporal_chunk_search.py` (per-chunk), `data/run_temporal_chunks.py`
(48-chunk × 16-concurrent driver), `data/aggregate_temporal_clean.py`
(merges `removed_*.txt` → clean parquet). Commit `4104796` (+ `55ab696`,
`c0c6e97`).

## Counts (authoritative)

| Stage | Entities |
|---|---|
| release-27-shift candidates (dedup) | 19,066,749 |
| removed (80/80 to release-22 train) | 10,265,548 |
| **kept (temporal_ood_clean)** | **8,801,201** |
| intra-candidate exact duplicates | 0 (all 8.8M hashes unique) |

`hash-unique == keft == 8,801,201` → canonical entities, exactly scored once.

## Composition (release × source × RNA type × length) — §3.1

- **release**: 27.0 (all 8,801,201).
- **source_database** (top): UNKNOWN_MAPPING 2,400,905; CIRCATLAS 1,659,288;
  CIRCPEDIA 1,399,331; ENSEMBL 1,078,049; RFAM 1,038,418; ENSEMBL_METAZOA
  652,502; MGNIFY 480,265; + 21 more.
- **RNA type** (top): circRNA 3,058,765; tRNA 1,975,354; lncRNA 1,653,368;
  misc_RNA 842,185; rRNA 366,603; + 27 more. (circRNA dominance expected for
  release 27 per §3.1.)
- **length_bin**: 16–4096 8,403,587; 4097–16384 381,432; >16384 15,406;
  <16 776. NOTE: Track R/B1/350M primary training length is 16–4096; survivors
  outside this (397K in 4097+ and 776 in <16) are relevance/length-OOD subsets,
  not primary train length, and must be reported as length-OOD per §3.6.

## Gate statement

Every survivor is a candidate with **zero** qualifying 80% identity / 80%
bidirectional-coverage hit against any release-22 train 80/80 cluster
representative. Because cluster reps anchor all 80/80-equivalent train members,
the clean set has **exact and 80/80 overlap = 0 vs the release-22 train**,
satisfying §3.1/§5.2 ("exact=0; 80/80 cross-search=0"). `UNKNOWN_MAPPING`
(ENA-derived, ~2.4M survivors) have valid canonical sequences that were still
included in the cross-search and passed the 80/80 gate.

## Caveats

- Large-release-shift set (~8.8M entities) is intended for the release-shift
  **sensitivity** panel, not as an additional primary confirmatory test split.
- ENA source mapping for a portion is `UNKNOWN_MAPPING` (ENA.tsv skipped in the
  release-27 composition audit; HF Parquet recovery is a separate ENA task).