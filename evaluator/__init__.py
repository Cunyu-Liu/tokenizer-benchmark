"""TokBench-RNA Phase 2 unified evaluator package.

Pure-Python (CPU-safe) scoring, tokenization, continuation/generation metrics,
external-model adapter schema, sealed-test gate, and oracle fixtures.

Neural inference is GPU-only (handled by model adapters, cpu_fallback_count=0).
"""
__version__ = "0.2.0"
