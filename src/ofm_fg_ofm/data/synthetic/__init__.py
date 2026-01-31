# FILE: src/ofm_fg_ofm/data/synthetic/__init__.py
"""
Exports for synthetic dataset generation.

We keep a backwards-compatible name `make_synthetic_hydrocracker`
because earlier pipeline code imports it.
"""
from .generate import make as make_synthetic_hydrocracker  # backward compatible

__all__ = ["make_synthetic_hydrocracker"]
