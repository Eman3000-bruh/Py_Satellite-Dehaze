"""
Satellite Image Dehazing Pipeline
==================================
Physics-based dehazing using the Atmospheric Scattering Model.

Pipeline stages:
  Stage 0 — Sky mask detection
  Stage 1 — Dark Channel Prior (DCP)
  Stage 2 — Guided filter transmission refinement
  Stage 3 — Scene recovery  (inverts the scattering model)
  Stage 4 — Mild CLAHE polish
  Stage 5 — Contrast stretch + Laplacian sharpening
"""

from .pipeline import PipelineConfig, run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
