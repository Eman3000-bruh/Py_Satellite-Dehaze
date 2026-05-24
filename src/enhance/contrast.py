"""
Stage 4 — Contrast Enhancement
================================
Applied *after* scene recovery as a light polish step.

Key difference from the underwater pipeline
--------------------------------------------
In the underwater pipeline CLAHE is the primary dehazing technique
(clip_limit=2.0, applied aggressively on the V channel before anything
else).

Here, CLAHE is a *post-processing* step only:
  • clip_limit 0.5–1.0  (much lower — aggressive CLAHE would amplify
    any residual haze or noise in sky regions)
  • Applied after the physics-based dehazing has already removed most
    of the haze

Also provides per-channel contrast stretching (linear remap) which can
optionally be followed by per-channel histogram equalisation.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CLAHE polish
# ---------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 0.75,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE on the V channel in HSV space as a post-dehaze polish.

    Parameters
    ----------
    image : np.ndarray
        BGR image, uint8.
    clip_limit : float
        CLAHE clip limit.  Keep low (0.5–1.0) for satellite imagery.
        Default 0.75.
    tile_grid : tuple[int, int]
        CLAHE tile grid size.  Default (8, 8).

    Returns
    -------
    np.ndarray
        BGR image, uint8, with mild contrast enhancement.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Per-channel contrast stretch
# ---------------------------------------------------------------------------

def contrast_stretch(
    image: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Linear per-channel contrast stretch.

    Remaps [p_low, p_high] percentile values to [0, 255], clipping
    extreme outliers.  This corrects any channel imbalance that remains
    after dehazing.

    Parameters
    ----------
    image : np.ndarray
        BGR image, uint8.
    percentile_low : float
        Lower percentile for stretch.  Default 1.0.
    percentile_high : float
        Upper percentile for stretch.  Default 99.0.

    Returns
    -------
    np.ndarray
        Contrast-stretched BGR image, uint8.
    """
    out = np.zeros_like(image, dtype=np.float32)
    for ch in range(3):
        channel = image[:, :, ch].astype(np.float32)
        lo = np.percentile(channel, percentile_low)
        hi = np.percentile(channel, percentile_high)
        if hi > lo:
            out[:, :, ch] = (channel - lo) / (hi - lo) * 255.0
        else:
            out[:, :, ch] = channel
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Optional histogram equalisation (per-channel)
# ---------------------------------------------------------------------------

def histogram_equalise(image: np.ndarray) -> np.ndarray:
    """Apply per-channel histogram equalisation.

    Parameters
    ----------
    image : np.ndarray
        BGR image, uint8.

    Returns
    -------
    np.ndarray
        Histogram-equalised BGR image, uint8.
    """
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(eq_channels)
