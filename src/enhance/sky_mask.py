"""
Stage 0 — Sky / Cloud Region Detection
=======================================
Detects sky and bright cloud pixels that violate the Dark Channel Prior
assumption.  These regions are excluded from atmospheric light estimation
and handled separately during scene recovery.

Why this matters
----------------
DCP assumes that in most outdoor patches at least one channel has a very
low intensity.  Sky and bright cloud regions are uniformly bright across
all channels, which breaks this assumption and causes DCP to:
  • Overestimate atmospheric light A
  • Produce incorrect (near-zero) transmission values in sky regions
  • Introduce severe colour distortion after scene recovery

Algorithm
---------
A pixel is classified as sky/cloud when:
  • V  (HSV value / brightness) > brightness_thresh   (default 220)
  • S  (HSV saturation)         < sat_thresh          (default  30)

Morphological closing fills small gaps and produces a clean binary mask.
"""

import cv2
import numpy as np


def detect_sky_mask(
    image: np.ndarray,
    brightness_thresh: int = 220,
    sat_thresh: int = 30,
    kernel_size: int = 15,
) -> np.ndarray:
    """Return a boolean mask that is True for sky / bright-cloud pixels.

    Parameters
    ----------
    image : np.ndarray
        BGR image, uint8.
    brightness_thresh : int
        HSV V-channel threshold. Pixels brighter than this are sky
        candidates.  Range 0–255.  Default 220.
    sat_thresh : int
        HSV S-channel threshold. Pixels less saturated than this are sky
        candidates.  Range 0–255.  Default 30.
    kernel_size : int
        Side length of the square morphological closing kernel used to
        fill holes in the mask.  Default 15.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (H, W).  True = sky / cloud pixel.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    raw_mask = (v > brightness_thresh) & (s < sat_thresh)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(
        raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    )

    return closed.astype(bool)
