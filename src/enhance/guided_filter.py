"""
Stage 2 — Guided Filter Transmission Refinement
================================================
He et al., TPAMI 2013: "Guided Image Filtering"

Problem
-------
The raw DCP transmission map is blocky and noisy because it is computed
via a patch-based minimum operation.  Applying scene recovery directly on
the raw map produces severe halo artefacts around object edges (buildings,
treelines, etc.).

Solution
--------
The guided filter smooths the transmission map while preserving edges by
using the original hazy image as a guidance signal.  It models the output
as a linear transform of the guide within local windows, which naturally
preserves edges present in the guide image.

Implementation
--------------
Preferred: ``cv2.ximgproc.guidedFilter`` from opencv-contrib.
Fallback:  Bilateral filter on the transmission map — less accurate but
           requires no extra dependency.
"""

import cv2
import numpy as np


def refine_transmission(
    image: np.ndarray,
    t_raw: np.ndarray,
    radius: int = 40,
    eps: float = 1e-3,
) -> np.ndarray:
    """Refine the raw transmission map using a guided filter.

    The original hazy image is used as the guide so that edges in the
    scene are preserved in the smoothed transmission map.

    Falls back to a bilateral filter automatically when
    ``cv2.ximgproc`` is not available (opencv-contrib not installed).

    Parameters
    ----------
    image : np.ndarray
        Float32 BGR image in range [0, 1].  Used as the guide.
    t_raw : np.ndarray
        Raw transmission map, shape (H, W), float32, range [0, 1].
    radius : int
        Neighbourhood radius for the guided filter.  Larger values
        produce smoother maps at the cost of detail.  Default 40.
    eps : float
        Regularisation parameter.  Controls smoothing strength vs edge
        preservation trade-off.  Default 1e-3.

    Returns
    -------
    np.ndarray
        Refined transmission map, shape (H, W), float32, range [0, 1].
    """
    # Convert guide to uint8 for cv2.ximgproc.guidedFilter
    guide_uint8 = (image * 255).astype(np.uint8)
    t_uint8 = (t_raw * 255).astype(np.uint8)

    try:
        t_refined_uint8 = cv2.ximgproc.guidedFilter(
            guide=guide_uint8,
            src=t_uint8,
            radius=radius,
            eps=eps * (255 ** 2),  # scale eps to uint8 range
            dDepth=-1,
        )
        t_refined = t_refined_uint8.astype(np.float32) / 255.0
    except AttributeError:
        # opencv-contrib not available → bilateral filter fallback
        t_refined_uint8 = cv2.bilateralFilter(
            t_uint8,
            d=9,
            sigmaColor=75,
            sigmaSpace=75,
        )
        t_refined = t_refined_uint8.astype(np.float32) / 255.0

    return np.clip(t_refined, 0.0, 1.0).astype(np.float32)
