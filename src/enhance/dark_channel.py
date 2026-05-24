"""
Stage 1 — Dark Channel Prior (DCP)
====================================
Implements the core of He et al. CVPR 2009:
  "Single Image Haze Removal Using Dark Channel Prior"

Key insight
-----------
In haze-free outdoor images (excluding sky), at least one colour channel
in any local patch has very low intensity.  Haze uniformly raises these
dark values.  The dark channel therefore gives a direct measure of haze
density.

The Atmospheric Scattering Model:
    I(x) = J(x) · t(x) + A · (1 - t(x))

Where:
    I(x)  = observed hazy pixel
    J(x)  = true scene radiance  (what we want)
    t(x)  = transmission map     (fraction of light that reaches camera)
    A     = atmospheric light    (global haze colour, usually bright white)

This module handles:
    Step 1 — Compute dark channel
    Step 2 — Estimate atmospheric light A  (sky-aware)
    Step 3 — Compute raw transmission map  t_raw(x)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Dark channel
# ---------------------------------------------------------------------------

def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Compute the dark channel of an image.

    For each pixel: take the minimum value across R, G, B channels, then
    take the local minimum over a square patch of side ``patch_size``.

    Uses ``cv2.erode`` with a rectangular structuring element as a fast
    O(N) approximation to the sliding-window patch minimum.

    Parameters
    ----------
    image : np.ndarray
        Float32 BGR image in range [0, 1].
    patch_size : int
        Side length of the local patch.  Default 15.

    Returns
    -------
    np.ndarray
        Dark channel map, shape (H, W), float32, range [0, 1].
    """
    # Minimum across colour channels
    min_channel = np.min(image, axis=2).astype(np.float32)

    # Patch minimum via morphological erosion (much faster than manual loop)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size)
    )
    return cv2.erode(min_channel, kernel)


# ---------------------------------------------------------------------------
# Step 2: Atmospheric light estimation
# ---------------------------------------------------------------------------

def estimate_atmospheric_light(
    image: np.ndarray,
    dark_ch: np.ndarray,
    sky_mask: np.ndarray,
    top_percent: float = 0.001,
) -> np.ndarray:
    """Estimate the global atmospheric light A from non-sky pixels.

    Algorithm:
      1. Restrict to pixels NOT in the sky mask.
      2. Find the top ``top_percent`` brightest pixels in the dark channel.
      3. Average the corresponding BGR values in the original image.

    Excluding sky pixels prevents the estimator from being biased by
    naturally bright regions that happen to look like heavy haze.

    Parameters
    ----------
    image : np.ndarray
        Float32 BGR image, range [0, 1].
    dark_ch : np.ndarray
        Dark channel map, shape (H, W), float32.
    sky_mask : np.ndarray
        Boolean mask. True = sky pixel.  Shape (H, W).
    top_percent : float
        Fraction of non-sky pixels to use.  Default 0.001 (0.1 %).

    Returns
    -------
    np.ndarray
        Atmospheric light vector, shape (3,), float32, range [0, 1].
    """
    non_sky = ~sky_mask

    # Fall back gracefully if the mask excludes everything
    if non_sky.sum() == 0:
        non_sky = np.ones_like(sky_mask, dtype=bool)

    flat_dark = dark_ch[non_sky].flatten()
    num_pixels = max(1, int(len(flat_dark) * top_percent))

    # Indices of the brightest dark-channel pixels (within non-sky region)
    brightest_local = np.argsort(flat_dark)[-num_pixels:]

    # Map back to 2-D image coordinates
    non_sky_coords = np.argwhere(non_sky)
    brightest_coords = non_sky_coords[brightest_local]

    A = np.mean(
        image[brightest_coords[:, 0], brightest_coords[:, 1]], axis=0
    )
    return A.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: Raw transmission map
# ---------------------------------------------------------------------------

def compute_transmission(
    image: np.ndarray,
    A: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
) -> np.ndarray:
    """Compute the raw (unrefined) transmission map.

    Formula:
        t_raw(x) = 1 - omega * dark_channel(I(x) / A)

    ``omega`` < 1 retains a small amount of haze for depth realism and
    prevents the recovered image from looking over-processed.

    Parameters
    ----------
    image : np.ndarray
        Float32 BGR image, range [0, 1].
    A : np.ndarray
        Atmospheric light vector, shape (3,), float32.
    patch_size : int
        Patch size for the inner dark channel computation.  Default 15.
    omega : float
        Haze retention factor.  0.95 is the standard value from He et al.

    Returns
    -------
    np.ndarray
        Raw transmission map, shape (H, W), float32, range [0, 1].
    """
    # Normalise by atmospheric light (avoid division by zero)
    A_safe = np.maximum(A, 1e-6)
    normalised = image / A_safe

    dc = dark_channel(normalised, patch_size=patch_size)
    t_raw = 1.0 - omega * dc
    return np.clip(t_raw, 0.0, 1.0).astype(np.float32)
