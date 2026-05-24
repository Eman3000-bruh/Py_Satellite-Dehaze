"""
Stage 3 — Scene Recovery
=========================
Inverts the Atmospheric Scattering Model to recover the haze-free image.

The Atmospheric Scattering Model:
    I(x) = J(x) · t(x) + A · (1 - t(x))

Solving for J(x):
    J(x) = (I(x) - A) / max(t(x), t_min) + A

Where:
    I(x)   = observed hazy pixel
    J(x)   = recovered (haze-free) pixel
    t(x)   = refined transmission map
    A      = atmospheric light
    t_min  = lower bound on transmission (prevents divide-by-zero in
             very dense haze regions, typically 0.1)

Sky regions
-----------
Pixels in the sky mask are not dehazed — they are blended back from
the original image to avoid the colour distortions that occur when DCP
is applied to naturally bright, low-saturation sky.
"""

import numpy as np


def recover_scene(
    hazy: np.ndarray,
    A: np.ndarray,
    transmission: np.ndarray,
    sky_mask: np.ndarray,
    t_min: float = 0.1,
) -> np.ndarray:
    """Recover the haze-free scene from the atmospheric scattering model.

    Parameters
    ----------
    hazy : np.ndarray
        Original hazy image, float32, range [0, 1], shape (H, W, 3).
    A : np.ndarray
        Atmospheric light vector, float32, shape (3,), range [0, 1].
    transmission : np.ndarray
        Refined transmission map, float32, shape (H, W), range [0, 1].
    sky_mask : np.ndarray
        Boolean mask.  True = sky pixel.  Shape (H, W).
    t_min : float
        Minimum transmission clamp.  Prevents divide-by-zero artefacts
        in very dense haze regions.  Default 0.1.

    Returns
    -------
    np.ndarray
        Recovered image, uint8, range [0, 255], shape (H, W, 3).
    """
    # Expand transmission to (H, W, 1) for broadcasting
    t = np.maximum(transmission[:, :, np.newaxis], t_min)

    # Invert the scattering model
    scene = (hazy - A) / t + A

    # Replace sky pixels with the original image to avoid DCP artefacts
    sky_3ch = np.stack([sky_mask] * 3, axis=2)
    scene[sky_3ch] = hazy[sky_3ch]

    scene_clipped = np.clip(scene, 0.0, 1.0)
    return (scene_clipped * 255).astype(np.uint8)
