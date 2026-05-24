"""
Stage 5 — Sharpening
=====================
Applies a Laplacian sharpening kernel as the final polish step.

The kernel used is the standard 3×3 unsharp/sharpening kernel:
    [[0, -1,  0],
     [-1,  5, -1],
     [0, -1,  0]]

This is identical to the filtering step in the upstream underwater
pipeline, so the same module is reused here.

Also provides an optional homomorphic filter for thin cloud removal.
Thin cloud cover is a *multiplicative* low-frequency signal rather than
an additive scattering effect, so it is best addressed in the frequency
domain.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Laplacian sharpening
# ---------------------------------------------------------------------------

SHARPEN_KERNEL = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]],
    dtype=np.float32,
)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Apply Laplacian sharpening to an image.

    Parameters
    ----------
    image : np.ndarray
        BGR image, uint8.

    Returns
    -------
    np.ndarray
        Sharpened BGR image, uint8.
    """
    sharpened = cv2.filter2D(image, ddepth=-1, kernel=SHARPEN_KERNEL)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Homomorphic filter (thin cloud removal)
# ---------------------------------------------------------------------------

def homomorphic_filter(
    image: np.ndarray,
    cutoff: float = 30.0,
    high_boost: float = 1.5,
    low_boost: float = 0.5,
) -> np.ndarray:
    """Apply a homomorphic filter to suppress thin cloud cover.

    Thin cloud is a multiplicative low-frequency illumination component.
    Working in the log domain converts the multiplicative model to
    additive, allowing a high-pass filter to suppress the cloud layer
    while preserving high-frequency scene detail.

    The filter is applied independently per channel.

    Parameters
    ----------
    image : np.ndarray
        BGR image, uint8.
    cutoff : float
        Frequency cut-off for the Gaussian high-pass envelope.
        Default 30.  Lower = more aggressive cloud suppression.
    high_boost : float
        Gain applied to high-frequency components.  Default 1.5.
    low_boost : float
        Gain applied to low-frequency components (≤ 1 attenuates clouds).
        Default 0.5.

    Returns
    -------
    np.ndarray
        Filtered BGR image, uint8.
    """
    img_float = image.astype(np.float32)
    out_channels = []

    for ch_idx in range(3):
        channel = img_float[:, :, ch_idx]

        # Log domain  (log1p avoids log(0))
        log_ch = np.log1p(channel)

        # FFT
        fft = np.fft.fft2(log_ch)
        fft_shift = np.fft.fftshift(fft)

        # Gaussian high-pass envelope
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        dist_sq = (X - ccol) ** 2 + (Y - crow) ** 2
        H = (high_boost - low_boost) * (
            1 - np.exp(-dist_sq / (2 * cutoff ** 2))
        ) + low_boost

        # Apply filter and inverse FFT
        filtered = fft_shift * H
        result = np.fft.ifft2(np.fft.ifftshift(filtered)).real

        # Back to linear domain
        out_channels.append(np.expm1(result))

    filtered_img = np.stack(out_channels, axis=2)

    # Normalise to [0, 255]
    filtered_img -= filtered_img.min()
    max_val = filtered_img.max()
    if max_val > 0:
        filtered_img = filtered_img / max_val * 255.0

    return np.clip(filtered_img, 0, 255).astype(np.uint8)
