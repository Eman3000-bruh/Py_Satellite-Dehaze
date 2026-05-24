"""
Utilities — Visualisation and I/O
===================================
Provides helpers for displaying pipeline stages and saving results.
Mirrors the utils.py API from the upstream underwater pipeline.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR uint8 array.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        BGR image, uint8.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file could not be decoded as an image.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def save_result(image: np.ndarray, output_path: str) -> None:
    """Save an image to disk, creating parent directories if needed.

    Parameters
    ----------
    image : np.ndarray
        BGR image, uint8.
    output_path : str
        Destination path.  Directories are created automatically.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 → RGB for matplotlib display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_comparison(
    original: np.ndarray,
    dehazed: np.ndarray,
    title: str = "Dehazing Result",
    save_path: Optional[str] = None,
) -> None:
    """Display original and dehazed images side by side.

    Parameters
    ----------
    original : np.ndarray
        Original hazy image, BGR uint8.
    dehazed : np.ndarray
        Dehazed image, BGR uint8.
    title : str
        Figure title.  Default "Dehazing Result".
    save_path : str, optional
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(_bgr_to_rgb(original))
    axes[0].set_title("Original (Hazy)")
    axes[0].axis("off")

    axes[1].imshow(_bgr_to_rgb(dehazed))
    axes[1].set_title("Dehazed")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved: {save_path}")
    plt.show()


def show_pipeline(
    stages: dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> None:
    """Display all intermediate pipeline stages in a grid.

    Parameters
    ----------
    stages : dict[str, np.ndarray]
        Ordered mapping of stage name → image (BGR uint8) or float map.
        Float maps (e.g. transmission) are displayed with a viridis
        colour map.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    n = len(stages)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.suptitle("Pipeline Stages", fontsize=14, fontweight="bold")
    axes_flat = np.array(axes).flatten()

    for idx, (name, img) in enumerate(stages.items()):
        ax = axes_flat[idx]
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            ax.imshow(img.squeeze(), cmap="viridis")
        else:
            ax.imshow(_bgr_to_rgb(img))
        ax.set_title(name)
        ax.axis("off")

    # Hide any unused axes
    for idx in range(len(stages), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Pipeline diagram saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    original : np.ndarray
        Reference image, uint8.
    enhanced : np.ndarray
        Enhanced image, uint8.

    Returns
    -------
    float
        PSNR in dB.  Returns inf if the images are identical.
    """
    mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))
