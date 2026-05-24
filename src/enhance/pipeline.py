"""
Pipeline Orchestrator
======================
Wires together all five stages of the satellite dehazing pipeline.

Stages
------
  Stage 0 — Sky mask detection          (sky_mask.py)
  Stage 1 — Dark Channel Prior          (dark_channel.py)
  Stage 2 — Guided filter refinement    (guided_filter.py)
  Stage 3 — Scene recovery              (scene_recovery.py)
  Stage 4 — Mild CLAHE polish           (contrast.py)
  Stage 5 — Contrast stretch + sharpen  (contrast.py + filtering.py)

Usage
-----
    from src.enhance import PipelineConfig, run_pipeline
    import cv2

    config = PipelineConfig()
    result = run_pipeline(cv2.imread("hazy.jpg"), config)
    cv2.imwrite("dehazed.jpg", result)
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .sky_mask import detect_sky_mask
from .dark_channel import dark_channel, estimate_atmospheric_light, compute_transmission
from .guided_filter import refine_transmission
from .scene_recovery import recover_scene
from .contrast import apply_clahe, contrast_stretch
from .filtering import sharpen


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tuneable parameters for the satellite dehazing pipeline.

    Attributes
    ----------
    Stage 0 — Sky mask
        sky_brightness_thresh : int
            HSV V-channel threshold above which a pixel is a sky candidate.
        sky_sat_thresh : int
            HSV S-channel threshold below which a pixel is a sky candidate.
        sky_kernel_size : int
            Morphological closing kernel size for the sky mask.

    Stage 1 — DCP
        dcp_patch_size : int
            Side length of the patch-minimum window for the dark channel.
        dcp_omega : float
            Haze retention factor (0.95 = recommended by He et al.).
        dcp_top_percent : float
            Fraction of pixels used to estimate atmospheric light.

    Stage 2 — Guided filter
        gf_radius : int
            Guided filter neighbourhood radius.
        gf_eps : float
            Guided filter regularisation parameter.

    Stage 3 — Scene recovery
        t_min : float
            Minimum allowed transmission value to prevent divide-by-zero.

    Stage 4 — CLAHE
        clahe_enabled : bool
            Toggle mild CLAHE post-processing.
        clahe_clip_limit : float
            CLAHE clip limit. Must be much lower than underwater (0.5–1.0).
        clahe_tile_grid : tuple
            CLAHE tile grid size.

    Stage 5 — Contrast + sharpen
        contrast_stretch_enabled : bool
            Toggle per-channel contrast stretching.
        sharpen_enabled : bool
            Toggle Laplacian sharpening.
    """

    # --- Stage 0: Sky mask ---
    sky_brightness_thresh: int = 220
    sky_sat_thresh: int = 30
    sky_kernel_size: int = 15

    # --- Stage 1: DCP ---
    dcp_patch_size: int = 15
    dcp_omega: float = 0.95
    dcp_top_percent: float = 0.001

    # --- Stage 2: Guided filter ---
    gf_radius: int = 40
    gf_eps: float = 1e-3

    # --- Stage 3: Scene recovery ---
    t_min: float = 0.1

    # --- Stage 4: CLAHE ---
    clahe_enabled: bool = True
    clahe_clip_limit: float = 0.75
    clahe_tile_grid: tuple = field(default_factory=lambda: (8, 8))

    # --- Stage 5: Contrast + sharpen ---
    contrast_stretch_enabled: bool = True
    sharpen_enabled: bool = True


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    image: np.ndarray,
    config: Optional[PipelineConfig] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run the full satellite dehazing pipeline on a single image.

    Parameters
    ----------
    image : np.ndarray
        Input hazy BGR image, uint8.
    config : PipelineConfig, optional
        Pipeline configuration.  Defaults to PipelineConfig() if None.
    verbose : bool
        If True, print progress to stdout.

    Returns
    -------
    result : np.ndarray
        Dehazed BGR image, uint8.
    stages : dict[str, np.ndarray]
        Dictionary of intermediate results keyed by stage name.
        Useful for visualisation with ``utils.show_pipeline()``.
    """
    if config is None:
        config = PipelineConfig()

    def log(msg: str) -> None:
        if verbose:
            print(f"  {msg}")

    stages: dict[str, np.ndarray] = {"0 · Input": image.copy()}

    # --- Stage 0: Sky mask ---
    log("Stage 0 — Sky mask detection …")
    sky_mask = detect_sky_mask(
        image,
        brightness_thresh=config.sky_brightness_thresh,
        sat_thresh=config.sky_sat_thresh,
        kernel_size=config.sky_kernel_size,
    )
    stages["0 · Sky Mask"] = (sky_mask.astype(np.uint8) * 255)

    # Normalise input to float32 [0, 1] for physics computations
    img_f = image.astype(np.float32) / 255.0

    # --- Stage 1: Dark Channel Prior ---
    log("Stage 1 — Dark Channel Prior …")
    dc = dark_channel(img_f, patch_size=config.dcp_patch_size)
    stages["1 · Dark Channel"] = dc

    A = estimate_atmospheric_light(
        img_f, dc, sky_mask, top_percent=config.dcp_top_percent
    )
    log(f"           Atmospheric light A = {A * 255}")

    t_raw = compute_transmission(
        img_f, A, patch_size=config.dcp_patch_size, omega=config.dcp_omega
    )
    stages["1 · Raw Transmission"] = t_raw

    # --- Stage 2: Guided filter refinement ---
    log("Stage 2 — Guided filter refinement …")
    t_refined = refine_transmission(
        img_f, t_raw, radius=config.gf_radius, eps=config.gf_eps
    )
    stages["2 · Refined Transmission"] = t_refined

    # --- Stage 3: Scene recovery ---
    log("Stage 3 — Scene recovery …")
    recovered = recover_scene(img_f, A, t_refined, sky_mask, t_min=config.t_min)
    stages["3 · Recovered Scene"] = recovered

    result = recovered.copy()

    # --- Stage 4: Mild CLAHE polish ---
    if config.clahe_enabled:
        log("Stage 4 — Mild CLAHE polish …")
        result = apply_clahe(
            result,
            clip_limit=config.clahe_clip_limit,
            tile_grid=config.clahe_tile_grid,
        )
    stages["4 · After CLAHE"] = result.copy()

    # --- Stage 5a: Contrast stretch ---
    if config.contrast_stretch_enabled:
        log("Stage 5a — Contrast stretch …")
        result = contrast_stretch(result)

    # --- Stage 5b: Sharpen ---
    if config.sharpen_enabled:
        log("Stage 5b — Laplacian sharpening …")
        result = sharpen(result)

    stages["5 · Final Output"] = result.copy()

    log("Pipeline complete.")
    return result, stages
