# Satellite Image Dehazing

Physics-based satellite image dehazing using the **Atmospheric Scattering Model** and **Dark Channel Prior** (He et al., CVPR 2009).

---

## Background

Haze in satellite imagery is caused by atmospheric scattering — the atmosphere adds a white/grey veil uniformly across all channels.  This is fundamentally different from underwater degradation (channel-selective absorption), so the heuristic CLAHE-on-V approach used for underwater imagery does **not** work here.

The scattering model is:

```
I(x) = J(x) · t(x) + A · (1 - t(x))
```

| Symbol | Meaning |
|--------|---------|
| `I(x)` | Observed hazy pixel |
| `J(x)` | True scene radiance (what we want) |
| `t(x)` | Transmission map — fraction of scene light reaching the camera |
| `A`    | Atmospheric light — global haze colour (usually bright white) |

Scene recovery inverts this:

```
J(x) = (I(x) - A) / max(t(x), t_min) + A
```

---

## Pipeline

```
Input
  │
  ▼
Stage 0 · Sky Mask Detection
  • Threshold on V > 220 and S < 30 (HSV)
  • Morphological closing to fill gaps
  │
  ▼
Stage 1 · Dark Channel Prior (DCP)
  • Compute dark channel via patch minimum (cv2.erode)
  • Estimate atmospheric light A  (sky pixels excluded)
  • Compute raw transmission map t(x)
  │
  ▼
Stage 2 · Guided Filter Refinement
  • Smooth t(x) using the original image as guidance
  • Preserves edges; prevents halo artefacts
  │
  ▼
Stage 3 · Scene Recovery
  • J = (I − A) / max(t, 0.1) + A
  • Sky pixels replaced from original (avoid DCP artefacts)
  │
  ▼
Stage 4 · Mild CLAHE Polish  (optional)
  • clip_limit = 0.75  (much lower than underwater pipeline)
  • Post-dehazing only
  │
  ▼
Stage 5 · Contrast Stretch + Sharpen
  • Per-channel linear stretch [p1, p99] → [0, 255]
  • Laplacian sharpening kernel
  │
  ▼
Output
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `opencv-contrib-python` is required for the guided filter
> (`cv2.ximgproc.guidedFilter`). If it is unavailable, the pipeline
> automatically falls back to a bilateral filter on the transmission map.

---

## Usage

```bash
# Basic dehazing
python enhance.py hazy_image.jpg

# Specify output
python enhance.py hazy_image.jpg --output results/dehazed.jpg

# Show before/after comparison
python enhance.py hazy_image.jpg --show-comparison

# Show all pipeline stages
python enhance.py hazy_image.jpg --show-pipeline

# Save comparison and pipeline diagram
python enhance.py hazy_image.jpg \
    --save-comparison results/compare.png \
    --save-pipeline results/pipeline.png

# Tune parameters
python enhance.py hazy_image.jpg \
    --omega 0.85 \
    --t-min 0.15 \
    --clahe-clip 1.0 \
    --gf-radius 60

# Disable optional stages
python enhance.py hazy_image.jpg --no-clahe --no-sharpen

# Verbose output
python enhance.py hazy_image.jpg --verbose
```

Full options:

```
positional arguments:
  input                 Path to the input hazy image

optional arguments:
  -o, --output          Output path
  --show-pipeline       Display all intermediate stages
  --show-comparison     Display before/after comparison
  --save-comparison     Save before/after comparison image
  --save-pipeline       Save pipeline stage grid

Stage 0 — Sky mask:
  --sky-brightness      V-channel threshold (default: 220)
  --sky-saturation      S-channel threshold (default: 30)

Stage 1 — Dark Channel Prior:
  --patch-size          DCP patch size (default: 15)
  --omega               Haze retention factor (default: 0.95)
  --top-percent         Top fraction for atmospheric light (default: 0.001)

Stage 2 — Guided filter:
  --gf-radius           Filter radius (default: 40)
  --gf-eps              Regularisation eps (default: 1e-3)

Stage 3 — Scene recovery:
  --t-min               Minimum transmission (default: 0.1)

Stage 4 — CLAHE:
  --no-clahe            Disable CLAHE post-processing
  --clahe-clip          CLAHE clip limit (default: 0.75)

Stage 5 — Contrast & sharpen:
  --no-contrast-stretch Disable contrast stretching
  --no-sharpen          Disable Laplacian sharpening

  --verbose, -v         Print stage-by-stage progress
```

---

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| DCP patch size | 15×15 px | Larger = smoother dark channel, more halo risk |
| DCP omega | 0.95 | Haze retention. 1.0 = full removal (looks unnatural) |
| Top-% for A | 0.1% | Fraction of non-sky pixels used to estimate atmospheric light |
| t_min | 0.1 | Prevents divide-by-zero in very dense haze |
| Guided filter radius | 40 px | Larger = smoother transmission |
| Guided filter eps | 1e-3 | Regularisation strength |
| CLAHE clip_limit | 0.75 | **Much lower than underwater (2.0)** — post-dehazing only |

---

## Datasets

| Dataset | Type | Notes |
|---------|------|-------|
| [RICE-I / RICE-II](https://github.com/BEEPLab/RICE) | Hazy satellite + ground truth | Most commonly used benchmark |
| [SateHaze1k](https://github.com/yuanli2333/Haze-Removal-Dataset) | Synthetic haze on satellite | 3 haze densities: thin/moderate/thick |
| RS-Haze | Remote sensing haze | Mixed terrain types |
| [Landsat-8](https://earthexplorer.usgs.gov/) | Real multispectral satellite | Free via USGS EarthExplorer |
| [Sentinel-2](https://scihub.copernicus.eu/) | Real multispectral satellite | Free via Copernicus |

For quick testing without datasets: Google Maps satellite screenshots of Delhi, Beijing, or LA (on bad air days) work well for visual evaluation.

---

## Comparison: Underwater vs Satellite

| Aspect | Underwater (existing) | Satellite (this project) |
|--------|-----------------------|--------------------------|
| Physical model | None (heuristic) | Atmospheric scattering model |
| Primary technique | CLAHE on V channel | Dark Channel Prior |
| Colour approach | Boost saturation (red lost in water) | No saturation boost (haze is white) |
| Sky regions | Not relevant | Must detect and handle separately |
| Transmission map | Not computed | Central to the pipeline |
| CLAHE role | Primary dehazing step | Post-processing polish only |
| CLAHE clip limit | 2.0 (aggressive) | 0.5–1.0 (light touch) |

---

## Known Limitations

1. **DCP sky failure** — White/bright sky regions always violate the dark channel prior.  The sky mask mitigates but does not fully solve this.  Expect residual artefacts at sky/scene boundaries in complex images.

2. **Dense thick cloud** — DCP and all classical single-image methods fail on opaque cloud cover.  This requires multi-temporal methods or deep learning approaches (SAR-DeCR, RSC-Net).  Out of scope for this project.

3. **Uniform haze** — If the entire image is uniformly hazy with no dark patches, the dark channel estimation of A becomes unreliable.

4. **Processing speed** — The guided filter is O(N), but on large satellite tiles (e.g. 10,000×10,000 px) the pipeline may take several seconds.  The dark channel uses `cv2.erode` as a fast approximation — avoid manual loops.

---

## Repository Structure

```
satellite-dehaze/
├── enhance.py                     # CLI entry point
├── requirements.txt
├── src/enhance/
│   ├── __init__.py
│   ├── pipeline.py                # PipelineConfig + run_pipeline()
│   ├── sky_mask.py                # Stage 0: sky/cloud region detection
│   ├── dark_channel.py            # Stage 1: DCP
│   ├── guided_filter.py           # Stage 2: transmission refinement
│   ├── scene_recovery.py          # Stage 3: invert scattering model
│   ├── contrast.py                # Stage 4: CLAHE + contrast stretch
│   ├── filtering.py               # Stage 5: sharpen + homomorphic
│   └── utils.py                   # show_pipeline(), show_comparison(), I/O
├── notebooks/
│   └── demo.ipynb
└── samples/
    └── README.md
```

---

## References

- He et al. (2009), *Single Image Haze Removal Using Dark Channel Prior*, CVPR
- He et al. (2013), *Guided Image Filtering*, TPAMI
- Pushpa et al. (2022), *Novel haze removal using multi-scale Retinex histogram equalization with U-Net Dense*, Earth Science Informatics
- Shen et al., *Haze and Thin Cloud Removal via Sphere Model Improved DCP*
