#!/usr/bin/env python3
"""
enhance.py — CLI entry point for the Satellite Image Dehazing pipeline
=======================================================================

Usage examples
--------------
  # Basic dehazing (defaults)
  python enhance.py input.jpg

  # Specify output path
  python enhance.py input.jpg --output results/dehazed.jpg

  # Show all pipeline stages
  python enhance.py input.jpg --show-pipeline

  # Save a before/after comparison
  python enhance.py input.jpg --save-comparison results/compare.png

  # Tune key parameters
  python enhance.py input.jpg --omega 0.85 --t-min 0.1 --clahe-clip 1.0

  # Disable optional stages
  python enhance.py input.jpg --no-clahe --no-sharpen

  # Verbose logging
  python enhance.py input.jpg --verbose
"""

import argparse
import os
import sys

from src.enhance import PipelineConfig, run_pipeline
from src.enhance.utils import load_image, save_result, show_comparison, show_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Satellite image dehazing via Dark Channel Prior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("input", help="Path to the input hazy image")
    p.add_argument("--output", "-o", default=None,
                   help="Output path (default: <input_stem>_dehazed.<ext>)")

    p.add_argument("--show-pipeline", action="store_true",
                   help="Display all intermediate pipeline stages")
    p.add_argument("--show-comparison", action="store_true",
                   help="Display side-by-side before/after comparison")
    p.add_argument("--save-comparison", default=None, metavar="PATH",
                   help="Save before/after comparison image to PATH")
    p.add_argument("--save-pipeline", default=None, metavar="PATH",
                   help="Save pipeline stage grid to PATH")

    g0 = p.add_argument_group("Stage 0 — Sky mask")
    g0.add_argument("--sky-brightness", type=int, default=220)
    g0.add_argument("--sky-saturation", type=int, default=30)

    g1 = p.add_argument_group("Stage 1 — Dark Channel Prior")
    g1.add_argument("--patch-size", type=int, default=15)
    g1.add_argument("--omega", type=float, default=0.95)
    g1.add_argument("--top-percent", type=float, default=0.001)

    g2 = p.add_argument_group("Stage 2 — Guided filter")
    g2.add_argument("--gf-radius", type=int, default=40)
    g2.add_argument("--gf-eps", type=float, default=1e-3)

    g3 = p.add_argument_group("Stage 3 — Scene recovery")
    g3.add_argument("--t-min", type=float, default=0.1)

    g4 = p.add_argument_group("Stage 4 — CLAHE")
    g4.add_argument("--no-clahe", action="store_true")
    g4.add_argument("--clahe-clip", type=float, default=0.75)

    g5 = p.add_argument_group("Stage 5 — Contrast & sharpen")
    g5.add_argument("--no-contrast-stretch", action="store_true")
    g5.add_argument("--no-sharpen", action="store_true")

    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading: {args.input}")
    image = load_image(args.input)

    config = PipelineConfig(
        sky_brightness_thresh=args.sky_brightness,
        sky_sat_thresh=args.sky_saturation,
        dcp_patch_size=args.patch_size,
        dcp_omega=args.omega,
        dcp_top_percent=args.top_percent,
        gf_radius=args.gf_radius,
        gf_eps=args.gf_eps,
        t_min=args.t_min,
        clahe_enabled=not args.no_clahe,
        clahe_clip_limit=args.clahe_clip,
        contrast_stretch_enabled=not args.no_contrast_stretch,
        sharpen_enabled=not args.no_sharpen,
    )

    if args.verbose:
        print("Running pipeline …")
    result, stages = run_pipeline(image, config, verbose=args.verbose)

    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}_dehazed{ext}"

    save_result(result, out_path)

    if args.show_comparison or args.save_comparison:
        show_comparison(
            original=image,
            dehazed=result,
            title=f"Satellite Dehazing — {os.path.basename(args.input)}",
            save_path=args.save_comparison,
        )

    if args.show_pipeline or args.save_pipeline:
        show_pipeline(stages, save_path=args.save_pipeline)


if __name__ == "__main__":
    main()
