# Sample Images

Place test images in this directory for quick evaluation.

## Recommended Datasets

| Dataset | URL | Notes |
|---------|-----|-------|
| RICE-I / RICE-II | https://github.com/BEEPLab/RICE | Hazy satellite + ground truth pairs.  Most commonly used benchmark. |
| SateHaze1k | https://github.com/yuanli2333/Haze-Removal-Dataset | Synthetic haze at 3 densities: thin / moderate / thick |
| RS-Haze | https://github.com/IDKiro/DehazeFormer | Remote sensing haze, mixed terrain |
| Landsat-8 | https://earthexplorer.usgs.gov/ | Free real multispectral imagery via USGS EarthExplorer |
| Sentinel-2 | https://scihub.copernicus.eu/ | Free real multispectral imagery via Copernicus Open Access Hub |

## Quick Test Without Downloading Datasets

Google Maps satellite screenshots of areas with known haze or smog work well
for visual evaluation (no ground-truth metrics, but good for qualitative checks):

- **Delhi, India** — persistent particulate haze year-round
- **Beijing, China** — seasonal smog events
- **Los Angeles, CA** — summer photochemical smog
- **Jakarta, Indonesia** — peat fire haze (seasonal)

Screenshot at zoom level 12–14 for a good mix of urban, vegetation, and sky.

## Running on a Sample

```bash
python enhance.py samples/my_hazy_image.jpg \
    --save-comparison samples/my_hazy_image_compare.png \
    --verbose
```
