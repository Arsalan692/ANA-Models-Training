"""
Phase 3 -- Preprocessing: raw AIDA images  ->  uniform 512-px RGB PNGs

For every row in manifest.csv:
  1. open the raw file (bmp / jpg / png / tif, various colour modes)
  2. convert to 3-channel RGB
  3. resize so the SHORTER side is 512 px, keeping the aspect ratio
  4. save as lossless PNG at data/processed/<image_path>

Deliberately NOT done: contrast stretching / histogram equalisation / any
per-image brightness change -- brightness is the Positive/Negative signal.

Finally zips data/processed + manifest.csv into data/AIDA_processed.zip for
upload to our PRIVATE Google Drive (AIDA licence: no redistribution).

Run:  python scripts/preprocess.py
"""
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

import config

ZIP_PATH = config.DATA_DIR / "AIDA_processed.zip"


def process_one(job):
    raw_file, image_path = job
    out_path = config.PROCESSED_DIR / image_path
    with Image.open(config.AIDA_RAW_DIR / raw_file) as im:
        # 'P' (palette) and 'RGBA' (with transparency) files -> plain RGB.
        im = im.convert("RGB")
        w, h = im.size
        scale = config.SHORT_SIDE / min(w, h)
        new_size = (round(w * scale), round(h * scale))
        # LANCZOS = high-quality resampling filter: averages neighbouring pixels
        # properly when shrinking, so small bright dots are not lost or aliased.
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="PNG")
        green_mean = np.asarray(im)[:, :, 1].mean()
    return {"image_path": image_path, "orig_w": w, "orig_h": h,
            "new_w": new_size[0], "new_h": new_size[1],
            "upscaled": scale > 1, "green_mean_after": green_mean}


def main():
    manifest = pd.read_csv(config.MANIFEST_PATH)
    file_map = pd.read_csv(config.DATA_DIR / "aida_file_map.csv")
    jobs = manifest.merge(file_map, on="image_path")[["raw_file", "image_path"]]

    print(f"Processing {len(jobs)} images -> {config.PROCESSED_DIR} ...")
    with ProcessPoolExecutor() as pool:
        results = pd.DataFrame(pool.map(process_one, jobs.itertuples(index=False, name=None), chunksize=16))

    # ------------------------------------------------ checks
    files = list(config.PROCESSED_DIR.rglob("*.png"))
    assert len(files) == len(manifest), f"{len(files)} PNGs for {len(manifest)} manifest rows"
    assert (results[["new_w", "new_h"]].min(axis=1) == config.SHORT_SIDE).all()

    # Brightness must be unchanged by preprocessing: compare with Phase 2 measurements
    eda = pd.read_csv(config.OUTPUTS_DIR / "eda" / "image_stats.csv")[["image_path", "green_mean"]]
    cmp = results.merge(eda, on="image_path")
    diff = (cmp["green_mean_after"] - cmp["green_mean"]).abs()

    sizes = (results["new_w"].astype(str) + "x" + results["new_h"].astype(str)).value_counts()
    report = [
        f"Processed images : {len(results)}",
        f"Output sizes     : {sizes.to_dict()}",
        f"Upscaled (original shorter side < 512): {int(results['upscaled'].sum())}",
        f"Brightness change vs original (0-255 scale): mean {diff.mean():.2f}, max {diff.max():.2f}",
    ]

    report.append(build_zip())
    (config.OUTPUTS_DIR / "phase3_preprocess_log.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


def build_zip():
    """Pack manifest.csv + all processed PNGs into one file for Google Drive.

    Called again after Phase 4 so the zip always contains the latest manifest.
    """
    files = sorted(config.PROCESSED_DIR.rglob("*.png"))
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as zf:
        # PNG is already compressed, so ZIP_STORED (no extra compression) is faster
        zf.write(config.MANIFEST_PATH, "manifest.csv")
        for f in files:
            zf.write(f, f.relative_to(config.DATA_DIR).as_posix())
    return f"Zip for Drive    : {ZIP_PATH.name}  ({ZIP_PATH.stat().st_size / 1e6:.0f} MB, {len(files)} images)"


if __name__ == "__main__":
    main()
