"""
Backfill CAMELYON16 Thumbnails
==============================
For all 20 slides that were already processed (raw .tifs deleted),
re-downloads each .tif, generates the same low-res Otsu thumbnail,
saves it as JPEG, and immediately deletes the raw .tif again.

This does NOT re-run patching, stain normalization, or feature extraction.
It only needs OpenSlide for the thumbnail call (~1-2 seconds per slide
after download).

Estimated cost:
  - 20 slides × ~250-600 MB each = ~6-8 GB total bandwidth
  - Per-slide: download time dominates (~30-120s depending on connection)
  - Processing per slide: <2s (just open slide + get_thumbnail + save JPEG)
  - Total estimated wall time: 15-40 minutes depending on download speed
"""

import csv
import time
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse the ingestion pipeline's functions directly
from download_and_extract_camelyon16 import (
    BASE_DIR,
    DATA_DIR,
    TEMP_WSI_DIR,
    THUMBNAILS_DIR,
    REFERENCE_CSV,
    download_slide,
    generate_tissue_mask,
    get_disk_info,
)

try:
    import openslide
except ImportError:
    print("[ERROR] openslide-python is required. Install with: pip install openslide-python")
    raise


def backfill_thumbnail(slide_row: dict) -> dict:
    """Download a single slide, save its thumbnail, delete the raw .tif."""
    slide_id = slide_row["slide_id"]
    filename = slide_row["filename"]
    source_url = slide_row["source_url"]
    label = int(slide_row["label"])

    thumb_path = THUMBNAILS_DIR / f"{slide_id}.jpg"

    # Skip if thumbnail already exists
    if thumb_path.exists():
        size_kb = thumb_path.stat().st_size / 1024
        print(f"  [SKIP] {slide_id}.jpg already exists ({size_kb:.1f} KB)")
        return {"slide_id": slide_id, "status": "cached", "thumb_size_kb": size_kb}

    wsi_path = TEMP_WSI_DIR / filename
    TEMP_WSI_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Download
    if not download_slide(source_url, wsi_path):
        return {"slide_id": slide_id, "status": "download_failed", "thumb_size_kb": 0}

    raw_mb = wsi_path.stat().st_size / (1024 * 1024)

    # Open with OpenSlide, generate thumbnail only
    try:
        slide = openslide.OpenSlide(str(wsi_path))
        _, _, thumb_np = generate_tissue_mask(slide)
        slide.close()
        del slide
    except Exception as e:
        print(f"  [ERROR] OpenSlide failed for {slide_id}: {e}")
        if wsi_path.exists():
            try:
                wsi_path.unlink()
            except Exception:
                pass
        return {"slide_id": slide_id, "status": "openslide_error", "thumb_size_kb": 0}

    import gc
    gc.collect()

    # Save thumbnail
    thumb_img = Image.fromarray(thumb_np)
    thumb_img.save(thumb_path, format="JPEG", quality=85, optimize=True)
    thumb_kb = thumb_path.stat().st_size / 1024

    # Keep one raw WSI file for inspection/demo if not already preserved
    sample_dir = DATA_DIR / "sample_wsi"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_tifs = list(sample_dir.glob("*.tif"))
    if len(sample_tifs) == 0 and wsi_path.exists():
        preserved_path = sample_dir / filename
        shutil.copy2(wsi_path, preserved_path)
        print(f"  [SAMPLE WSI] Preserved copy of {filename} ({raw_mb:.1f} MB) in data/sample_wsi/")

    # Delete raw WSI from temp with retry on Windows file lock
    if wsi_path.exists():
        for retry in range(5):
            try:
                wsi_path.unlink()
                break
            except PermissionError:
                gc.collect()
                time.sleep(1.0)
            except Exception:
                break

    elapsed = time.time() - start
    print(f"  [OK] {slide_id}.jpg saved ({thumb_kb:.1f} KB) | "
          f"Raw: {raw_mb:.0f} MB | Time: {elapsed:.1f}s")

    return {
        "slide_id": slide_id,
        "status": "ok",
        "thumb_size_kb": thumb_kb,
        "raw_mb": raw_mb,
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 70)
    print("CAMELYON16 THUMBNAIL BACKFILL")
    print(f"Reference CSV: {REFERENCE_CSV}")
    print(f"Output Dir:    {THUMBNAILS_DIR}")
    print("=" * 70)

    with open(REFERENCE_CSV, "r") as f:
        slides = list(csv.DictReader(f))

    print(f"\n{len(slides)} slides in reference CSV.\n")

    results = []
    for idx, row in enumerate(slides, 1):
        print(f"\n>>> [{idx}/{len(slides)}] {row['slide_id']}")
        result = backfill_thumbnail(row)
        results.append(result)

    # Summary
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    total_kb = sum(r["thumb_size_kb"] for r in ok)

    print("\n" + "=" * 70)
    print(f"BACKFILL COMPLETE: {len(ok)}/{len(slides)} thumbnails saved")
    print(f"Total thumbnail disk usage: {total_kb:.1f} KB ({total_kb / 1024:.2f} MB)")
    print("=" * 70)

    print(f"\n{'SLIDE ID':<14} {'STATUS':<18} {'THUMB SIZE':>10}")
    print("-" * 46)
    for r in results:
        size_str = f"{r['thumb_size_kb']:.1f} KB" if r["thumb_size_kb"] > 0 else "—"
        print(f"{r['slide_id']:<14} {r['status']:<18} {size_str:>10}")


if __name__ == "__main__":
    main()
