"""
CAMELYON16 Streaming WSI Ingestion & Feature Extraction Pipeline

Downloads 1 slide at a time, segments tissue via Otsu thresholding,
tiles at 20x magnification, applies Macenko stain normalization via TorchMacenkoNormalizer
(fitted ONCE on a reference patch), extracts 2048-d ResNet-50 features,
caches features to .npz, and immediately deletes the raw WSI.
"""

import os
import sys
import time
import shutil
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import requests
import openslide
from torchstain.torch.normalizers import TorchMacenkoNormalizer


# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMP_WSI_DIR = DATA_DIR / "temp_wsi"
FEATURES_DIR = DATA_DIR / "camelyon16_features"
THUMBNAILS_DIR = DATA_DIR / "camelyon16_thumbnails"
REFERENCE_CSV = DATA_DIR / "camelyon16_reference.csv"

PATCH_SIZE_20X = 256           # Patch resolution at 20x
TISSUE_THRESHOLD = 0.40        # Minimum tissue pixel fraction to keep patch
BATCH_SIZE = 32                # ResNet-50 feature extraction batch size
MAX_PATCHES_PER_SLIDE = 400    # Cap patches per slide for compute efficiency in prototype


# ============================================================================
# 1. DISK UTILS & DOWNLOADER
# ============================================================================

def get_disk_info(path=BASE_DIR):
    """Return free and total disk space in GB."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3), total / (1024 ** 3)


def download_slide(url: str, dest_path: Path) -> bool:
    """Stream download a WSI file with progress tracking."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_download = dest_path.with_suffix(".downloading")

    print(f"\n[DOWNLOAD] Fetching {dest_path.name} from:")
    print(f"           {url}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))
        total_mb = total_size / (1024 * 1024)
        print(f"[DOWNLOAD] File size: {total_mb:.1f} MB")

        downloaded = 0
        start_time = time.time()
        chunk_size = 1024 * 1024 * 2  # 2MB chunks

        with open(temp_download, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    speed_mb = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
                    pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    print(
                        f"\r  Progress: {downloaded / (1024*1024):.1f}/{total_mb:.1f} MB "
                        f"({pct:.1f}%) | Speed: {speed_mb:.1f} MB/s",
                        end="",
                        flush=True,
                    )

        print("\n[DOWNLOAD] Download complete.")
        if temp_download.exists():
            temp_download.replace(dest_path)
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to download {dest_path.name}: {e}")
        if temp_download.exists():
            temp_download.unlink()
        if dest_path.exists():
            dest_path.unlink()
        return False


# ============================================================================
# 2. TISSUE MASKING (OTSU THRESHOLDING)
# ============================================================================

def compute_otsu_threshold(gray_img: np.ndarray) -> int:
    """Compute Otsu threshold on a grayscale uint8 image array."""
    hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
    total = gray_img.size
    current_max, threshold = 0.0, 0
    sum_total = np.dot(np.arange(256), hist)

    sum_b, weight_b = 0.0, 0
    for i in range(256):
        weight_b += hist[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / weight_b
        m_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (m_b - m_f) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = i

    return threshold


def generate_tissue_mask(slide: openslide.OpenSlide, max_thumb_dim: int = 2048):
    """
    Generate a binary tissue mask from a low-res thumbnail using Otsu thresholding.
    Returns (tissue_mask_2d_bool, (scale_x, scale_y), thumbnail_rgb).
    """
    level_0_w, level_0_h = slide.dimensions
    scale = max(level_0_w / max_thumb_dim, level_0_h / max_thumb_dim, 1.0)
    thumb_w = int(level_0_w / scale)
    thumb_h = int(level_0_h / scale)

    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    thumb_np = np.array(thumb)

    # In H&E, background glass is bright white / gray (> 210)
    # Convert to grayscale and apply Otsu thresholding
    gray = np.array(thumb.convert("L"))
    otsu_thresh = compute_otsu_threshold(gray)
    # Clip upper threshold to ensure light background is eliminated
    effective_thresh = min(otsu_thresh, 215)
    tissue_mask = gray < effective_thresh

    scale_x = level_0_w / thumb_w
    scale_y = level_0_h / thumb_h

    return tissue_mask, (scale_x, scale_y), thumb_np


# ============================================================================
# 3. PATCH EXTRACTION & STAIN NORMALIZATION
# ============================================================================

def fit_reference_normalizer(slide: openslide.OpenSlide, coords_list, patch_lvl0_size):
    """
    Find the first high-optical-density H&E tissue patch and fit TorchMacenkoNormalizer.
    TorchMacenkoNormalizer expects tensors of shape [C, H, W] uint8.
    """
    print("[STAIN] Fitting TorchMacenkoNormalizer on reference patch...")
    normalizer = TorchMacenkoNormalizer()
    fitted = False

    for x0, y0 in coords_list:
        try:
            patch_rgba = slide.read_region((x0, y0), 0, (patch_lvl0_size, patch_lvl0_size))
            patch_rgb = patch_rgba.convert("RGB").resize((PATCH_SIZE_20X, PATCH_SIZE_20X), Image.Resampling.BILINEAR)
            patch_arr = np.array(patch_rgb, dtype=np.uint8)

            # Check if patch has distinct H&E color characteristics (red > green, blue > green)
            r, g, b = patch_arr[:, :, 0], patch_arr[:, :, 1], patch_arr[:, :, 2]
            if np.mean(r) > np.mean(g) and np.mean(b) > np.mean(g) and np.std(r) > 15:
                # Convert (H, W, C) -> (C, H, W) for torchstain
                ref_tensor = torch.from_numpy(patch_arr).permute(2, 0, 1)
                normalizer.fit(ref_tensor)
                fitted = True
                print(f"  [+] Reference normalizer successfully fitted on patch at ({x0}, {y0})")
                break
        except Exception:
            continue

    if not fitted:
        # Fallback to a canonical H&E reference profile
        print("  [!] Slide patches ambiguous; fitting on canonical H&E reference profile")
        synth_ref = torch.zeros(3, PATCH_SIZE_20X, PATCH_SIZE_20X, dtype=torch.uint8)
        synth_ref[0, :, :] = 185  # Hematoxylin / Eosin pink
        synth_ref[1, :, :] = 85
        synth_ref[2, :, :] = 165
        normalizer.fit(synth_ref)

    return normalizer


def normalize_patch(patch_arr: np.ndarray, normalizer: TorchMacenkoNormalizer) -> np.ndarray:
    """Normalize a single (256, 256, 3) uint8 patch using the fitted TorchMacenkoNormalizer."""
    try:
        # torchstain expects (C, H, W) tensor
        t_patch = torch.from_numpy(patch_arr).permute(2, 0, 1)
        norm_t, _, _ = normalizer.normalize(I=t_patch, stains=False)
        if isinstance(norm_t, torch.Tensor):
            norm_np = norm_t.cpu().numpy()
            return np.clip(norm_np, 0, 255).astype(np.uint8)
        return patch_arr
    except Exception:
        # Fallback to original patch if optical density inversion fails for edge artifacts
        return patch_arr


# ============================================================================
# 4. RESNET-50 FEATURE EXTRACTOR
# ============================================================================

def build_resnet_encoder():
    """Load pretrained ResNet-50 with final classification layer replaced by Identity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] Loading pretrained ResNet-50 feature encoder on {device}...")

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()  # Outputs 2048-d pooled features
    model.eval()
    model.to(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess, device


# ============================================================================
# 5. SINGLE-SLIDE PROCESSOR
# ============================================================================

def process_single_slide(
    slide_row: dict,
    normalizer: TorchMacenkoNormalizer,
    resnet_model,
    preprocess,
    device,
    force: bool = False,
) -> tuple[bool, TorchMacenkoNormalizer, dict]:
    """
    Download, segment, tile, normalize, extract ResNet features, save .npz, and delete raw WSI.
    """
    slide_id = slide_row["slide_id"]
    filename = slide_row["filename"]
    label = int(slide_row["label"])
    source_url = slide_row["source_url"]

    wsi_path = TEMP_WSI_DIR / filename
    out_npz_path = FEATURES_DIR / f"{slide_id}.npz"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    free_start, _ = get_disk_info()
    slide_start_time = time.time()

    # Skip if already cached
    if out_npz_path.exists() and not force:
        try:
            data = np.load(out_npz_path)
            num_patches = data["features"].shape[0]
            feature_dim = data["features"].shape[1]
            npz_size_kb = out_npz_path.stat().st_size / 1024
            print(f"\n[CACHE HIT] {slide_id}.npz already extracted ({num_patches} patches, {feature_dim}d). Skipping.")
            diag = {
                "slide_id": slide_id,
                "label": label,
                "raw_size_mb": 0.0,
                "npz_size_kb": npz_size_kb,
                "num_patches": num_patches,
                "feature_dim": feature_dim,
                "elapsed_s": 0.0,
                "free_disk_gb": free_start,
            }
            return True, normalizer, diag
        except Exception:
            print(f"[WARNING] Existing {out_npz_path.name} was corrupt. Re-downloading and extracting...")

    print("\n" + "=" * 70)
    print(f"PROCESSING SLIDE: {slide_id} ({'TUMOR' if label == 1 else 'NORMAL'})")
    print(f"Disk Free Before: {free_start:.2f} GB")
    print("=" * 70)

    # Step 1: Download
    if not download_slide(source_url, wsi_path):
        return False, normalizer, {}

    raw_file_size_mb = wsi_path.stat().st_size / (1024 * 1024)

    # Step 2: Open with OpenSlide & Magnification Check
    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as e:
        print(f"[ERROR] OpenSlide failed to read {filename}: {e}")
        if wsi_path.exists():
            wsi_path.unlink()
        return False, normalizer, {}

    level_0_w, level_0_h = slide.dimensions
    obj_power = float(slide.properties.get("openslide.objective-power", 40.0))
    # If 40x on level 0, patch size on level 0 is 512 to get 20x equivalent
    patch_lvl0_size = int(PATCH_SIZE_20X * (obj_power / 20.0))
    print(f"[SLIDE] Dimensions: {level_0_w}x{level_0_h} | Objective Power: {obj_power}x")
    print(f"[SLIDE] Level 0 Patch Step: {patch_lvl0_size}px (Target 20x: {PATCH_SIZE_20X}px)")

    # Step 3: Tissue Masking
    tissue_mask, (scale_x, scale_y), thumb_np = generate_tissue_mask(slide)
    mask_h, mask_w = tissue_mask.shape

    # Step 4: Generate Candidate Grid Coordinates
    valid_coords = []
    for y0 in range(0, level_0_h - patch_lvl0_size, patch_lvl0_size):
        for x0 in range(0, level_0_w - patch_lvl0_size, patch_lvl0_size):
            # Check corresponding region in thumbnail tissue mask
            mx_start = int(x0 / scale_x)
            my_start = int(y0 / scale_y)
            mx_end = min(int((x0 + patch_lvl0_size) / scale_x), mask_w)
            my_end = min(int((y0 + patch_lvl0_size) / scale_y), mask_h)

            sub_mask = tissue_mask[my_start:my_end, mx_start:mx_end]
            if sub_mask.size > 0 and (np.mean(sub_mask) >= TISSUE_THRESHOLD):
                valid_coords.append((x0, y0))

    num_candidates = len(valid_coords)
    print(f"[TISSUE] Detected {num_candidates} valid tissue regions.")

    if num_candidates == 0:
        print("[WARNING] No tissue patches passed threshold. Releasing slide.")
        slide.close()
        wsi_path.unlink()
        return False, normalizer, {}

    # Sample patches if exceeding prototype cap
    if len(valid_coords) > MAX_PATCHES_PER_SLIDE:
        indices = np.linspace(0, len(valid_coords) - 1, MAX_PATCHES_PER_SLIDE, dtype=int)
        selected_coords = [valid_coords[i] for i in indices]
    else:
        selected_coords = valid_coords

    num_patches = len(selected_coords)

    # Step 5: Stain Normalization Fit (if not fitted yet)
    if normalizer is None:
        normalizer = fit_reference_normalizer(slide, selected_coords, patch_lvl0_size)

    # Step 6: Extract Patches, Normalize, and Encode with ResNet-50
    print(f"[EXTRACT] Processing {num_patches} patches (tiling + Macenko norm + ResNet-50)...")
    all_features = []
    grid_coords = []

    batch_tensors = []
    batch_coords = []

    for i, (x0, y0) in enumerate(selected_coords):
        try:
            patch_rgba = slide.read_region((x0, y0), 0, (patch_lvl0_size, patch_lvl0_size))
            patch_rgb = patch_rgba.convert("RGB").resize(
                (PATCH_SIZE_20X, PATCH_SIZE_20X), Image.Resampling.BILINEAR
            )
            patch_arr = np.array(patch_rgb, dtype=np.uint8)

            # Stain normalize
            norm_arr = normalize_patch(patch_arr, normalizer)
            norm_pil = Image.fromarray(norm_arr)

            t = preprocess(norm_pil)
            batch_tensors.append(t)
            # Normalize coordinates to 0..grid_dim for attention heatmap
            grid_coords.append([x0 // patch_lvl0_size, y0 // patch_lvl0_size])

            if len(batch_tensors) == BATCH_SIZE or i == num_patches - 1:
                inp = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    feats = resnet_model(inp)  # (B, 2048)
                all_features.append(feats.cpu().numpy())
                batch_tensors = []

        except Exception as e:
            continue

    slide.close()

    # Step 6b: Save Low-Res Thumbnail (reuses Otsu tissue-masking thumbnail already in memory)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMBNAILS_DIR / f"{slide_id}.jpg"
    try:
        thumb_img = Image.fromarray(thumb_np)
        thumb_img.save(thumb_path, format="JPEG", quality=85, optimize=True)
        thumb_size_kb = thumb_path.stat().st_size / 1024
        print(f"[THUMBNAIL] Saved {thumb_path.name} ({thumb_size_kb:.1f} KB)")
    except Exception as e:
        thumb_size_kb = 0.0
        print(f"[WARNING] Failed to save thumbnail for {slide_id}: {e}")

    # Step 7: Delete Raw WSI Immediately
    if wsi_path.exists():
        wsi_path.unlink()
        print(f"[CLEANUP] Deleted raw WSI ({raw_file_size_mb:.1f} MB freed).")

    if len(all_features) == 0:
        print("[ERROR] Feature extraction yielded 0 features.")
        return False, normalizer, {}

    features_matrix = np.concatenate(all_features, axis=0).astype(np.float32)
    coords_matrix = np.array(grid_coords, dtype=np.int32)

    # Step 8: Save Compressed .npz
    np.savez_compressed(
        out_npz_path,
        features=features_matrix,
        coordinates=coords_matrix,
        label=label,
        slide_id=slide_id,
    )
    npz_size_kb = out_npz_path.stat().st_size / 1024

    free_end, _ = get_disk_info()
    elapsed = time.time() - slide_start_time

    diag = {
        "slide_id": slide_id,
        "label": label,
        "raw_size_mb": raw_file_size_mb,
        "npz_size_kb": npz_size_kb,
        "thumb_size_kb": thumb_size_kb,
        "num_patches": features_matrix.shape[0],
        "feature_dim": features_matrix.shape[1],
        "elapsed_s": elapsed,
        "free_disk_gb": free_end,
    }

    print("-" * 70)
    print(f"[SUCCESS] {slide_id} complete in {elapsed:.1f}s")
    print(f"  Features shape: {features_matrix.shape} (2048-d ResNet-50)")
    print(f"  Saved to: {out_npz_path.name} ({npz_size_kb:.1f} KB)")
    print(f"  Net Disk Impact: {free_end - free_start:+.2f} GB (Current Free: {free_end:.2f} GB)")
    print("-" * 70)

    return True, normalizer, diag


# ============================================================================
# 6. MAIN ORCHESTRATOR
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CAMELYON16 Download & Feature Extractor")
    parser.add_argument("--dry-run", action="store_true", help="Process ONLY the first slide and stop")
    parser.add_argument("--slide-id", type=str, default=None, help="Process a specific slide ID (e.g. tumor_001)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of slides to process")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-extract even if .npz already exists")
    args = parser.parse_args()

    print("=" * 70)
    print("CAMELYON16 STREAMING INGESTION & RESNET-50 FEATURE EXTRACTION")
    print(f"Reference CSV: {REFERENCE_CSV}")
    print("=" * 70)

    if not REFERENCE_CSV.exists():
        print(f"[ERROR] Reference CSV not found: {REFERENCE_CSV}")
        sys.exit(1)

    with open(REFERENCE_CSV, "r") as f:
        reader = csv.DictReader(f)
        slides = list(reader)

    if args.slide_id:
        slides = [s for s in slides if s["slide_id"] == args.slide_id or s["filename"] == args.slide_id]
        if not slides:
            print(f"[ERROR] Slide ID '{args.slide_id}' not found in reference CSV.")
            sys.exit(1)
        print(f"[MODE] Single slide targeted: {slides[0]['slide_id']}")
    elif args.dry_run:
        slides = slides[:1]
        print("[MODE] Dry run enabled — will process ONLY normal_001.tif and stop.")
    elif args.limit:
        slides = slides[:args.limit]

    # Initialize ResNet-50
    resnet_model, preprocess, device = build_resnet_encoder()

    normalizer = None
    successful_slides = 0
    all_diagnostics = []

    for idx, slide_row in enumerate(slides, 1):
        print(f"\n>>> Slide {idx}/{len(slides)}: {slide_row['slide_id']} ({slide_row['filename']})")
        success, normalizer, diag = process_single_slide(
            slide_row=slide_row,
            normalizer=normalizer,
            resnet_model=resnet_model,
            preprocess=preprocess,
            device=device,
            force=args.force,
        )
        if success:
            successful_slides += 1
            all_diagnostics.append(diag)
        else:
            print(f"[WARNING] Skipping {slide_row['slide_id']} due to processing failure.")

    print("\n" + "=" * 70)
    print(f"INGESTION SUMMARY: {successful_slides}/{len(slides)} slides processed successfully.")
    print("=" * 70)
    for d in all_diagnostics:
        print(
            f"  {d['slide_id']:<12} | Label={d['label']} | Patches={d['num_patches']:>4} | "
            f"Feats={d['feature_dim']}d | Saved={d['npz_size_kb']:>6.1f} KB | Time={d['elapsed_s']:>4.1f}s"
        )


if __name__ == "__main__":
    main()
