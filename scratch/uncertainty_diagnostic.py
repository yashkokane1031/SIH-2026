"""
MC-Dropout Uncertainty Distribution Diagnostic
================================================
Runs MC-Dropout inference on all 20 real CAMELYON16 slides and 50 synthetic
test slides, then compares uncertainty distributions across correct vs
incorrect predictions and real vs synthetic data.

This is a READ-ONLY diagnostic — no thresholds or pipeline code are changed.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from pipeline import (
    AttentionMIL,
    train_model,
    mc_dropout_inference,
    route_slides,
    load_real_camelyon16_bags,
    generate_synthetic_bags,
    SEED,
    set_seed,
    FEATURE_DIM,
    NUM_SLIDES_TRAIN,
    NUM_SLIDES_VAL,
    NUM_SLIDES_TEST,
    URGENT_PROB_THRESHOLD,
    ROUTINE_PROB_THRESHOLD,
    UNCERTAINTY_THRESHOLD,
)

set_seed(SEED)

# ============================================================
# 1. REAL CAMELYON16
# ============================================================
print("=" * 90)
print("LOADING & TRAINING ON REAL CAMELYON16 (20 SLIDES)")
print("=" * 90, flush=True)

feat_dir = BASE_DIR / "data" / "camelyon16_features"
train_bags, val_bags, test_bags = load_real_camelyon16_bags(str(feat_dir))
all_real_bags = train_bags + val_bags + test_bags

feature_dim = train_bags[0]["features"].shape[1]
real_model = AttentionMIL(feature_dim=feature_dim)
real_model = train_model(real_model, train_bags, val_bags)

print("\n[MC-DROPOUT] Running 20-pass inference on all 20 real slides...", flush=True)
real_results = mc_dropout_inference(real_model, all_real_bags)
real_results = route_slides(real_results)

# Tag each result with split membership
test_ids = {b["slide_id"] for b in test_bags}
val_ids = {b["slide_id"] for b in val_bags}
train_ids = {b["slide_id"] for b in train_bags}

for r in real_results:
    if r["slide_id"] in test_ids:
        r["split"] = "test"
    elif r["slide_id"] in val_ids:
        r["split"] = "val"
    else:
        r["split"] = "train"
    r["pred_correct"] = (r["mean_prob"] >= 0.5 and r["true_label"] == 1) or \
                        (r["mean_prob"] < 0.5 and r["true_label"] == 0)

# ============================================================
# 2. SYNTHETIC BENCHMARK
# ============================================================
print("\n" + "=" * 90)
print("LOADING & TRAINING ON SYNTHETIC BENCHMARK (550 SLIDES)")
print("=" * 90, flush=True)

set_seed(SEED)
syn_train = generate_synthetic_bags(NUM_SLIDES_TRAIN, label_prefix="train", start_idx=0)
syn_val = generate_synthetic_bags(NUM_SLIDES_VAL, label_prefix="val", start_idx=0)
syn_test = generate_synthetic_bags(NUM_SLIDES_TEST, label_prefix="test", start_idx=0)

syn_model = AttentionMIL(feature_dim=FEATURE_DIM)
syn_model = train_model(syn_model, syn_train, syn_val)

print("\n[MC-DROPOUT] Running 20-pass inference on 50 synthetic test slides...", flush=True)
syn_results = mc_dropout_inference(syn_model, syn_test)
syn_results = route_slides(syn_results)

for r in syn_results:
    r["split"] = "test"
    r["pred_correct"] = (r["mean_prob"] >= 0.5 and r["true_label"] == 1) or \
                        (r["mean_prob"] < 0.5 and r["true_label"] == 0)

# ============================================================
# UTILITY
# ============================================================
def distribution_stats(values, label=""):
    arr = np.array(values)
    if len(arr) == 0:
        print(f"  {label}: N=0 (no data)")
        return
    print(f"  {label} (N={len(arr)}):")
    print(f"    min    = {arr.min():.6f}")
    print(f"    max    = {arr.max():.6f}")
    print(f"    mean   = {arr.mean():.6f}")
    print(f"    median = {np.median(arr):.6f}")
    print(f"    std    = {arr.std():.6f}")
    print(f"    Q25    = {np.percentile(arr, 25):.6f}")
    print(f"    Q75    = {np.percentile(arr, 75):.6f}")

def print_sorted_table(results, sort_key="std_prob"):
    sorted_r = sorted(results, key=lambda x: x[sort_key])
    print(f"\n{'RANK':>4} {'SLIDE ID':<16} {'SPLIT':<6} {'TRUE':>4} {'PROB':>10} "
          f"{'UNCERT':>10} {'TIER':<10} {'CORRECT':>7}")
    print("-" * 85)
    for i, r in enumerate(sorted_r, 1):
        correct_str = "YES" if r["pred_correct"] else "NO"
        print(f"{i:>4} {r['slide_id']:<16} {r['split']:<6} {r['true_label']:>4} "
              f"{r['mean_prob']:>10.6f} {r['std_prob']:>10.6f} {r['tier']:<10} {correct_str:>7}")

# ============================================================
# STEP 1A: REAL DATA UNCERTAINTY DISTRIBUTION
# ============================================================
print("\n\n" + "#" * 90)
print("STEP 1A: REAL CAMELYON16 UNCERTAINTY DISTRIBUTION (ALL 20 SLIDES)")
print("#" * 90)

real_uncerts = [r["std_prob"] for r in real_results]
real_correct = [r["std_prob"] for r in real_results if r["pred_correct"]]
real_incorrect = [r["std_prob"] for r in real_results if not r["pred_correct"]]

distribution_stats(real_uncerts, "ALL 20 REAL SLIDES")
distribution_stats(real_correct, "CORRECT predictions (at 0.5 cutoff)")
distribution_stats(real_incorrect, "INCORRECT predictions (at 0.5 cutoff)")

print_sorted_table(real_results)

# ============================================================
# STEP 1B: SYNTHETIC DATA UNCERTAINTY DISTRIBUTION
# ============================================================
print("\n\n" + "#" * 90)
print("STEP 1B: SYNTHETIC BENCHMARK UNCERTAINTY DISTRIBUTION (50 TEST SLIDES)")
print("#" * 90)

syn_uncerts = [r["std_prob"] for r in syn_results]
syn_correct = [r["std_prob"] for r in syn_results if r["pred_correct"]]
syn_incorrect = [r["std_prob"] for r in syn_results if not r["pred_correct"]]

distribution_stats(syn_uncerts, "ALL 50 SYNTHETIC TEST SLIDES")
distribution_stats(syn_correct, "CORRECT predictions (at 0.5 cutoff)")
distribution_stats(syn_incorrect, "INCORRECT predictions (at 0.5 cutoff)")

print_sorted_table(syn_results)

# ============================================================
# STEP 2: COMPARATIVE ANALYSIS
# ============================================================
print("\n\n" + "#" * 90)
print("STEP 2: REAL vs SYNTHETIC UNCERTAINTY COMPARISON")
print("#" * 90)

real_arr = np.array(real_uncerts)
syn_arr = np.array(syn_uncerts)

print(f"\n  {'METRIC':<35} {'REAL (N=20)':>14} {'SYNTHETIC (N=50)':>16} {'RATIO (S/R)':>12}")
print("  " + "-" * 80)
for label, fn in [("Min", np.min), ("Max", np.max), ("Mean", np.mean),
                   ("Median", np.median), ("Std", np.std)]:
    rv = fn(real_arr)
    sv = fn(syn_arr)
    ratio = sv / rv if rv > 1e-10 else float("inf")
    print(f"  {label:<35} {rv:>14.6f} {sv:>16.6f} {ratio:>12.2f}x")

# Correlation: point-biserial between uncertainty and correctness
from scipy import stats as sp_stats

real_correct_binary = np.array([1 if r["pred_correct"] else 0 for r in real_results])
real_uncert_arr = np.array([r["std_prob"] for r in real_results])
try:
    pb_corr, pb_pval = sp_stats.pointbiserialr(real_correct_binary, real_uncert_arr)
    print(f"\n  Point-Biserial Correlation (uncertainty vs correctness, real data):")
    print(f"    r = {pb_corr:.4f}, p-value = {pb_pval:.4f}")
    if pb_pval < 0.05:
        print(f"    Interpretation: Statistically significant at p<0.05")
    else:
        print(f"    Interpretation: NOT statistically significant at p<0.05 (N=20 is very small)")
except Exception as e:
    print(f"\n  Point-biserial correlation failed: {e}")

syn_correct_binary = np.array([1 if r["pred_correct"] else 0 for r in syn_results])
syn_uncert_arr = np.array([r["std_prob"] for r in syn_results])
try:
    pb_corr_syn, pb_pval_syn = sp_stats.pointbiserialr(syn_correct_binary, syn_uncert_arr)
    print(f"\n  Point-Biserial Correlation (uncertainty vs correctness, synthetic data):")
    print(f"    r = {pb_corr_syn:.4f}, p-value = {pb_pval_syn:.4f}")
except Exception as e:
    print(f"\n  Point-biserial correlation (synthetic) failed: {e}")

# ============================================================
# STEP 2 CONTINUED: THRESHOLD SWEEP TRADEOFF
# ============================================================
print("\n\n" + "#" * 90)
print("STEP 2 CONTINUED: THRESHOLD SWEEP TRADEOFF ON REAL DATA")
print("#" * 90)

# What threshold would catch tumor_007 and tumor_010?
tumor_007 = next(r for r in real_results if r["slide_id"] == "tumor_007")
tumor_010 = next(r for r in real_results if r["slide_id"] == "tumor_010")
catch_threshold = min(tumor_007["std_prob"], tumor_010["std_prob"])

print(f"\n  tumor_007 uncertainty: {tumor_007['std_prob']:.6f}")
print(f"  tumor_010 uncertainty: {tumor_010['std_prob']:.6f}")
print(f"  To catch BOTH, threshold must be <= {catch_threshold:.6f}")

# Sweep thresholds
print(f"\n  {'THRESHOLD':>10} {'CAUGHT_MISSES':>14} {'FALSE_FLAGS':>12} {'TOTAL_UNCERTAIN':>16} "
      f"{'SLIDES_FLAGGED'}")
print("  " + "-" * 85)

# The 3 real misses (tumor_007 test, tumor_010 test, tumor_008 val)
real_misses = [r for r in real_results if r["true_label"] == 1 and not r["pred_correct"]]
miss_ids = {r["slide_id"] for r in real_misses}

thresholds_to_test = sorted(set(
    [0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.045, 0.04, 0.03, 0.02, 0.01] +
    [catch_threshold, catch_threshold - 0.001]
))

for thresh in thresholds_to_test:
    flagged = [r for r in real_results if r["std_prob"] >= thresh]
    caught_misses = [r for r in flagged if r["slide_id"] in miss_ids]
    false_flags = [r for r in flagged if r["slide_id"] not in miss_ids]
    flagged_ids = [r["slide_id"] for r in flagged]
    marker = " <-- catches both test misses" if all(
        any(r["slide_id"] == sid for r in flagged) for sid in ["tumor_007", "tumor_010"]
    ) else ""
    print(f"  {thresh:>10.4f} {len(caught_misses):>14} {len(false_flags):>12} "
          f"{len(flagged):>16}   {', '.join(flagged_ids)}{marker}")

# Also show: what if we used the "middle zone" approach (prob between 0.3 and 0.7)?
print("\n\n  ALTERNATIVE: Flag slides in the 'middle probability zone' (0.30 < prob < 0.70):")
print(f"  {'SLIDE ID':<16} {'TRUE':>4} {'PROB':>10} {'UNCERT':>10} {'CORRECT':>7} {'CURRENT TIER':<10}")
print("  " + "-" * 70)
for r in sorted(real_results, key=lambda x: x["mean_prob"]):
    if 0.30 < r["mean_prob"] < 0.70:
        correct_str = "YES" if r["pred_correct"] else "NO"
        print(f"  {r['slide_id']:<16} {r['true_label']:>4} {r['mean_prob']:>10.6f} "
              f"{r['std_prob']:>10.6f} {correct_str:>7} {r['tier']:<10}")

print("\n\nDIAGNOSTIC COMPLETE. No code was modified.", flush=True)
