"""
Simulated Domain Shift / Multi-Site Staining Robustness Analysis

Proxy test evaluating model resilience against simulated scanner and staining
variations (affine feature-space transformations) on both the real CAMELYON16
test set (4 slides) and the synthetic benchmark test set (50 slides).

NOTE: This is explicitly a SIMULATED robustness proxy test, not real multi-center
clinical data validation.
"""

import sys
import copy
from pathlib import Path

# Add project root to path
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
    MC_DROPOUT_RUNS,
)


def apply_simulated_domain_shift(
    bags: list[dict],
    scale_std: float = 0.15,
    shift_std: float = 0.10,
    seed: int = 12345,
) -> list[dict]:
    """
    Apply a slide-consistent synthetic affine feature shift to simulate
    inter-laboratory staining and scanner profile variations.

    Transformation per slide:
        F_shifted = s * F + b
    where:
        s ~ Normal(1.0, scale_std), clipped to [0.70, 1.30]
        b ~ Normal(0.0, shift_std) across feature dimensions
    """
    rng = np.random.RandomState(seed)
    shifted_bags = []

    for bag in bags:
        shifted_bag = copy.deepcopy(bag)
        feats = bag["features"].astype(np.float32)
        n_patches, feat_dim = feats.shape

        # Sample per-slide scale and shift vector
        s = float(np.clip(rng.normal(1.0, scale_std), 0.70, 1.30))
        b = rng.normal(0.0, shift_std, size=(1, feat_dim)).astype(np.float32)

        # Apply affine transformation consistently across all patches
        shifted_feats = s * feats + b

        shifted_bag["features"] = shifted_feats
        shifted_bag["shift_metadata"] = {
            "scale_factor": round(s, 4),
            "shift_norm": round(float(np.linalg.norm(b)), 4),
        }
        shifted_bags.append(shifted_bag)

    return shifted_bags


def evaluate_cohort(model, bags: list[dict]):
    """Run MC-Dropout inference, route slides into tiers, and compute ROC-AUC."""
    results = mc_dropout_inference(model, bags)
    results = route_slides(results)
    y_true = [r["true_label"] for r in results]
    y_prob = [r["mean_prob"] for r in results]
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return results, auc


def print_comparison_table(title: str, orig_results: list[dict], shift_results: list[dict], orig_auc: float, shift_auc: float):
    """Print formatted side-by-side comparison table."""
    print("\n" + "=" * 95)
    print(f" {title.upper()} — ORIGINAL VS. SIMULATED SHIFT COMPARISON")
    print("=" * 95)
    print(f"  Baseline ROC-AUC : {orig_auc:.4f}")
    print(f"  Shifted  ROC-AUC : {shift_auc:.4f}  (Delta: {shift_auc - orig_auc:+.4f})")
    print("-" * 95)
    header = (
        f"{'SLIDE ID':<12} {'TRUE':>4} | "
        f"{'ORIG PROB':>9} {'UNCERT':>7} {'TIER':<9} | "
        f"{'SHIFT PROB':>10} {'UNCERT':>7} {'TIER':<9} | "
        f"{'TIER CHANGE':<15}"
    )
    print(header)
    print("-" * 95)

    tier_counts_orig = {"urgent": 0, "routine": 0, "uncertain": 0}
    tier_counts_shift = {"urgent": 0, "routine": 0, "uncertain": 0}
    tier_flips = 0

    for r_orig, r_shift in zip(orig_results, shift_results):
        sid = r_orig["slide_id"]
        true_label = r_orig["true_label"]

        p_orig = r_orig["mean_prob"]
        u_orig = r_orig["std_prob"]
        t_orig = r_orig["tier"]
        tier_counts_orig[t_orig] += 1

        p_shift = r_shift["mean_prob"]
        u_shift = r_shift["std_prob"]
        t_shift = r_shift["tier"]
        tier_counts_shift[t_shift] += 1

        if t_orig != t_shift:
            tier_flips += 1
            change_str = f"{t_orig} -> {t_shift} (*)"
        else:
            change_str = "No Change"

        row = (
            f"{sid:<12} {true_label:>4} | "
            f"{p_orig:>9.4f} {u_orig:>7.4f} {t_orig:<9} | "
            f"{p_shift:>10.4f} {u_shift:>7.4f} {t_shift:<9} | "
            f"{change_str:<15}"
        )
        print(row)

    print("-" * 95)
    print("TIER DISTRIBUTION COMPARISON:")
    print(f"  Original : Urgent={tier_counts_orig['urgent']}, Routine={tier_counts_orig['routine']}, Uncertain={tier_counts_orig['uncertain']}")
    print(f"  Shifted  : Urgent={tier_counts_shift['urgent']}, Routine={tier_counts_shift['routine']}, Uncertain={tier_counts_shift['uncertain']}")
    print(f"  Total Tier Migrations / Flips: {tier_flips} of {len(orig_results)} slides")
    print("=" * 95)


def main():
    print("=" * 80)
    print("SIMULATED DOMAIN SHIFT & MULTI-SITE STAINING ROBUSTNESS TEST")
    print("=" * 80)
    print("[FRAMEWORK] Synthetic affine feature perturbation on test cohorts:")
    print("            s ~ N(1.0, 0.15) [scale], b ~ N(0.0, 0.10) [shift]")
    print("[NOTE]      Simulated proxy analysis (Not real multi-site data).")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # PART 1: REAL CAMELYON16 TEST SET (4 slides)
    # ------------------------------------------------------------------------
    print("\n>>> PART 1: Evaluating on Real CAMELYON16 Test Set (4 slides)...")
    set_seed(SEED)
    real_feat_dir = BASE_DIR / "data" / "camelyon16_features"
    train_bags, val_bags, test_bags = load_real_camelyon16_bags(str(real_feat_dir))

    # Train model on real 12-slide training set
    model_real = AttentionMIL(feature_dim=2048)
    model_real = train_model(model_real, train_bags, val_bags)

    # Create shifted test set
    shifted_test_bags_real = apply_simulated_domain_shift(
        test_bags, scale_std=0.15, shift_std=0.10, seed=42
    )

    # Evaluate both
    orig_res_real, orig_auc_real = evaluate_cohort(model_real, test_bags)
    shift_res_real, shift_auc_real = evaluate_cohort(model_real, shifted_test_bags_real)

    print_comparison_table(
        "Real CAMELYON16 Test Set (N=4)",
        orig_res_real,
        shift_res_real,
        orig_auc_real,
        shift_auc_real,
    )

    # ------------------------------------------------------------------------
    # PART 2: SYNTHETIC BENCHMARK TEST SET (50 slides)
    # ------------------------------------------------------------------------
    print("\n>>> PART 2: Evaluating on Synthetic Benchmark Test Set (50 slides)...")
    set_seed(SEED)
    train_bags_synth = generate_synthetic_bags(NUM_SLIDES_TRAIN, label_prefix="train", start_idx=0)
    val_bags_synth = generate_synthetic_bags(NUM_SLIDES_VAL, label_prefix="val", start_idx=0)
    test_bags_synth = generate_synthetic_bags(NUM_SLIDES_TEST, label_prefix="test", start_idx=0)

    model_synth = AttentionMIL(feature_dim=FEATURE_DIM)
    model_synth = train_model(model_synth, train_bags_synth, val_bags_synth)

    # Create shifted synthetic test set
    shifted_test_bags_synth = apply_simulated_domain_shift(
        test_bags_synth, scale_std=0.15, shift_std=0.10, seed=42
    )

    orig_res_synth, orig_auc_synth = evaluate_cohort(model_synth, test_bags_synth)
    shift_res_synth, shift_auc_synth = evaluate_cohort(model_synth, shifted_test_bags_synth)

    print_comparison_table(
        "Synthetic Benchmark Test Set (N=50)",
        orig_res_synth,
        shift_res_synth,
        orig_auc_synth,
        shift_auc_synth,
    )


if __name__ == "__main__":
    main()
