import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from pipeline import (
    AttentionMIL,
    train_model,
    mc_dropout_inference,
    route_slides,
    load_real_camelyon16_bags,
    SEED,
    set_seed,
    URGENT_PROB_THRESHOLD,
    ROUTINE_PROB_THRESHOLD,
    UNCERTAINTY_THRESHOLD,
)

# Fix seed for deterministic evaluation
set_seed(SEED)

# Load real CAMELYON16 bags
feat_dir = BASE_DIR / "data" / "camelyon16_features"
train_bags, val_bags, test_bags = load_real_camelyon16_bags(str(feat_dir))

# Initialize model
feature_dim = train_bags[0]["features"].shape[1]
model = AttentionMIL(feature_dim=feature_dim)

# Train model using standard pipeline procedure
model = train_model(model, train_bags, val_bags)

# Evaluate splits
def evaluate_split(split_name, bags):
    results = mc_dropout_inference(model, bags)
    results = route_slides(results)
    y_true = [r["true_label"] for r in results]
    y_prob = [r["mean_prob"] for r in results]
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return results, auc

train_res, train_auc = evaluate_split("Train", train_bags)
val_res, val_auc = evaluate_split("Val", val_bags)
test_res, test_auc = evaluate_split("Test", test_bags)

all_res = train_res + val_res + test_res
all_y_true = [r["true_label"] for r in all_res]
all_y_prob = [r["mean_prob"] for r in all_res]
combined_auc = roc_auc_score(all_y_true, all_y_prob)

print("\n" + "=" * 90)
print("REAL CAMELYON16 CLINICAL SAFETY & SENSITIVITY AUDIT")
print("=" * 90)
print(f"Urgent Probability Threshold   : >= {URGENT_PROB_THRESHOLD:.2f}")
print(f"Routine Probability Threshold  : <= {ROUTINE_PROB_THRESHOLD:.2f}")
print(f"Uncertainty Standard Dev Thresh: >= {UNCERTAINTY_THRESHOLD:.2f}")
print("=" * 90)

def analyze_cohort(name, results, auc_val):
    malignant_slides = [r for r in results if r["true_label"] == 1]
    benign_slides = [r for r in results if r["true_label"] == 0]
    
    total_mal = len(malignant_slides)
    total_ben = len(benign_slides)
    total_slides = len(results)
    
    # Triage tiers for malignant
    mal_urgent = [r for r in malignant_slides if r["tier"] == "urgent"]
    mal_uncertain = [r for r in malignant_slides if r["tier"] == "uncertain"]
    mal_routine = [r for r in malignant_slides if r["tier"] == "routine"]
    
    # Flagged as non-routine (safe / caught)
    mal_caught = len(mal_urgent) + len(mal_uncertain)
    mal_missed = len(mal_routine)
    
    # Standard classification sensitivity (at prob >= 0.5 threshold)
    tp_at_half = sum(1 for r in malignant_slides if r["mean_prob"] >= 0.5)
    
    # Triage Sensitivity (Urgent or Uncertain vs Routine)
    triage_sensitivity = (mal_caught / total_mal * 100) if total_mal > 0 else 0.0
    
    # Standard Sensitivity (Prob >= 0.5)
    binary_sensitivity_05 = (tp_at_half / total_mal * 100) if total_mal > 0 else 0.0
    
    # Urgent Sensitivity (Tier 1 only)
    urgent_sensitivity = (len(mal_urgent) / total_mal * 100) if total_mal > 0 else 0.0

    print(f"\n--- {name.upper()} COHORT (N={total_slides}, Malignant={total_mal}, Benign={total_ben}) ---")
    print(f"ROC-AUC: {auc_val:.4f}")
    print(f"Malignant Cases Breakdown:")
    print(f"  - Routed to Tier 1 (Urgent)   : {len(mal_urgent)}/{total_mal}")
    print(f"  - Routed to Tier 3 (Uncertain): {len(mal_uncertain)}/{total_mal}")
    print(f"  - Routed to Tier 2 (Routine)  : {len(mal_routine)}/{total_mal} [SILENT MISSES]")
    print(f"\nSafety Fraction (Urgent + Uncertain) / Total Malignant: {mal_caught}/{total_mal} ({triage_sensitivity:.2f}%)")
    print(f"Silent Miss Rate (Routine / Total Malignant)          : {mal_missed}/{total_mal} ({(mal_missed/total_mal*100) if total_mal>0 else 0:.2f}%)")
    print(f"Binary Sensitivity at 0.5 Classification Cutoff        : {tp_at_half}/{total_mal} ({binary_sensitivity_05:.2f}%)")
    print(f"High-Confidence Urgent Sensitivity (Tier 1 Only)       : {len(mal_urgent)}/{total_mal} ({urgent_sensitivity:.2f}%)")
    
    print("\nDetailed Slide Breakdown:")
    print(f"{'SLIDE ID':<16} {'TRUE':>4} {'PROB':>10} {'UNCERT':>10} {'TIER':<10} {'STATUS':<20}")
    print("-" * 75)
    for r in results:
        t = r["true_label"]
        p = r["mean_prob"]
        u = r["std_prob"]
        tier = r["tier"]
        if t == 1:
            status = "CAUGHT (Urgent)" if tier == "urgent" else "CAUGHT (Flagged)" if tier == "uncertain" else "SILENT MISS"
        else:
            status = "Correct Benign" if tier == "routine" else "False Urgent" if tier == "urgent" else "False Flagged"
        print(f"{r['slide_id']:<16} {t:>4} {p:>10.6f} {u:>10.6f} {tier:<10} {status:<20}")
        
    return malignant_slides, mal_routine

print("\n" + "#" * 90)
print("1. TEST SET ANALYSIS")
print("#" * 90)
test_mal, test_misses = analyze_cohort("Held-Out Test Set", test_res, test_auc)

print("\n" + "#" * 90)
print("2. VALIDATION SET ANALYSIS")
print("#" * 90)
val_mal, val_misses = analyze_cohort("Validation Set", val_res, val_auc)

print("\n" + "#" * 90)
print("3. TRAINING SET ANALYSIS")
print("#" * 90)
train_mal, train_misses = analyze_cohort("Training Set", train_res, train_auc)

print("\n" + "#" * 90)
print("4. COMBINED REAL DATASET (TRAIN + VAL + TEST, N=20)")
print("#" * 90)
all_mal, all_misses = analyze_cohort("Combined Real CAMELYON16", all_res, combined_auc)

print("\n" + "#" * 90)
print("5. DETAILED AUDIT OF SILENT MISSES (MALIGNANT SLIDES IN ROUTINE TIER)")
print("#" * 90)

all_malignant_in_routine = [r for r in all_res if r["true_label"] == 1 and r["tier"] == "routine"]

if not all_malignant_in_routine:
    print("No malignant slides were routed to Routine tier across the dataset.")
else:
    print(f"Total Malignant Slides in Routine across all 20 slides: {len(all_malignant_in_routine)}")
    print("-" * 90)
    for r in all_malignant_in_routine:
        p = r["mean_prob"]
        u = r["std_prob"]
        dist_below_urgent = URGENT_PROB_THRESHOLD - p
        dist_below_uncertain = UNCERTAINTY_THRESHOLD - u
        
        print(f"Slide ID: {r['slide_id']}")
        print(f"  - Ground Truth Label    : Malignant (true_label = 1)")
        print(f"  - Assigned Triage Tier  : {r['tier'].upper()}")
        print(f"  - Exact Mean Probability: {p:.6f}")
        print(f"  - Exact Uncertainty (SD): {u:.6f}")
        print(f"  - Distance below Urgent Threshold ({URGENT_PROB_THRESHOLD:.2f})   : {dist_below_urgent:+.6f} (prob is {dist_below_urgent:.6f} below {URGENT_PROB_THRESHOLD:.2f})")
        print(f"  - Distance below Uncertain Threshold ({UNCERTAINTY_THRESHOLD:.2f}): {dist_below_uncertain:+.6f} (SD is {dist_below_uncertain:.6f} below {UNCERTAINTY_THRESHOLD:.2f})")
        print("-" * 90)
