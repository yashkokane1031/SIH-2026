import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
)

set_seed(SEED)
train_bags, val_bags, test_bags = load_real_camelyon16_bags("data/camelyon16_features")

model = AttentionMIL(feature_dim=2048)
model = train_model(model, train_bags, val_bags)

def eval_split(split_name, bags):
    results = mc_dropout_inference(model, bags)
    results = route_slides(results)
    y_true = [r["true_label"] for r in results]
    y_prob = [r["mean_prob"] for r in results]
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return results, auc

train_res, train_auc = eval_split("Train", train_bags)
val_res, val_auc = eval_split("Val", val_bags)
test_res, test_auc = eval_split("Test", test_bags)

print("=" * 80)
print("OVERALL METRICS SUMMARY:")
print(f"  Train AUC (12 slides): {train_auc:.4f}")
print(f"  Val   AUC (4 slides) : {val_auc:.4f}")
print(f"  Test  AUC (4 slides) : {test_auc:.4f}")
print("=" * 80)

def print_table(title, results):
    print(f"\n{title} SET (N={len(results)}):")
    print("=" * 80)
    print(f"{'SLIDE ID':<16} {'TRUE':>4} {'PROB':>8} {'UNCERT':>8} {'TIER':<10} {'OK':>4}")
    print("-" * 80)
    tier_counts = {"urgent": 0, "routine": 0, "uncertain": 0}
    for r in results:
        t = r["true_label"]
        p = r["mean_prob"]
        u = r["std_prob"]
        tier = r["tier"]
        tier_counts[tier] += 1
        ok = "Y" if (t == 1 and tier == "urgent") or (t == 0 and tier == "routine") else ""
        sid = r["slide_id"]
        print(f"{sid:<16} {t:>4} {p:>8.4f} {u:>8.4f} {tier:<10} {ok:>4}")
    print("-" * 80)
    print(f"Triage Tier Distribution: Urgent={tier_counts['urgent']}, Routine={tier_counts['routine']}, Uncertain={tier_counts['uncertain']}")
    print("=" * 80)

print_table("VALIDATION", val_res)
print_table("TEST", test_res)
print_table("TRAINING", train_res)
