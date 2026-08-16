"""
Report script: saves per-epoch AUC log and runs triage on 200 test slides.
Reuses all components from pipeline.py.
"""

import json
import csv
import time
from datetime import datetime, timezone

from pipeline import (
    set_seed, generate_synthetic_bags, AttentionMIL, train_model,
    mc_dropout_inference, route_slides, export_results,
    SEED, NUM_SLIDES_TRAIN, NUM_SLIDES_VAL, MC_DROPOUT_RUNS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    TUMOR_MEAN_SHIFT_MU, TUMOR_MEAN_SHIFT_STD, FEATURE_NOISE_STD,
    TUMOR_PATCH_FRAC_MIN, TUMOR_PATCH_FRAC_MAX,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from pipeline import MILBagDataset


# ============================================================================
# CONFIG
# ============================================================================
LARGE_TEST_SIZE = 200
AUC_LOG_PATH = "training_auc_log.csv"
LARGE_TRIAGE_JSON = "triage_results_200.json"
REPORT_PATH = "run_report.txt"


def train_with_logging(model, train_bags, val_bags):
    """Train and return per-epoch AUC log as a list of dicts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_loader = DataLoader(MILBagDataset(train_bags), batch_size=1, shuffle=True)
    val_loader = DataLoader(MILBagDataset(val_bags), batch_size=1, shuffle=False)

    epoch_log = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_losses = []
        train_preds, train_labels = [], []

        for batch in train_loader:
            features = batch["features"].squeeze(0).to(device)
            label = batch["label"].squeeze().to(device)
            logit, _ = model(features)
            loss = F.binary_cross_entropy_with_logits(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.append(torch.sigmoid(logit).item())
            train_labels.append(label.item())

        train_auc = roc_auc_score(train_labels, train_preds)

        # --- Val ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].squeeze(0).to(device)
                label = batch["label"].squeeze().to(device)
                logit, _ = model(features)
                val_preds.append(torch.sigmoid(logit).item())
                val_labels.append(label.item())

        val_auc = roc_auc_score(val_labels, val_preds)
        avg_loss = float(np.mean(train_losses))

        row = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "train_auc": round(train_auc, 6),
            "val_auc": round(val_auc, 6),
        }
        epoch_log.append(row)
        print(
            f"  Epoch {epoch:2d}/{NUM_EPOCHS}  |  "
            f"Loss: {avg_loss:.6f}  |  "
            f"Train AUC: {train_auc:.4f}  |  "
            f"Val AUC: {val_auc:.4f}"
        )

    return model, epoch_log


def main():
    start = time.time()
    set_seed(SEED)

    # --- Generate data ---
    print("[1/4] Generating data...")
    train_bags = generate_synthetic_bags(NUM_SLIDES_TRAIN, "train", 0)
    val_bags = generate_synthetic_bags(NUM_SLIDES_VAL, "val", 0)
    test_bags = generate_synthetic_bags(LARGE_TEST_SIZE, "test", 0)

    n_mal_test = sum(b["label"] for b in test_bags)
    print(f"  Train: {len(train_bags)}  |  Val: {len(val_bags)}  |  Test: {len(test_bags)} ({n_mal_test} malignant)")

    # --- Train with full logging ---
    print("\n[2/4] Training ABMIL (logging per-epoch AUC)...")
    model = AttentionMIL()
    model, epoch_log = train_with_logging(model, train_bags, val_bags)

    # Save AUC log to CSV
    with open(AUC_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_auc", "val_auc"])
        writer.writeheader()
        writer.writerows(epoch_log)
    print(f"\n  AUC log saved to: {AUC_LOG_PATH}")

    # --- MC-Dropout on 200 test slides ---
    print(f"\n[3/4] MC-Dropout inference on {LARGE_TEST_SIZE} test slides ({MC_DROPOUT_RUNS} passes each)...")
    results = mc_dropout_inference(model, test_bags)
    results = route_slides(results)

    # --- Tier distribution ---
    tier_counts = {"urgent": 0, "routine": 0, "uncertain": 0}
    for r in results:
        tier_counts[r["tier"]] += 1

    total = len(results)
    tier_pct = {k: round(100 * v / total, 1) for k, v in tier_counts.items()}

    # --- Export large JSON ---
    export_results(results, LARGE_TRIAGE_JSON)
    print(f"  Triage JSON saved to: {LARGE_TRIAGE_JSON}")

    # --- Build report ---
    elapsed = time.time() - start
    report_lines = [
        "=" * 60,
        "BREAST CANCER TRIAGE - FULL RUN REPORT",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 60,
        "",
        "--- PER-EPOCH TRAINING LOG ---",
        f"{'Epoch':>6}  {'Loss':>10}  {'Train AUC':>10}  {'Val AUC':>10}",
        "-" * 44,
    ]
    for row in epoch_log:
        report_lines.append(
            f"{row['epoch']:>6}  {row['train_loss']:>10.6f}  {row['train_auc']:>10.4f}  {row['val_auc']:>10.4f}"
        )

    report_lines += [
        "",
        f"--- TRIAGE DISTRIBUTION (N={total} test slides) ---",
        "",
        f"  Malignant slides in test set: {n_mal_test}/{total} ({round(100*n_mal_test/total, 1)}%)",
        "",
        f"  URGENT    : {tier_counts['urgent']:>4} slides  ({tier_pct['urgent']}%)",
        f"  ROUTINE   : {tier_counts['routine']:>4} slides  ({tier_pct['routine']}%)",
        f"  UNCERTAIN : {tier_counts['uncertain']:>4} slides  ({tier_pct['uncertain']}%)",
        "",
        f"--- TIMING ---",
        f"  Total runtime: {elapsed:.1f}s",
        f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}",
        "",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\nFiles saved:")
    print(f"  {AUC_LOG_PATH}")
    print(f"  {LARGE_TRIAGE_JSON}")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()
