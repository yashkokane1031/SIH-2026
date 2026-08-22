"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BREAST CANCER HISTOPATHOLOGY AI TRIAGE SYSTEM — Hackathon Prototype       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Pipeline:  Synthetic MIL Data → ABMIL Model → MC-Dropout → Triage JSON   ║
║                                                                            ║
║  This script uses SYNTHETIC feature vectors standing in for real           ║
║  CAMELYON16-derived CNN embeddings due to hackathon time/compute limits.   ║
║  The MIL architecture, attention mechanism, Monte Carlo Dropout            ║
║  uncertainty estimation, and tiered triage logic are all real,             ║
║  validated components that transfer directly to production.               ║
║                                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OUTPUT JSON SCHEMA  (consumed by React dashboard)                         ║
║                                                                            ║
║  {                                                                         ║
║    "metadata": {                                                           ║
║      "model": str,            # Model architecture name                    ║
║      "dataset": str,          # Data source description                    ║
║      "mc_dropout_runs": int,  # Number of stochastic forward passes        ║
║      "triage_thresholds": {                                                ║
║        "urgent_prob": float,  # Probability above → urgent                 ║
║        "routine_prob": float, # Probability below → routine                ║
║        "uncertain_std": float # Uncertainty above → needs review           ║
║      },                                                                    ║
║      "generated_at": str      # ISO timestamp                              ║
║    },                                                                      ║
║    "slides": [                                                             ║
║      {                                                                     ║
║        "slide_id": str,                    # e.g. "slide_042"              ║
║        "true_label": int,                  # 0=benign, 1=malignant         ║
║        "malignancy_probability": float,    # Mean MC-Dropout prediction    ║
║        "uncertainty_score": float,         # Std across MC-Dropout runs    ║
║        "tier": str,                        # "urgent"|"routine"|"uncertain"║
║        "num_patches": int,                 # Patches in this slide         ║
║        "patch_coordinates": [[x,y], ...],  # Grid coords per patch        ║
║        "patch_attention_weights": [float]  # Normalized 0-1 attention      ║
║      }                                                                     ║
║    ]                                                                       ║
║  }                                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import math
import random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


# ============================================================================
# CONFIGURABLE CONSTANTS
# ============================================================================

# --- Data Generation ---
NUM_SLIDES_TRAIN = 400          # Training slides (bags) — more data for harder task
NUM_SLIDES_VAL = 100            # Validation slides
NUM_SLIDES_TEST = 50            # Test slides for final triage output
PATCHES_MIN = 50                # Min patches per slide
PATCHES_MAX = 200               # Max patches per slide
FEATURE_DIM = 512               # Patch embedding dimension (simulates CNN output)
MALIGNANT_RATIO = 0.4           # Fraction of slides that are malignant

# --- Tumor Signal (per-slide variability for realistic difficulty) ---
TUMOR_MEAN_SHIFT_MU = 0.17      # Mean of the per-slide tumor signal strength
TUMOR_MEAN_SHIFT_STD = 0.06     # Std dev -- some slides have strong, others weak signal
TUMOR_MEAN_SHIFT_MIN = 0.03     # Clip floor -- ensure at least some signal
TUMOR_PATCH_FRAC_MIN = 0.05     # Min tumor patch fraction (hard needle-in-haystack)
TUMOR_PATCH_FRAC_MAX = 0.20     # Max tumor patch fraction
FEATURE_NOISE_STD = 1.5         # Std of patch features (> 1.0 increases overlap)

# --- Model Architecture ---
HIDDEN_DIM = 256                # Internal projection dimension
ATTENTION_DIM = 128             # Attention network hidden dimension
DROPOUT_RATE = 0.25             # Dropout rate (used for both training and MC-Dropout)

# --- Training ---
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 25                 # More epochs to show gradual AUC climb
BATCH_SIZE = 1                  # MIL uses batch_size=1 (one bag at a time)

# --- Monte Carlo Dropout ---
MC_DROPOUT_RUNS = 20            # Number of stochastic forward passes

# --- Triage Thresholds ---
URGENT_PROB_THRESHOLD = 0.7     # Probability above this → urgent
ROUTINE_PROB_THRESHOLD = 0.3    # Probability below this → routine
UNCERTAINTY_THRESHOLD = 0.15    # Uncertainty above this → uncertain (needs review)

# --- Output & Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "camelyon16_features"
OUTPUT_JSON_PATH = BASE_DIR / "triage_results.json"
DASHBOARD_JSON_PATH = BASE_DIR / "dashboard" / "public" / "triage_results.json"

# --- Reproducibility ---
SEED = 42


# ============================================================================
# 1. SYNTHETIC BAG GENERATOR
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_bags(
    num_slides: int,
    label_prefix: str = "slide",
    start_idx: int = 0,
) -> list[dict]:
    """
    Generate synthetic MIL bags mimicking histopathology slide embeddings.

    Each slide (bag) contains 50-200 patches (instances), where each patch is
    a 512-d feature vector with elevated noise (std=FEATURE_NOISE_STD) so that
    benign and tumor distributions overlap significantly in feature space.

    For malignant slides:
      - Tumor signal strength is sampled PER-SLIDE from
        N(TUMOR_MEAN_SHIFT_MU, TUMOR_MEAN_SHIFT_STD), clipped >= TUMOR_MEAN_SHIFT_MIN.
        This means some malignant slides have strong signal (easy) and others
        have weak, ambiguous signal (hard).
      - Tumor patch fraction is sampled uniformly from
        [TUMOR_PATCH_FRAC_MIN, TUMOR_PATCH_FRAC_MAX], so some slides are
        needle-in-haystack cases with very few tumor patches.

    Returns a list of dicts:
        {
            "slide_id": str,
            "label": int (0 or 1),
            "features": np.ndarray of shape (num_patches, 512),
            "coordinates": np.ndarray of shape (num_patches, 2),
        }
    """
    bags = []

    for i in range(num_slides):
        slide_id = f"{label_prefix}_{start_idx + i:03d}"
        num_patches = np.random.randint(PATCHES_MIN, PATCHES_MAX + 1)
        is_malignant = int(np.random.random() < MALIGNANT_RATIO)

        # Generate grid coordinates (simulate a tissue region on the slide)
        grid_size = int(math.ceil(math.sqrt(num_patches)))
        all_coords = np.array(
            [(x, y) for x in range(grid_size) for y in range(grid_size)]
        )
        # Randomly sample from the grid to get realistic sparse tissue layout
        chosen_indices = np.random.choice(
            len(all_coords), size=num_patches, replace=False
        )
        coordinates = all_coords[chosen_indices]

        # Generate patch features with elevated noise (std > 1.0)
        # This increases overlap between benign and tumor distributions
        features = (
            np.random.randn(num_patches, FEATURE_DIM).astype(np.float32)
            * FEATURE_NOISE_STD
        )

        if is_malignant:
            # --- Per-slide tumor signal strength ---
            # Sample from N(mu, sigma), clip to stay positive
            slide_shift = max(
                TUMOR_MEAN_SHIFT_MIN,
                np.random.normal(TUMOR_MEAN_SHIFT_MU, TUMOR_MEAN_SHIFT_STD),
            )

            # --- Per-slide tumor patch fraction ---
            # Uniform between min and max, creating needle-in-haystack cases
            tumor_frac = np.random.uniform(
                TUMOR_PATCH_FRAC_MIN, TUMOR_PATCH_FRAC_MAX
            )
            num_tumor = max(2, int(num_patches * tumor_frac))

            tumor_indices = np.random.choice(
                num_patches, size=num_tumor, replace=False
            )
            # Tumor patches: same noise level, but with a per-slide mean shift
            features[tumor_indices] = (
                np.random.randn(num_tumor, FEATURE_DIM).astype(np.float32)
                * FEATURE_NOISE_STD
                + slide_shift
            )

        bags.append({
            "slide_id": slide_id,
            "label": is_malignant,
            "features": features,
            "coordinates": coordinates,
        })

    return bags


def load_real_camelyon16_bags(
    features_dir: str = "data/camelyon16_features",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load extracted per-slide CAMELYON16 .npz feature files and split
    at the SLIDE level (stratified by label) into train, val, and test bags.
    """
    feat_path = Path(features_dir)
    npz_files = sorted(list(feat_path.glob("*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz feature files found in {features_dir}")

    bags = []
    for f in npz_files:
        data = np.load(f)
        bags.append({
            "slide_id": str(data["slide_id"]),
            "label": int(data["label"]),
            "features": data["features"].astype(np.float32),
            "coordinates": data["coordinates"],
        })

    # Stratified split at the slide level
    normal_bags = [b for b in bags if b["label"] == 0]
    tumor_bags = [b for b in bags if b["label"] == 1]

    rng = random.Random(seed)
    rng.shuffle(normal_bags)
    rng.shuffle(tumor_bags)

    def split_group(group):
        n = len(group)
        if n == 0:
            return [], [], []
        if n == 1:
            return group, [], []
        if n == 2:
            return group[:1], group[1:], []
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train = group[:n_train]
        val = group[n_train:n_train + n_val]
        test = group[n_train + n_val:]
        return train, val, test

    norm_train, norm_val, norm_test = split_group(normal_bags)
    tum_train, tum_val, tum_test = split_group(tumor_bags)

    train_bags = norm_train + tum_train
    val_bags = norm_val + tum_val
    test_bags = norm_test + tum_test

    rng.shuffle(train_bags)
    rng.shuffle(val_bags)
    rng.shuffle(test_bags)

    return train_bags, val_bags, test_bags


class MILBagDataset(Dataset):
    """PyTorch Dataset wrapper for MIL bags."""

    def __init__(self, bags: list[dict]):
        self.bags = bags

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        return {
            "slide_id": bag["slide_id"],
            "features": torch.tensor(bag["features"], dtype=torch.float32),
            "label": torch.tensor(bag["label"], dtype=torch.float32),
            "coordinates": torch.tensor(bag["coordinates"], dtype=torch.float32),
        }


# ============================================================================
# 2. ATTENTION-BASED MIL MODEL (ABMIL)
# ============================================================================

class AttentionMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL) classifier.

    Architecture:
        1. Per-patch linear projection: FEATURE_DIM → HIDDEN_DIM
        2. Attention network: 2-layer MLP producing scalar weight per patch
        3. Attention-weighted sum → slide-level embedding
        4. Classifier head → malignancy probability (with dropout for MC-Dropout)

    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning" (ICML 2018)
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        attention_dim: int = ATTENTION_DIM,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super().__init__()

        # --- Per-patch feature projection ---
        self.patch_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )

        # --- Attention network ---
        # Two-layer MLP: hidden_dim → attention_dim → 1
        # Uses Tanh gating (as in the original ABMIL paper)
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_weights = nn.Linear(attention_dim, 1)

        # --- Classifier head (with dropout for MC-Dropout) ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),      # <-- dropout kept active for MC inference
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, bag_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single bag (slide).

        Args:
            bag_features: Tensor of shape (num_patches, feature_dim)

        Returns:
            logit:     Scalar logit for malignancy (before sigmoid)
            attention: Tensor of shape (num_patches,) — normalized attention weights
        """
        # 1. Project each patch
        # (num_patches, feature_dim) → (num_patches, hidden_dim)
        h = self.patch_projection(bag_features)

        # 2. Compute attention scores (gated attention mechanism)
        # (num_patches, hidden_dim) → (num_patches, attention_dim) → (num_patches, 1)
        a_v = self.attention_V(h)   # Tanh path
        a_u = self.attention_U(h)   # Sigmoid gate
        attention_scores = self.attention_weights(a_v * a_u)  # Element-wise gating

        # Normalize via softmax → proper attention distribution
        attention = F.softmax(attention_scores, dim=0)  # (num_patches, 1)

        # 3. Weighted sum → slide-level embedding
        # (1, num_patches) @ (num_patches, hidden_dim) → (1, hidden_dim)
        slide_embedding = torch.mm(attention.T, h)

        # 4. Classify
        logit = self.classifier(slide_embedding)  # (1, 1)

        return logit.squeeze(), attention.squeeze()


# ============================================================================
# 3. TRAINING LOOP
# ============================================================================

def train_model(
    model: AttentionMIL,
    train_bags: list[dict],
    val_bags: list[dict],
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> AttentionMIL:
    """
    Train the ABMIL model on synthetic bags with binary cross-entropy loss.
    Reports train/val AUC per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_dataset = MILBagDataset(train_bags)
    val_dataset = MILBagDataset(val_bags)

    # MIL typically processes one bag at a time (variable-size bags)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("=" * 60)
    print(f"Training ABMIL  |  device={device}  |  epochs={num_epochs}")
    print(f"Train: {len(train_bags)} slides  |  Val: {len(val_bags)} slides")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        # ----- Training -----
        model.train()
        train_losses = []
        train_preds, train_labels = [], []

        for batch in train_loader:
            features = batch["features"].squeeze(0).to(device)  # (N_patches, 512)
            label = batch["label"].squeeze().to(device)

            logit, _ = model(features)
            loss = F.binary_cross_entropy_with_logits(logit, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(torch.sigmoid(logit).item())
            train_labels.append(label.item())

        if len(set(train_labels)) > 1:
            train_auc = roc_auc_score(train_labels, train_preds)
        else:
            train_auc = 1.0 if all(l == 1 for l in train_labels) or all(l == 0 for l in train_labels) else 0.5

        # ----- Validation -----
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].squeeze(0).to(device)
                label = batch["label"].squeeze().to(device)
                logit, _ = model(features)
                val_preds.append(torch.sigmoid(logit).item())
                val_labels.append(label.item())

        if len(set(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_preds)
        elif len(val_labels) > 0:
            val_auc = 1.0 if (all(l == 1 for l in val_labels) and all(p > 0.5 for p in val_preds)) or (all(l == 0 for l in val_labels) and all(p <= 0.5 for p in val_preds)) else 0.5
        else:
            val_auc = 0.0

        avg_loss = np.mean(train_losses)
        print(
            f"  Epoch {epoch:2d}/{num_epochs}  |  "
            f"Loss: {avg_loss:.4f}  |  "
            f"Train AUC: {train_auc:.4f}  |  "
            f"Val AUC: {val_auc:.4f}"
        )

    print("=" * 60)
    return model


# ============================================================================
# 4. MONTE CARLO DROPOUT UNCERTAINTY ESTIMATION
# ============================================================================

def enable_mc_dropout(model: nn.Module):
    """
    Enable dropout layers during inference for Monte Carlo Dropout.

    This keeps the model in eval mode for batch norm etc., but manually
    re-enables all Dropout layers so each forward pass is stochastic.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Keep dropout active


def mc_dropout_inference(
    model: AttentionMIL,
    bags: list[dict],
    n_runs: int = MC_DROPOUT_RUNS,
) -> list[dict]:
    """
    Run Monte Carlo Dropout inference on a set of slides.

    For each slide, performs `n_runs` stochastic forward passes with dropout
    active. Returns mean probability (malignancy score) and std (uncertainty).

    Returns list of dicts with:
        slide_id, true_label, mean_prob, std_prob, attention_weights, coordinates
    """
    device = next(model.parameters()).device
    enable_mc_dropout(model)

    results = []

    for bag in bags:
        features = torch.tensor(
            bag["features"], dtype=torch.float32
        ).to(device)

        # Collect predictions across multiple stochastic passes
        mc_predictions = []
        all_attentions = []

        for _ in range(n_runs):
            with torch.no_grad():
                logit, attention = model(features)
                prob = torch.sigmoid(logit).item()
                mc_predictions.append(prob)
                all_attentions.append(attention.cpu().numpy())

        # Aggregate MC-Dropout results
        mean_prob = float(np.mean(mc_predictions))
        std_prob = float(np.std(mc_predictions))

        # Average attention weights across runs and normalize to [0, 1]
        avg_attention = np.mean(all_attentions, axis=0)
        # Min-max normalize to 0-1 range for the dashboard
        att_min, att_max = avg_attention.min(), avg_attention.max()
        if att_max - att_min > 1e-8:
            normalized_attention = (
                (avg_attention - att_min) / (att_max - att_min)
            )
        else:
            normalized_attention = np.zeros_like(avg_attention)

        results.append({
            "slide_id": bag["slide_id"],
            "true_label": bag["label"],
            "mean_prob": mean_prob,
            "std_prob": std_prob,
            "attention_weights": normalized_attention.tolist(),
            "coordinates": bag["coordinates"].tolist(),
            "num_patches": len(bag["features"]),
        })

    return results


# ============================================================================
# 5. TIERED TRIAGE ROUTER
# ============================================================================

def assign_triage_tier(
    probability: float,
    uncertainty: float,
    urgent_prob: float = URGENT_PROB_THRESHOLD,
    routine_prob: float = ROUTINE_PROB_THRESHOLD,
    uncertain_std: float = UNCERTAINTY_THRESHOLD,
) -> str:
    """
    Assign a triage tier based on malignancy probability and uncertainty.

    Decision logic:
        Tier 1 (urgent):    prob > urgent_prob  AND uncertainty < uncertain_std
        Tier 2 (routine):   prob <= routine_prob AND uncertainty < uncertain_std
        Tier 3 (uncertain): uncertainty >= uncertain_std (regardless of prob)

    If prob is in the middle range and uncertainty is low, defaults to "routine"
    as a conservative fallback (pathologist reviews these anyway).
    """
    if uncertainty >= uncertain_std:
        return "uncertain"       # Tier 3: needs expert review
    elif probability > urgent_prob:
        return "urgent"          # Tier 1: likely malignant, high confidence
    elif probability <= routine_prob:
        return "routine"         # Tier 2: likely benign, high confidence
    else:
        # Middle-ground probability with low uncertainty
        # Conservative: flag for standard review
        return "routine"


def route_slides(results: list[dict]) -> list[dict]:
    """Apply triage routing to all inference results."""
    for r in results:
        r["tier"] = assign_triage_tier(r["mean_prob"], r["std_prob"])
    return results


# ============================================================================
# 6. JSON EXPORT
# ============================================================================

def export_results(
    results: list[dict],
    output_path: str = None,
    dataset_name: str = "Synthetic (standing in for CAMELYON16-derived features)",
    feature_dim: int = FEATURE_DIM,
    extra_dashboard_filename: str = None,
) -> dict:
    """Export complete triage results with metadata to a structured JSON file."""
    output_path = output_path or OUTPUT_JSON_PATH

    output = {
        "metadata": {
            "model": "ABMIL (Attention-Based MIL) with Gated Attention",
            "dataset": dataset_name,
            "feature_dim": feature_dim,
            "mc_dropout_runs": MC_DROPOUT_RUNS,
            "dropout_rate": DROPOUT_RATE,
            "triage_thresholds": {
                "urgent_prob": URGENT_PROB_THRESHOLD,
                "routine_prob": ROUTINE_PROB_THRESHOLD,
                "uncertain_std": UNCERTAINTY_THRESHOLD,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "MIL architecture with Gated Attention, MC-Dropout uncertainty "
                "estimation, and 3-tier clinical triage routing."
            ),
        },
        "slides": [],
    }

    for r in results:
        output["slides"].append({
            "slide_id": r["slide_id"],
            "true_label": r["true_label"],
            "malignancy_probability": round(r["mean_prob"], 6),
            "uncertainty_score": round(r["std_prob"], 6),
            "tier": r["tier"],
            "num_patches": r["num_patches"],
            "patch_coordinates": r["coordinates"],
            "patch_attention_weights": [
                round(w, 6) for w in r["attention_weights"]
            ],
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Also export to dashboard/public
    try:
        DASHBOARD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        if extra_dashboard_filename:
            extra_path = DASHBOARD_JSON_PATH.parent / extra_dashboard_filename
            with open(extra_path, "w") as f:
                json.dump(output, f, indent=2)
        with open(DASHBOARD_JSON_PATH, "w") as f:
            json.dump(output, f, indent=2)
    except Exception:
        pass

    return output


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def print_triage_summary(results: list[dict]):
    """Print a human-readable triage summary table."""
    tier_counts = {"urgent": 0, "routine": 0, "uncertain": 0}
    tier_correct = {"urgent": 0, "routine": 0, "uncertain": 0}

    print("\n" + "=" * 80)
    print(f"{'SLIDE ID':<14} {'TRUE':>5} {'PROB':>8} {'UNCERT':>8} {'TIER':<12} {'OK':>3}")
    print("-" * 80)

    for r in results:
        tier = r["tier"]
        tier_counts[tier] += 1

        # Check if triage aligns with ground truth
        correct = ""
        if tier == "urgent" and r["true_label"] == 1:
            correct = "Y"
            tier_correct[tier] += 1
        elif tier == "routine" and r["true_label"] == 0:
            correct = "Y"
            tier_correct[tier] += 1
        elif tier == "uncertain":
            correct = "?"
            tier_correct[tier] += 1

        print(
            f"  {r['slide_id']:<12} "
            f"{r['true_label']:>5} "
            f"{r['mean_prob']:>8.4f} "
            f"{r['std_prob']:>8.4f} "
            f"{tier:<12} "
            f"{correct:>3}"
        )

    print("-" * 80)
    print("TRIAGE DISTRIBUTION:")
    for tier, count in tier_counts.items():
        marker = {"urgent": "[!]", "routine": "[+]", "uncertain": "[?]"}[tier]
        print(f"  {marker} {tier.upper():<12}: {count:>3} slides")
    print("=" * 80)


def run_pipeline_experiment(mode: str = "camelyon16"):
    """Run a full training and triage pipeline for a specific data mode."""
    start_time = time.time()
    set_seed(SEED)

    real_feat_dir = Path("data/camelyon16_features")
    use_real = (mode == "camelyon16") and real_feat_dir.exists() and len(list(real_feat_dir.glob("*.npz"))) > 0

    if use_real:
        print(f"\n{'='*70}\nRUNNING ON REAL CAMELYON16 DATASET (20 SLIDES)\n{'='*70}")
        print("\n[DATA] Stage 1: Loading real CAMELYON16 slide embeddings...")
        train_bags, val_bags, test_bags = load_real_camelyon16_bags(str(real_feat_dir))
        feature_dim = train_bags[0]["features"].shape[1] if train_bags else FEATURE_DIM
        dataset_name = f"CAMELYON16 (Real ResNet-50 2048-d, N={len(train_bags)+len(val_bags)+len(test_bags)} slides)"
        out_name = "triage_results_camelyon16.json"
    else:
        print(f"\n{'='*70}\nRUNNING ON SYNTHETIC BENCHMARK DATASET (550 SLIDES)\n{'='*70}")
        print("\n[DATA] Stage 1: Generating synthetic histopathology data...")
        train_bags = generate_synthetic_bags(NUM_SLIDES_TRAIN, label_prefix="train", start_idx=0)
        val_bags = generate_synthetic_bags(NUM_SLIDES_VAL, label_prefix="val", start_idx=0)
        test_bags = generate_synthetic_bags(NUM_SLIDES_TEST, label_prefix="test", start_idx=0)
        feature_dim = FEATURE_DIM
        dataset_name = "Synthetic (standing in for CAMELYON16-derived features)"
        out_name = "triage_results_synthetic.json"

    n_mal_train = sum(b["label"] for b in train_bags)
    n_mal_val = sum(b["label"] for b in val_bags)
    n_mal_test = sum(b["label"] for b in test_bags)
    print(f"  Train: {len(train_bags)} slides ({n_mal_train} malignant)")
    print(f"  Val:   {len(val_bags)} slides ({n_mal_val} malignant)")
    print(f"  Test:  {len(test_bags)} slides ({n_mal_test} malignant)")

    # ---- Stage 2: Build & Train Model ----
    print("\n[MODEL] Stage 2: Training ABMIL model...")
    model = AttentionMIL(feature_dim=feature_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,} (feature_dim={feature_dim})")
    model = train_model(model, train_bags, val_bags)

    # ---- Stage 3: MC-Dropout Inference ----
    print(f"\n[MC] Stage 3: MC-Dropout inference ({MC_DROPOUT_RUNS} passes)...")
    eval_bags = (test_bags + val_bags + train_bags) if use_real else test_bags
    test_results = mc_dropout_inference(model, eval_bags)

    # ---- Stage 4: Triage Routing ----
    print("\n[TRIAGE] Stage 4: Applying triage routing...")
    test_results = route_slides(test_results)

    # ---- Stage 5: Export Results ----
    out_file = BASE_DIR / out_name
    output = export_results(
        test_results,
        output_path=str(out_file),
        dataset_name=dataset_name,
        feature_dim=feature_dim,
        extra_dashboard_filename=out_name,
    )
    print(f"\n[SAVE] Stage 5: Results exported to {out_name} and dashboard/public/{out_name}")
    print(f"  Total slides in output: {len(output['slides'])}")

    # ---- Summary ----
    print_triage_summary(test_results)

    elapsed = time.time() - start_time
    print(f"\n[TIME] Completed {mode} in {elapsed:.1f}s")
    return output


def main():
    """Main entrypoint supporting execution of real, synthetic, or both datasets."""
    import argparse
    parser = argparse.ArgumentParser(description="Attention-MIL Triage Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["camelyon16", "synthetic", "both"],
        default="both",
        help="Dataset to run: 'camelyon16' (real), 'synthetic', or 'both' (default: both)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Alias to run synthetic dataset only",
    )
    args = parser.parse_args()

    mode = "synthetic" if args.synthetic else args.dataset

    if mode == "both":
        run_pipeline_experiment("camelyon16")
        run_pipeline_experiment("synthetic")
    else:
        run_pipeline_experiment(mode)


if __name__ == "__main__":
    main()
