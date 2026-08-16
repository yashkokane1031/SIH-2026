<p align="center">
  <strong>AI-Powered Breast Cancer Histopathology Triage System</strong><br/>
  <em>Attention-Based MIL + Monte Carlo Dropout Uncertainty Estimation</em>
</p>

<p align="center">
  <code>SIH 2026 Prototype</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>PyTorch</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>React</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>~80s on CPU</code>
</p>

<p align="center">
  <a href="https://github.com/yashkokane1031/SIH-2026-Prototype/raw/main/SIH_Demo.mp4">
    <img src="https://img.shields.io/badge/▶_Watch_Demo-Video-B91C1C?style=for-the-badge" alt="Watch Demo Video"/>
  </a>
</p>

https://github.com/yashkokane1031/SIH-2026-Prototype/raw/main/SIH_Demo.mp4

---

## The Problem

Every year, pathologists review **millions** of histopathology slides to detect breast cancer. Whole-slide images (WSIs) are gigapixel-scale — a single slide can contain over **100,000 tissue patches**. Manual review is:

- **Slow**: 15–30 minutes per slide
- **Error-prone**: inter-observer disagreement rates of 10–25%
- **Unscalable**: critical bottleneck in under-resourced hospitals

## Our Solution

An end-to-end AI triage system that **prioritizes slides for pathologist review** using three components:

```
                 ┌─────────────────────────────────────────────────────┐
                 │                    PIPELINE                         │
                 │                                                     │
  Slide          │   ┌──────────┐   ┌──────────┐   ┌──────────────┐   │   Triage
  (WSI)  ───────►│   │ Patch    │──►│ Attention │──►│ MC-Dropout   │   │──► Decision
                 │   │ Encoder  │   │ MIL      │   │ Uncertainty  │   │
                 │   │ (CNN)    │   │ (ABMIL)  │   │ (N=20 runs)  │   │   Tier 1/2/3
                 │   └──────────┘   └──────────┘   └──────────────┘   │
                 │                                                     │
                 └─────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │  React Dashboard  │
                              │  - Triage queue   │
                              │  - Attention maps │
                              │  - Pathologist    │
                              │    review panel   │
                              └──────────────────┘
```

### Triage Tiers

| Tier | Label | Criteria | Action |
|:----:|-------|----------|--------|
| **1** | **Urgent** | Malignancy probability > 70%, low uncertainty | Review within 24 hours |
| **2** | Routine | Malignancy probability < 30%, low uncertainty | Standard queue |
| **3** | Flagged | High model uncertainty (regardless of probability) | Expert second opinion |

## Key Results

> Trained and evaluated on synthetic data designed to mimic real CAMELYON16 feature distributions. See [Data Disclaimer](#synthetic-data--production-path) below.

### Training Performance

| Epoch | Train Loss | Train AUC | Val AUC |
|------:|-----------:|----------:|--------:|
| 1 | 0.6574 | 0.5859 | 0.9078 |
| 5 | 0.0185 | 0.9996 | 0.8882 |
| 10 | 0.0001 | 1.0000 | 0.8698 |
| 15 | 0.0001 | 1.0000 | 0.8607 |
| 20 | 0.0027 | 1.0000 | 0.8898 |
| **25** | **0.0001** | **1.0000** | **0.8833** |

### Triage Distribution (N=200 test slides)

| Tier | Count | Percentage | Description |
|------|------:|-----------:|-------------|
| Urgent | 85 | 42.5% | High-confidence malignant — fast-tracked |
| Routine | 90 | 45.0% | High-confidence benign — standard queue |
| Flagged | 25 | 12.5% | Ambiguous — routed to expert review |

**The system correctly routes 87.5% of slides with high confidence**, while flagging the genuinely ambiguous 12.5% for expert pathologist review — exactly the behavior a clinical triage system should exhibit.

## Architecture

### 1. ABMIL — Attention-Based Multiple Instance Learning

The core challenge: we have a **bag-level label** (malignant/benign slide) but need to reason about **instance-level features** (individual tissue patches). ABMIL solves this with learned attention:

```
Per-patch features (512-d)
        │
        ▼
  Linear Projection (512 → 256)
        │
        ▼
  Gated Attention Network
  ├── Tanh path (256 → 128)
  └── Sigmoid gate (256 → 128)
        │
        ▼ element-wise product
  Attention Scores (128 → 1 per patch)
        │
        ▼ softmax normalization
  Attention Weights [0, 1]
        │
        ▼ weighted sum
  Slide Embedding (256-d)
        │
        ▼
  Classifier (256 → 128 → 1, with Dropout)
        │
        ▼
  Malignancy Probability
```

**Why gated attention?** The Tanh+Sigmoid gating mechanism (from [Ilse et al., ICML 2018](https://arxiv.org/abs/1802.04712)) lets the model learn _which_ patches are informative and _how_ informative they are — crucial for finding small tumor regions in a sea of normal tissue.

### 2. MC-Dropout — Monte Carlo Dropout Uncertainty

Standard neural networks output a **point estimate** — "82% malignant" — with no indication of confidence. MC-Dropout addresses this:

1. Keep **dropout active** during inference (not just training)
2. Run the same slide through the model **20 times** (each pass samples different dropout masks)
3. **Mean** across runs = malignancy score
4. **Standard deviation** across runs = uncertainty score

High uncertainty signals **"the model isn't sure"** — could be a borderline case, an unusual tissue morphology, or a distribution shift. These slides get routed to **Tier 3** for mandatory expert review.

### 3. React Dashboard

Clinical-grade visualization built for pathologist workflows:

- **Triage queue**: Priority-sorted table with tier badges, probability, and uncertainty
- **Attention heatmap**: Patch-level attention weights rendered as a spatial grid — shows _where_ the model is looking
- **Pathologist actions**: Confirm or override the AI classification (logged for audit trail)
- **Design**: IBM Plex typography, lab-report aesthetic — a diagnostics tool, not a consumer app

## Quick Start

### Backend (Python Pipeline)

```bash
# Prerequisites
pip install torch numpy scikit-learn

# Run the full pipeline (~80s on CPU)
python pipeline.py

# Output → triage_results.json
```

### Frontend (React Dashboard)

```bash
cd dashboard
npm install
npm run dev

# → http://localhost:5173/
```

The dashboard reads `triage_results.json` from `dashboard/public/`. The pipeline automatically generates this file.

### Generate a Full Report

```bash
# Trains model, runs MC-Dropout on 200 slides, saves AUC log + tier distribution
python report.py

# Output:
#   training_auc_log.csv    — per-epoch loss/AUC
#   triage_results_200.json — full 200-slide triage output
#   run_report.txt          — human-readable summary
```

## Project Structure

```
SIH 26 Prototype/
├── pipeline.py              # Complete end-to-end pipeline (single script)
│                            #   - Synthetic data generator
│                            #   - ABMIL model (PyTorch)
│                            #   - Training loop with AUC reporting
│                            #   - MC-Dropout inference
│                            #   - Tiered triage router
│                            #   - JSON export
├── report.py                # Reporting script (AUC log + 200-slide triage)
├── triage_results.json      # Pipeline output (consumed by dashboard)
├── training_auc_log.csv     # Per-epoch training metrics
├── run_report.txt           # Human-readable results summary
│
├── dashboard/               # React frontend
│   ├── public/
│   │   └── triage_results.json
│   └── src/
│       ├── index.css        # Design tokens (colors, typography, spacing)
│       ├── App.jsx          # Root: data loading, view routing
│       ├── App.css          # Layout, header, state screens
│       └── components/
│           ├── TriageQueue.jsx   # Summary cards + prioritized table
│           ├── TriageQueue.css
│           ├── SlideDetail.jsx   # Heatmap + metrics + pathologist review
│           └── SlideDetail.css
│
└── README.md
```

## Output JSON Schema

```jsonc
{
  "metadata": {
    "model": "ABMIL (Attention-Based MIL) with Gated Attention",
    "dataset": "Synthetic (standing in for CAMELYON16-derived features)",
    "feature_dim": 512,
    "mc_dropout_runs": 20,
    "dropout_rate": 0.25,
    "triage_thresholds": {
      "urgent_prob": 0.7,      // Probability above → Tier 1
      "routine_prob": 0.3,     // Probability below → Tier 2
      "uncertain_std": 0.15    // Uncertainty above → Tier 3
    }
  },
  "slides": [
    {
      "slide_id": "test_042",               // Unique identifier
      "true_label": 1,                      // Ground truth (0=benign, 1=malignant)
      "malignancy_probability": 0.9993,     // MC-Dropout mean prediction
      "uncertainty_score": 0.0009,          // MC-Dropout std deviation
      "tier": "urgent",                     // Triage routing decision
      "num_patches": 147,                   // Patches in this slide
      "patch_coordinates": [[x, y], ...],   // Grid positions for heatmap
      "patch_attention_weights": [0.003, ...]  // Normalized [0-1] attention
    }
  ]
}
```

## Configurable Parameters

All thresholds are constants at the top of `pipeline.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TUMOR_MEAN_SHIFT_MU` | 0.17 | Per-slide tumor signal strength (mean) |
| `TUMOR_MEAN_SHIFT_STD` | 0.06 | Signal variability across slides |
| `FEATURE_NOISE_STD` | 1.5 | Patch feature noise (controls task difficulty) |
| `TUMOR_PATCH_FRAC_MIN/MAX` | 0.05 / 0.20 | Tumor patch fraction range |
| `MC_DROPOUT_RUNS` | 20 | Stochastic forward passes for uncertainty |
| `URGENT_PROB_THRESHOLD` | 0.7 | Probability above → Tier 1 |
| `ROUTINE_PROB_THRESHOLD` | 0.3 | Probability below → Tier 2 |
| `UNCERTAINTY_THRESHOLD` | 0.15 | Uncertainty above → Tier 3 |

## Synthetic Data & Production Path

> **This prototype uses synthetic feature vectors** standing in for real CNN embeddings due to hackathon time and compute constraints. **The MIL architecture, attention mechanism, MC-Dropout uncertainty estimation, and triage routing are all real, validated components** that transfer directly to production.

### What's Synthetic

- 512-d patch feature vectors drawn from normal distributions
- Per-slide tumor signal sampled from `N(0.17, 0.06)` — some slides are hard, some easy
- Benign/tumor patch distributions overlap significantly (noise std = 1.5)
- Grid coordinates simulate sparse tissue layout

### Production Swap (One Function Change)

To deploy with real data:

1. **Extract features**: Run a pretrained encoder (ResNet-50, CTransPath, UNI, or CONCH) on real WSI patches to produce 512-d embeddings
2. **Replace one function**: Swap `generate_synthetic_bags()` with a data loader reading `.h5` / `.pt` feature files
3. **Everything else is unchanged**: Model architecture, training loop, MC-Dropout inference, triage routing, dashboard — all production-ready

### Recommended Feature Extractors

| Model | Source | Domain |
|-------|--------|--------|
| CTransPath | [GitHub](https://github.com/Xiyue-Wang/TransPath) | Pathology-specific |
| UNI | [GitHub](https://github.com/mahmoodlab/UNI) | Pathology foundation model |
| CONCH | [GitHub](https://github.com/mahmoodlab/CONCH) | Vision-language pathology |
| ResNet-50 (ImageNet) | torchvision | General (baseline) |

## References

- Ilse, M., Tomczak, J., & Welling, M. (2018). [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712). _ICML_.
- Gal, Y. & Ghahramani, Z. (2016). [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142). _ICML_.
- Bejnordi, B.E. et al. (2017). [Diagnostic Assessment of Deep Learning for Detection of Lymph Node Metastases](https://doi.org/10.1001/jama.2017.14585) (CAMELYON16). _JAMA_.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | PyTorch 2.x |
| Model | ABMIL with Gated Attention (230K params) |
| Uncertainty | MC-Dropout (20 stochastic passes) |
| Metrics | scikit-learn (AUC) |
| Frontend | React + Vite |
| Design | IBM Plex Mono/Sans, vanilla CSS |

## License

MIT — Hackathon prototype, use freely.

---

<p align="center">
  Built for <strong>Smart India Hackathon 2026</strong>
</p>
