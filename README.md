<p align="center">
  <strong>AI-Powered Breast Cancer Histopathology Triage System</strong><br/>
  <em>Attention-Based MIL + Monte Carlo Dropout Uncertainty Estimation + Dual Dataset Evaluation</em>
</p>

<p align="center">
  <code>SIH 2026 Prototype (v2)</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>PyTorch</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>TorchMacenko</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>ResNet-50</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>React</code>
</p>

<p align="center">
  <a href="https://github.com/yashkokane1031/SIH-2026-Prototype/raw/main/SIH_Demo.mp4">
    <img src="https://img.shields.io/badge/▶_Watch_Demo-Video_(v1_Baseline)-B91C1C?style=for-the-badge" alt="Watch Demo Video"/>
  </a>
</p>

> 📌 **Project Milestones & Versioning:**
> * **`v1-prototype` (Git Tag):** The original Round 1 submission state using the tuned synthetic MIL pipeline and initial dashboard.
> * **`main` (Current State):** Full integration of real **CAMELYON16** gigapixel WSIs (streaming ingestion, Macenko stain normalization, ResNet-50 2048-d feature encoding) alongside a live dual-dataset dashboard switcher.

---

## The Problem

Every year, pathologists review **millions** of histopathology slides to detect breast cancer metastases in sentinel lymph nodes. Whole-Slide Images (WSIs) are gigapixel-scale — a single slide can contain over **100,000 tissue patches**. Manual review is:

- **Slow**: 15–30 minutes per slide
- **Error-prone**: Inter-observer disagreement rates of 10–25%
- **Unscalable**: Critical bottleneck in under-resourced hospital networks

## Our Solution

An end-to-end AI triage system that **prioritizes slides for pathologist review** using three core components:

```
                 ┌─────────────────────────────────────────────────────────┐
                 │                        PIPELINE                         │
                 │                                                         │
  Slide (WSI)    │   ┌──────────────┐   ┌──────────┐   ┌──────────────┐   │   Triage
  (Real/Synth) ──┼──►│ TorchMacenko │──►│ Attention │──►│ MC-Dropout   │───┼──► Decision
                 │   │ + ResNet-50  │   │ MIL      │   │ Uncertainty  │   │
                 │   │ (2048-d)     │   │ (ABMIL)  │   │ (N=20 runs)  │   │   Tier 1/2/3
                 │   └──────────────┘   └──────────┘   └──────────────┘   │
                 │                                                         │
                 └─────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     React Dashboard       │
                               │  - [Real] / [Synth] toggle│
                               │  - Spatial attention maps │
                               │  - Pathologist audit log  │
                               └───────────────────────────┘
```

### 3-Tier Clinical Triage Routing

| Tier | Label | Routing Criteria | Recommended Action |
|:----:|:---|:---|:---|
| **1** | **Urgent** | Malignancy probability $\ge 70\%$, low uncertainty | Fast-track review within 24 hours |
| **2** | **Routine** | Malignancy probability $\le 30\%$, low uncertainty | Standard queue review |
| **3** | **Flagged / Uncertain** | Predictive standard deviation $\ge 0.15$ (high uncertainty) | Mandatory expert second opinion |

---

## Datasets & Results

The system evaluates two distinct cohorts, both accessible directly in the React dashboard:

### 1. Real CAMELYON16 Cohort ($N=20$ Whole Slide Images)
* **Ingestion:** 20 balanced gigapixel WSIs (10 Normal, 10 Tumor) streamed from the AWS S3 open data mirror.
* **Preprocessing:** Otsu tissue segmentation, tiled at 20x ($256 \times 256$), normalized with `TorchMacenkoNormalizer`, and encoded via pretrained **ResNet-50** into $2048$-d embeddings ($8,000$ real patches total).
* **Split:** 12 Train (6 Normal, 6 Tumor), 4 Validation (2 Normal, 2 Tumor), 4 Test (2 Normal, 2 Tumor).

| Split | Number of Slides | ROC-AUC | Clinical Breakdown / Observations |
|:---|:---:|:---:|:---|
| **Train** | 12 slides | **1.0000** | Overfitting to the 12-slide training set (623K parameters). |
| **Validation** | 4 slides | **0.7500** | Correctly identified macrometastasis (`tumor_001` @ 0.8982), 1 false positive (`normal_010`). |
| **Held-Out Test** | 4 slides | **0.7500** | 3 out of 4 pairwise comparisons correctly ranked. Correctly identified benigns (`normal_001`, `normal_002`). |

*(Note: MC-Dropout inference yields minor run-to-run stochastic variance of $\pm 0.01\text{--}0.02$ due to random dropout mask sampling across the 20 forward passes).*

> ⚠️ **Micrometastasis Under-Calling Limitation:** On real held-out slides with small metastatic foci (`tumor_007` prob 0.4343, `tumor_010` prob 0.2405), the 400-patch grid and 12-slide training budget resulted in lower predicted probabilities. Addressing this requires scaling to a larger training cohort and pathology-specific foundation models.

---

### 2. Tuned Synthetic Benchmark ($N=550$ Slides, $N=50$ Test Slides)
* **Configuration:** 512-d embeddings, $\mu_{\text{shift}}=0.17$, $\sigma=0.06$, noise std $=1.50$, 400 Train / 100 Val / 50 Test slides.
* **Validation ROC-AUC:** **`0.8833`**
* **Test ROC-AUC:** **`0.8637`**

#### Test Set Triage Distribution ($N=50$ slides)
| Tier | Slide Count | Percentage | Clinical Role |
|:---|:---:|:---:|:---|
| **Urgent (Tier 1)** | 19 slides | **38.0%** | High-confidence malignant cases |
| **Routine (Tier 2)** | 24 slides | **48.0%** | High-confidence benign cases |
| **Flagged / Uncertain (Tier 3)** | 7 slides | **14.0%** | Borderline / ambiguous cases routed for expert review |

---

## React Dashboard with Dual-Dataset Switcher

The React dashboard includes a header toggle that allows instant switching between both datasets:
```
[ Real CAMELYON16 ]   [ Synthetic Benchmark ]
```

* **Real CAMELYON16 View:** Displays the 20 real WSI feature bags, complete with patch coordinates, attention weights, and triage routing.
* **Synthetic Benchmark View:** Displays the 50 test slides exhibiting the full 3-tier spread (including all 7 flagged uncertain cases).
* **Attention Heatmap Viewer:** Visualizes patch-level attention distributions to explain model reasoning.
* **Pathologist Audit Trail:** Interactive *Confirm* and *Override* action logging.

---

## Quick Start

### 1. Python Environment Setup

Install the exact verified dependency versions:

```bash
# Core deep learning & scientific computing
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.24.0 scikit-learn>=1.3.0 requests>=2.28.0

# Whole-Slide Image handling & stain normalization (Windows-compatible binary wheels)
pip install openslide-bin==4.0.1.2 openslide-python==1.4.6 torchstain==1.4.1
```

### 2. Run the Pipeline

```bash
# Run both Real CAMELYON16 and Synthetic pipelines (default):
python pipeline.py --dataset both

# Or run a specific dataset:
python pipeline.py --dataset camelyon16
python pipeline.py --dataset synthetic
```

### 3. Launch the React Dashboard

```bash
cd dashboard
npm install
npm run dev

# Open http://localhost:5173/ in your browser
```

### 4. Evaluate Splits & Reproduce Diagnostics

```bash
# Run the split evaluation diagnostic:
python scratch/eval_splits.py
```

---

## Repository Structure

```
SIH 26 Prototype/
├── pipeline.py                              # End-to-end ABMIL + MC-Dropout training & triage pipeline
├── download_and_extract_camelyon16.py       # Streaming WSI download, Macenko norm & ResNet-50 extraction
├── report.py                                # Benchmark reporting utility
├── scratch/
│   └── eval_splits.py                       # Split-level validation/test AUC evaluation tool
├── data/
│   ├── camelyon16_reference.csv             # Slide manifest with labels & AWS S3 URLs
│   ├── macenko_sample_comparison.png        # Authentic WSI patch before/after Macenko normalization proof
│   └── camelyon16_features/                 # Cached ResNet-50 feature embeddings (20 .npz files, ~28 MB)
│
├── dashboard/                               # React + Vite frontend
│   ├── public/
│   │   ├── triage_results_camelyon16.json   # Live data feed for Real CAMELYON16
│   │   └── triage_results_synthetic.json    # Live data feed for Synthetic Benchmark
│   └── src/
│       ├── App.jsx                          # Root component with dataset switcher
│       └── components/
│           ├── TriageQueue.jsx              # Prioritized triage queue
│           └── SlideDetail.jsx              # Spatial attention heatmap & audit panel
│
└── README.md
```

---

## Limitations & Future Work

1. **Real-Data Cohort Size:** Current real-world validation is conducted on a 20-slide subset ($N=20$). Scaling to the full 270+ CAMELYON16 training set is planned for production deployment.
2. **Micrometastasis Sensitivity:** Improving sensitivity on tiny metastatic foci ($<2\text{ mm}$) through denser multiscale patch tiling and hard negative mining.
3. **Model Calibration:** Implementing temperature scaling, Expected Calibration Error (ECE) quantification, and reliability diagrams to optimize triage thresholds.
4. **Foundation Model Integration:** Replacing standard ImageNet ResNet-50 with pathology-specific vision foundation models (such as **UNI**, **CONCH**, or **Prov-GigaPath**).
5. **Multi-Center Stain Robustness:** Validating Macenko normalization across diverse scanner profiles and staining protocols.

---

## Tech Stack

| Component | Technology |
|:---|:---|
| **Deep Learning & Modeling** | PyTorch, TorchVision, Scikit-Learn |
| **Pathology & Stain Normalization** | OpenSlide (`openslide-python` + `openslide-bin`), TorchStain (`TorchMacenkoNormalizer`) |
| **Feature Extraction** | Pretrained ResNet-50 ($2048$-d embeddings) |
| **Uncertainty Quantification** | Monte Carlo Dropout ($N=20$ stochastic forward passes) |
| **Frontend Dashboard** | React 19, Vite, IBM Plex Typography, Vanilla CSS |

---

<p align="center">
  Built for <strong>Smart India Hackathon 2026</strong>
</p>
