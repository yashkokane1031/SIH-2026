import React, { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";

/* =========================================================
   DATA HELPERS
========================================================= */

const TIER_MAP = { urgent: "URGENT", uncertain: "FLAGGED", routine: "ROUTINE" };

function mapSlideToCase(slide) {
  const tier = TIER_MAP[slide.tier] || "ROUTINE";
  const prob = +(slide.malignancy_probability * 100).toFixed(2);
  const notes = {
    URGENT: "High malignancy — immediate pathologist review recommended",
    FLAGGED: "Model uncertainty elevated — expert confirmation required",
    ROUTINE: "Low malignancy probability — standard processing queue",
  };
  return {
    id: slide.slide_id,
    patient: `Case ${slide.slide_id}`,
    tissue: slide.slide_id.startsWith("test_") ? "Synthetic" : "Lymph Node",
    tier,
    probability: prob,
    uncertainty: +slide.uncertainty_score.toFixed(4),
    patches: slide.num_patches,
    status: "Pending",
    date: new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }),
    note: notes[tier],
    trueLabel: slide.true_label,
    patchCoordinates: slide.patch_coordinates,
    patchAttentionWeights: slide.patch_attention_weights,
  };
}

/* =========================================================
   HEATMAP COLOR — GAMMA-CORRECTED MAGMA COLORMAP
   Ported from verified SlideDetail.jsx
========================================================= */

function getPercentile(arr, p) {
  if (!arr || arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (idx - lower);
}

function heatmapColor(value) {
  const raw = Math.max(0, Math.min(1, value));
  const clamp = Math.pow(raw, 0.45);
  const stops = [
    { pos: 0.0, r: 15, g: 23, b: 42 },
    { pos: 0.2, r: 49, g: 46, b: 129 },
    { pos: 0.45, r: 168, g: 85, b: 247 },
    { pos: 0.7, r: 245, g: 158, b: 11 },
    { pos: 0.9, r: 239, g: 68, b: 68 },
    { pos: 1.0, r: 254, g: 240, b: 138 },
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (clamp >= stops[i].pos && clamp <= stops[i + 1].pos) {
      lo = stops[i]; hi = stops[i + 1]; break;
    }
  }
  const t = hi.pos === lo.pos ? 0 : (clamp - lo.pos) / (hi.pos - lo.pos);
  const r = Math.round(lo.r + t * (hi.r - lo.r));
  const g = Math.round(lo.g + t * (hi.g - lo.g));
  const b = Math.round(lo.b + t * (hi.b - lo.b));
  return `rgb(${r}, ${g}, ${b})`;
}

/* =========================================================
   NAVIGATION
========================================================= */

function Navigation({ activePage, setActivePage }) {
  return (
    <nav className="navbar">
      <button className="brand" onClick={() => setActivePage("dashboard")}>
        <div className="brand-mark"><span /><span /><span /></div>
        <div><strong>PATHO</strong><small>INTELLIGENCE</small></div>
      </button>

      <div className="nav-links">
        <button className={activePage === "dashboard" ? "active" : ""} onClick={() => setActivePage("dashboard")}>Overview</button>
        <button onClick={() => { setActivePage("dashboard"); setTimeout(() => document.getElementById("queue")?.scrollIntoView({ behavior: "smooth" }), 50); }}>Cases</button>
        <button onClick={() => { setActivePage("dashboard"); setTimeout(() => document.getElementById("workflow")?.scrollIntoView({ behavior: "smooth" }), 50); }}>How it works</button>
      </div>
    </nav>
  );
}

/* =========================================================
   STATUS BADGE
========================================================= */

function StatusBadge({ tier }) {
  return (
    <span className={`badge ${tier.toLowerCase()}`}>
      <i />{tier}
    </span>
  );
}

/* =========================================================
   DONUT
========================================================= */

function Donut({ value }) {
  const radius = 43;
  const circumference = 2 * Math.PI * radius;
  const progress = circumference - (value / 100) * circumference;
  return (
    <div className="donut">
      <svg viewBox="0 0 110 110">
        <circle cx="55" cy="55" r={radius} className="donut-bg" />
        <circle cx="55" cy="55" r={radius} className="donut-value" strokeDasharray={circumference} strokeDashoffset={progress} />
      </svg>
      <div className="donut-label">
        <strong>{value.toFixed(1)}%</strong>
        <span>malignant</span>
      </div>
    </div>
  );
}

/* =========================================================
   CLINICAL HERO — Telemetry Donut Chart Status Card
========================================================= */

function ClinicalHero({ casesData, activeDataset, onScrollToQueue }) {
  const urgent = casesData.filter((c) => c.tier === "URGENT").length;
  const flagged = casesData.filter((c) => c.tier === "FLAGGED").length;
  const routine = casesData.filter((c) => c.tier === "ROUTINE").length;
  const total = casesData.length;
  const flaggedPct = total > 0 ? ((flagged / total) * 100).toFixed(1) : "0.0";
  const urgentPct = total > 0 ? ((urgent / total) * 100).toFixed(1) : "0.0";
  const routinePct = total > 0 ? ((routine / total) * 100).toFixed(1) : "0.0";

  // SVG Donut Calculations
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const urgentArc = total > 0 ? (urgent / total) * circumference : 0;
  const flaggedArc = total > 0 ? (flagged / total) * circumference : 0;
  const routineArc = total > 0 ? (routine / total) * circumference : 0;

  return (
    <section className="hero-section">
      <div className="hero-grid-bg" />

      <div className="hero-content">
        <span className="hero-label">CLINICAL DECISION SUPPORT</span>

        <h1>
          Precision triage.
          <br />
          <span>Evidence-driven review.</span>
        </h1>

        <p>
          AI-assisted whole-slide analysis combining attention-based MIL
          with MC-Dropout uncertainty estimation to prioritize cases for
          pathologist review.
        </p>

        <div className="hero-buttons">
          <button onClick={onScrollToQueue}>Review cases →</button>
          <button className="light-button" onClick={() => document.getElementById("workflow")?.scrollIntoView({ behavior: "smooth" })}>How it works</button>
        </div>
      </div>

      <div className="hero-visual hero-data-panel">
        <div className="hero-readout">
          <div className="readout-header">
            <div className="readout-live-indicator">
              <span className="readout-dot" />
              <span>LIVE TRIAGE TELEMETRY</span>
            </div>
            <span className="readout-dataset-pill">
              {activeDataset === "camelyon16" ? "CAMELYON16 (20 WSI)" : "SYNTHETIC (50 WSI)"}
            </span>
          </div>

          {/* Donut Chart & Telemetry Legend */}
          <div className="readout-donut-section">
            <div className="readout-donut-wrapper">
              <svg viewBox="0 0 130 130" className="readout-donut-svg">
                {/* Background Track */}
                <circle
                  cx="65"
                  cy="65"
                  r={radius}
                  className="readout-donut-track"
                  strokeWidth="14"
                  fill="none"
                />
                {/* Urgent Arc (Crimson) */}
                {urgent > 0 && (
                  <circle
                    cx="65"
                    cy="65"
                    r={radius}
                    stroke="#EF4444"
                    strokeWidth="14"
                    fill="none"
                    strokeDasharray={`${urgentArc} ${circumference}`}
                    strokeDashoffset="0"
                    transform="rotate(-90 65 65)"
                    className="donut-segment"
                  />
                )}
                {/* Flagged Arc (Amber) */}
                {flagged > 0 && (
                  <circle
                    cx="65"
                    cy="65"
                    r={radius}
                    stroke="#F59E0B"
                    strokeWidth="14"
                    fill="none"
                    strokeDasharray={`${flaggedArc} ${circumference}`}
                    strokeDashoffset={-urgentArc}
                    transform="rotate(-90 65 65)"
                    className="donut-segment"
                  />
                )}
                {/* Routine Arc (Teal) */}
                {routine > 0 && (
                  <circle
                    cx="65"
                    cy="65"
                    r={radius}
                    stroke="#10B981"
                    strokeWidth="14"
                    fill="none"
                    strokeDasharray={`${routineArc} ${circumference}`}
                    strokeDashoffset={-(urgentArc + flaggedArc)}
                    transform="rotate(-90 65 65)"
                    className="donut-segment"
                  />
                )}
                {/* Center Label */}
                <text x="65" y="61" textAnchor="middle" className="donut-center-total">
                  {total}
                </text>
                <text x="65" y="75" textAnchor="middle" className="donut-center-sub">
                  SLIDES
                </text>
              </svg>
            </div>

            {/* Donut Legend */}
            <div className="readout-donut-legend">
              <div className="legend-row row-urgent">
                <div className="legend-left">
                  <span className="legend-bullet bullet-urgent" />
                  <span className="legend-name">URGENT</span>
                </div>
                <div className="legend-right">
                  <strong className="legend-count">{urgent}</strong>
                  <span className="legend-pct">({urgentPct}%)</span>
                </div>
              </div>

              <div className="legend-row row-flagged">
                <div className="legend-left">
                  <span className="legend-bullet bullet-flagged" />
                  <span className="legend-name">FLAGGED</span>
                </div>
                <div className="legend-right">
                  <strong className="legend-count">{flagged}</strong>
                  <span className="legend-pct">({flaggedPct}%)</span>
                </div>
              </div>

              <div className="legend-row row-routine">
                <div className="legend-left">
                  <span className="legend-bullet bullet-routine" />
                  <span className="legend-name">ROUTINE</span>
                </div>
                <div className="legend-right">
                  <strong className="legend-count">{routine}</strong>
                  <span className="legend-pct">({routinePct}%)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Footer Telemetry Strip */}
          <div className="readout-footer">
            <div className="footer-metric">
              <small>TOTAL COHORT</small>
              <b>{total} SLIDES</b>
            </div>
            <div className="footer-metric">
              <small>UNCERTAINTY ROUTER</small>
              <b>σ &gt; 0.15</b>
            </div>
            <div className="footer-metric">
              <small>ARCHITECTURE</small>
              <b>ABMIL + MC-DROPOUT</b>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* =========================================================
   MINI SLIDE HEATMAP THUMBNAIL (Real Per-Slide Attention Map)
========================================================= */

function SlideMiniHeatmap({ patchCoordinates, patchAttentionWeights }) {
  const miniData = useMemo(() => {
    const coords = patchCoordinates || [];
    const weights = patchAttentionWeights || [];
    if (coords.length === 0) return null;

    const xs = coords.map(([x]) => x);
    const ys = coords.map(([, y]) => y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const spanX = Math.max(1, maxX - minX);
    const spanY = Math.max(1, maxY - minY);
    const isCompactGrid = spanX <= 25 && spanY <= 25;

    let cMinX = minX, cMaxX = maxX, cMinY = minY, cMaxY = maxY;
    if (!isCompactGrid) {
      const p02X = getPercentile(xs, 2), p98X = getPercentile(xs, 98);
      const p02Y = getPercentile(ys, 2), p98Y = getPercentile(ys, 98);
      const coreSpanX = Math.max(1, p98X - p02X);
      const coreSpanY = Math.max(1, p98Y - p02Y);
      cMinX = p02X - coreSpanX * 0.05;
      cMaxX = p98X + coreSpanX * 0.05;
      cMinY = p02Y - coreSpanY * 0.05;
      cMaxY = p98Y + coreSpanY * 0.05;
    }

    const cSpanX = Math.max(1, cMaxX - cMinX);
    const cSpanY = Math.max(1, cMaxY - cMinY);
    const w = 260, h = 130, pad = 10;
    const availW = w - pad * 2, availH = h - pad * 2;
    const scale = Math.min(availW / cSpanX, availH / cSpanY);
    const offX = pad + (availW - cSpanX * scale) / 2;
    const offY = pad + (availH - cSpanY * scale) / 2;
    const patchSize = isCompactGrid
      ? Math.max(7, scale * 0.95)
      : Math.max(6, Math.min(13, Math.sqrt((availW * availH) / (coords.length * 1.5))));

    const patches = coords.map(([x, y], i) => ({
      px: offX + (x - cMinX) * scale,
      py: offY + (y - cMinY) * scale,
      color: heatmapColor(weights[i] ?? 0),
    }));

    return { w, h, patchSize, patches, count: coords.length };
  }, [patchCoordinates, patchAttentionWeights]);

  if (!miniData) {
    return (
      <div className="mini-slide-fallback">
        <span>No Patch Data</span>
      </div>
    );
  }

  return (
    <div className="mini-heatmap-container">
      <svg
        className="mini-heatmap-svg"
        viewBox={`0 0 ${miniData.w} ${miniData.h}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <rect width={miniData.w} height={miniData.h} fill="#090E1A" rx="7" />
        {/* Subtle grid watermark */}
        <line x1="0" y1="65" x2={miniData.w} y2="65" stroke="rgba(59, 130, 246, 0.07)" strokeWidth="1" strokeDasharray="3 3" />
        <line x1="130" y1="0" x2="130" y2={miniData.h} stroke="rgba(59, 130, 246, 0.07)" strokeWidth="1" strokeDasharray="3 3" />
        {miniData.patches.map((p, idx) => (
          <rect
            key={idx}
            x={p.px - miniData.patchSize / 2}
            y={p.py - miniData.patchSize / 2}
            width={miniData.patchSize}
            height={miniData.patchSize}
            fill={p.color}
            rx="1"
          />
        ))}
      </svg>
      <div className="mini-heatmap-overlay">
        <span className="mini-tag-patches">{miniData.count} patches</span>
        <span className="mini-tag-wsi">WSI Heatmap</span>
      </div>
    </div>
  );
}

/* =========================================================
   ATTENTION MAP — Bounding-box SVG with density crop
   Ported from verified SlideDetail.jsx
========================================================= */

function Heatmap({ patchCoordinates, patchAttentionWeights, selectedPatch, setSelectedPatch }) {
  const heatmapData = useMemo(() => {
    const coords = patchCoordinates || [];
    const weights = patchAttentionWeights || [];
    if (coords.length === 0) return null;

    const xs = coords.map(([x]) => x);
    const ys = coords.map(([, y]) => y);

    const fullMinX = Math.min(...xs), fullMaxX = Math.max(...xs);
    const fullMinY = Math.min(...ys), fullMaxY = Math.max(...ys);
    const spanX = Math.max(1, fullMaxX - fullMinX);
    const spanY = Math.max(1, fullMaxY - fullMinY);
    const isCompactGrid = spanX <= 25 && spanY <= 25;

    let cropMinX = fullMinX, cropMaxX = fullMaxX;
    let cropMinY = fullMinY, cropMaxY = fullMaxY;

    if (!isCompactGrid) {
      const p02X = getPercentile(xs, 2), p98X = getPercentile(xs, 98);
      const p02Y = getPercentile(ys, 2), p98Y = getPercentile(ys, 98);
      const coreSpanX = Math.max(1, p98X - p02X);
      const coreSpanY = Math.max(1, p98Y - p02Y);
      cropMinX = p02X - coreSpanX * 0.04;
      cropMaxX = p98X + coreSpanX * 0.04;
      cropMinY = p02Y - coreSpanY * 0.04;
      cropMaxY = p98Y + coreSpanY * 0.04;
    }

    const cropSpanX = Math.max(1, cropMaxX - cropMinX);
    const cropSpanY = Math.max(1, cropMaxY - cropMinY);
    const canvasW = 600, canvasH = 600, pad = 24;
    const availW = canvasW - pad * 2, availH = canvasH - pad * 2;
    const scale = Math.min(availW / cropSpanX, availH / cropSpanY);
    const offX = pad + (availW - cropSpanX * scale) / 2;
    const offY = pad + (availH - cropSpanY * scale) / 2;

    const patchSize = isCompactGrid
      ? Math.max(16, scale * 0.92)
      : Math.max(16, Math.min(26, Math.sqrt((availW * availH) / (coords.length * 1.5))));

    const patches = coords.map(([x, y], i) => ({
      index: i, origX: x, origY: y,
      px: offX + (x - cropMinX) * scale,
      py: offY + (y - cropMinY) * scale,
      weight: weights[i] ?? 0,
      color: heatmapColor(weights[i] ?? 0),
    }));

    return { canvasW, canvasH, patchSize, patches };
  }, [patchCoordinates, patchAttentionWeights]);

  return (
    <div className="attention-panel">
      <div className="viewer-top">
        <div>
          <span className="mini-label">ABMIL VISUALIZATION</span>
          <h3>Attention Map</h3>
          <p>Patch-level attention weights — brighter regions indicate high model focus.</p>
        </div>
      </div>

      <div className="attention-map-wrapper">
        {heatmapData && (
          <svg className="attention-svg" viewBox={`0 0 ${heatmapData.canvasW} ${heatmapData.canvasH}`} preserveAspectRatio="xMidYMid meet">
            <rect x="0" y="0" width={heatmapData.canvasW} height={heatmapData.canvasH} fill="#0C1222" rx="6" />
            {heatmapData.patches.map((p) => {
              const isHovered = selectedPatch && selectedPatch.id === p.index;
              const half = heatmapData.patchSize / 2;
              return (
                <rect
                  key={p.index}
                  className="attention-patch"
                  x={p.px - half} y={p.py - half}
                  width={heatmapData.patchSize} height={heatmapData.patchSize}
                  fill={p.color} rx="2"
                  stroke={isHovered ? "#FFFFFF" : "rgba(0,0,0,0.35)"}
                  strokeWidth={isHovered ? 2.5 : 0.6}
                  onClick={() => setSelectedPatch({ id: p.index, score: p.weight, row: p.origY, column: p.origX })}
                >
                  <title>{`Patch #${p.index} (${p.origX}, ${p.origY})\nAttention: ${p.weight.toFixed(4)}`}</title>
                </rect>
              );
            })}
          </svg>
        )}

        <div className="attention-scale">
          <span>Low</span>
          <div className="attention-gradient" />
          <span>High</span>
        </div>

        {selectedPatch && (
          <div className="selected-patch-card">
            <span>SELECTED PATCH</span>
            <strong>Patch #{selectedPatch.id}</strong>
            <div><small>Position</small><b>({selectedPatch.column}, {selectedPatch.row})</b></div>
            <div><small>Attention weight</small><b>{(selectedPatch.score * 100).toFixed(1)}%</b></div>
          </div>
        )}
      </div>
    </div>
  );
}

/* =========================================================
   DOCTOR REVIEW — Session-persistent per slide_id
========================================================= */

function DoctorReview({ casesData, reviews, onSaveReview, onOpenCase }) {
  const [selectedSlideId, setSelectedSlideId] = useState("");
  const [doctorName, setDoctorName] = useState("");
  const [diagnosis, setDiagnosis] = useState("");
  const [comment, setComment] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (selectedSlideId && reviews[selectedSlideId]) {
      const r = reviews[selectedSlideId];
      setDoctorName(r.doctorName || "");
      setDiagnosis(r.diagnosis || "");
      setComment(r.comment || "");
    } else {
      setDoctorName(""); setDiagnosis(""); setComment("");
    }
  }, [selectedSlideId, reviews]);

  useEffect(() => {
    if (casesData.length > 0 && !selectedSlideId) {
      setSelectedSlideId(casesData[0].id);
    }
  }, [casesData, selectedSlideId]);

  function saveReview(event) {
    event.preventDefault();
    if (!selectedSlideId) return;
    onSaveReview(selectedSlideId, {
      doctorName, diagnosis, comment,
      timestamp: new Date().toISOString(),
    });
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  }

  return (
    <section className="doctor-review-section" id="doctor-review">
      <div className="doctor-review-intro">
        <span>CLINICAL REVIEW WORKSPACE</span>
        <h2>Doctor<br /><em>review.</em></h2>
        <p>A dedicated review space for pathology professionals to record their interpretation after reviewing the model evidence.</p>
        <button className="review-case-button" onClick={() => { if (casesData.length > 0) onOpenCase(casesData[0]); }}>Open priority case →</button>
      </div>

      <div className="review-cassette">
        <div className="cassette-top">
          <div><span>PATHO / CLINICAL REVIEW</span><strong>DOCTOR REVIEW</strong></div>
          <div className="cassette-light"><i />READY</div>
        </div>

        <form onSubmit={saveReview}>
          <div className="cassette-row">
            <label>
              DOCTOR NAME
              <input value={doctorName} onChange={(e) => setDoctorName(e.target.value)} placeholder="Enter reviewer name" />
            </label>
            <label>
              CASE
              <select value={selectedSlideId} onChange={(e) => setSelectedSlideId(e.target.value)}>
                {casesData.map((c) => (
                  <option key={c.id} value={c.id}>{c.id}{reviews[c.id] ? " ✓" : ""}</option>
                ))}
              </select>
            </label>
          </div>

          <label>
            CLINICAL IMPRESSION
            <select value={diagnosis} onChange={(e) => setDiagnosis(e.target.value)}>
              <option value="">Select interpretation</option>
              <option value="Malignant">Malignant</option>
              <option value="Benign">Benign</option>
              <option value="Indeterminate">Indeterminate</option>
              <option value="Requires further review">Requires further review</option>
            </select>
          </label>

          <label>
            REVIEW NOTES
            <textarea value={comment} onChange={(e) => setComment(e.target.value)} placeholder="Add clinical observations, supporting evidence or follow-up notes..." rows="4" />
          </label>

          <div className="cassette-bottom">
            <div className="cassette-status"><span />PRIVATE CLINICAL WORKSPACE</div>
            <button type="submit" className="save-review-button">{saved ? "✓ Review saved" : "Save doctor review"}</button>
          </div>
        </form>

        <div className="cassette-reel reel-left"><span /></div>
        <div className="cassette-reel reel-right"><span /></div>
      </div>
    </section>
  );
}

/* =========================================================
   DATASET TOGGLE
========================================================= */

function DatasetToggle({ activeDataset, setActiveDataset }) {
  return (
    <div className="dataset-toggle">
      <button className={activeDataset === "camelyon16" ? "toggle-active" : ""} onClick={() => setActiveDataset("camelyon16")}>
        Real CAMELYON16
      </button>
      <button className={activeDataset === "synthetic" ? "toggle-active" : ""} onClick={() => setActiveDataset("synthetic")}>
        Synthetic Benchmark
      </button>
    </div>
  );
}

/* =========================================================
   MAIN APP
========================================================= */

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [selectedCase, setSelectedCase] = useState(null);
  const [filter, setFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [selectedPatch, setSelectedPatch] = useState(null);
  const [activeDataset, setActiveDataset] = useState("camelyon16");
  const [reviews, setReviews] = useState({});

  // Loaded data
  const [camelyonSlides, setCamelyonSlides] = useState([]);
  const [syntheticSlides, setSyntheticSlides] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/triage_results_camelyon16.json").then((r) => r.json()),
      fetch("/triage_results_synthetic.json").then((r) => r.json()),
    ]).then(([cam, syn]) => {
      setCamelyonSlides(cam.slides.map(mapSlideToCase));
      setSyntheticSlides(syn.slides.map(mapSlideToCase));
      setLoading(false);
    }).catch((err) => {
      console.error("Failed to load triage data:", err);
      setLoading(false);
    });
  }, []);

  const casesData = activeDataset === "camelyon16" ? camelyonSlides : syntheticSlides;

  const urgent = casesData.filter((c) => c.tier === "URGENT").length;
  const flagged = casesData.filter((c) => c.tier === "FLAGGED").length;
  const routine = casesData.filter((c) => c.tier === "ROUTINE").length;

  const filteredCases = casesData.filter((item) => {
    const tierMatch = filter === "ALL" || item.tier === filter;
    const searchMatch = item.id.toLowerCase().includes(search.toLowerCase());
    return tierMatch && searchMatch;
  });

  const openCase = useCallback((item) => {
    setSelectedCase(item);
    setSelectedPatch(null);
    setActivePage("case");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  function updateCaseStatus(status) {
    const updater = (oldCases) => oldCases.map((c) => c.id === selectedCase.id ? { ...c, status } : c);
    setCamelyonSlides(updater);
    setSyntheticSlides(updater);
    setSelectedCase({ ...selectedCase, status });
  }

  function handleSaveReview(slideId, review) {
    setReviews((prev) => ({ ...prev, [slideId]: review }));
  }

  /* =====================================================
     LOADING STATE
  ===================================================== */

  if (loading) {
    return (
      <div className="app">
        <Navigation activePage={activePage} setActivePage={setActivePage} />
        <main style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
          <div style={{ textAlign: "center", color: "var(--muted)" }}>
            <p style={{ fontSize: "1.2rem", letterSpacing: "0.08em", fontWeight: 600 }}>LOADING TRIAGE DATA...</p>
          </div>
        </main>
      </div>
    );
  }

  /* =====================================================
     CASE PAGE
  ===================================================== */

  if (activePage === "case" && selectedCase) {
    const existingReview = reviews[selectedCase.id];
    return (
      <div className="app">
        <Navigation activePage={activePage} setActivePage={setActivePage} />
        <main className="case-page">
          <button className="back-link" onClick={() => { setActivePage("dashboard"); setSelectedCase(null); }}>← Back to overview</button>

          <section className="case-heading">
            <div>
              <div className="case-id-line">
                <span className="case-number">{selectedCase.id}</span>
                <StatusBadge tier={selectedCase.tier} />
                {selectedCase.trueLabel !== undefined && (
                  <span className={`ground-truth-badge ${selectedCase.trueLabel === 1 ? "gt-malignant" : "gt-benign"}`}>
                    GT: {selectedCase.trueLabel === 1 ? "Malignant" : "Benign"}
                  </span>
                )}
              </div>
              <h1>pathology analysis</h1>
              <p>Review the model prediction, uncertainty, and attention evidence for this case.</p>
            </div>
            <div className="model-status"><span />MODEL ONLINE</div>
          </section>

          <section className="case-metrics">
            <div className="case-metric large">
              <div>
                <span>MALIGNANCY PROBABILITY</span>
                <strong className="red-text">{selectedCase.probability}%</strong>
                <p>{selectedCase.probability >= 70 ? "High confidence malignant prediction" : selectedCase.probability >= 30 ? "Moderate — requires expert review" : "Low malignancy probability"}</p>
              </div>
              <Donut value={selectedCase.probability} />
            </div>

            <div className="case-metric">
              <span>UNCERTAINTY</span>
              <strong>{selectedCase.uncertainty}</strong>
              <small>MC-Dropout estimate</small>
            </div>

            <div className="case-metric">
              <span>PATCHES ANALYZED</span>
              <strong>{selectedCase.patches.toLocaleString()}</strong>
              <small>attention-weighted instances</small>
            </div>
          </section>

          <section className="analysis-layout">
            <div>
              <Heatmap
                patchCoordinates={selectedCase.patchCoordinates}
                patchAttentionWeights={selectedCase.patchAttentionWeights}
                selectedPatch={selectedPatch}
                setSelectedPatch={setSelectedPatch}
              />

              <div className="method-card">
                <div className="method-title">
                  <span>MODEL INTERPRETATION</span>
                  <span className="confidence">{selectedCase.probability >= 70 ? "HIGH CONFIDENCE" : selectedCase.probability >= 30 ? "MODERATE" : "LOW RISK"}</span>
                </div>
                <div className="method-grid">
                  <div><b>01</b><strong>Patch extraction</strong><p>Tissue regions are divided into smaller image patches.</p></div>
                  <div><b>02</b><strong>ABMIL attention</strong><p>Attention identifies the most influential regions.</p></div>
                  <div><b>03</b><strong>MC-Dropout</strong><p>Multiple predictions estimate model uncertainty.</p></div>
                </div>
              </div>
            </div>

            <aside className="decision-panel">
              <div className="decision-head">
                <span>Model DECISION</span>
                <span className="live-dot">LIVE</span>
              </div>

              <div className="decision-result">
                <span>TRIAGE RESULT</span>
                <h2>{selectedCase.tier === "URGENT" ? "Malignant" : selectedCase.tier === "FLAGGED" ? "Needs Review" : "Likely Benign"}</h2>
                <p>{selectedCase.note}. The highlighted regions represent areas receiving higher model attention.</p>
              </div>

              <div className="decision-list">
                <div><span>Probability</span><b>{selectedCase.probability}%</b></div>
                <div><span>Uncertainty</span><b>{selectedCase.uncertainty}</b></div>
                <div><span>Model</span><b>ABMIL v1.0</b></div>
                <div><span>Status</span><b>{selectedCase.status}</b></div>
              </div>

              {existingReview && (
                <div className="saved-review-card">
                  <strong>Previous Review</strong>
                  <div><small>Doctor</small><span>{existingReview.doctorName || "—"}</span></div>
                  <div><small>Impression</small><span>{existingReview.diagnosis || "—"}</span></div>
                  {existingReview.comment && <div><small>Notes</small><span>{existingReview.comment}</span></div>}
                  <div><small>Saved</small><span>{new Date(existingReview.timestamp).toLocaleString()}</span></div>
                </div>
              )}

              <div className="decision-buttons">
                <button className="primary-button" onClick={() => updateCaseStatus("Reviewed")}>✓ Mark as reviewed</button>
                <button className="secondary-button" onClick={() => updateCaseStatus("Escalated")}>Escalate case</button>
              </div>

              <div className="clinical-note">
                <strong>Clinical note</strong>
                <p>Model predictions are decision support only. Final diagnosis should be made by a qualified pathology professional.</p>
              </div>
            </aside>
          </section>
        </main>
      </div>
    );
  }

  /* =====================================================
     DASHBOARD
  ===================================================== */

  return (
    <div className="app">
      <Navigation activePage={activePage} setActivePage={setActivePage} />

      <main>
        <ClinicalHero
          casesData={casesData}
          activeDataset={activeDataset}
          onScrollToQueue={() => document.getElementById("queue")?.scrollIntoView({ behavior: "smooth" })}
        />

        {/* STATS */}
        <section className="stats-section">
          <div className="stat"><span>URGENT</span><strong>{urgent}</strong><small>immediate review</small></div>
          <div className="stat"><span>FLAGGED</span><strong>{flagged}</strong><small>expert review</small></div>
          <div className="stat"><span>ROUTINE</span><strong>{routine}</strong><small>standard queue</small></div>
          <div className="stat"><span>TOTAL SLIDES</span><strong>{casesData.length}</strong><small>in dataset</small></div>
        </section>

        {/* REVIEW QUEUE */}
        <section className="queue-section" id="queue">
          <div className="section-heading">
            <div>
              <span>CASE MANAGEMENT</span>
              <h2>Review queue</h2>
              <p>Start with the cases that require the most attention.</p>
            </div>
            <div className="queue-controls">
              <DatasetToggle activeDataset={activeDataset} setActiveDataset={setActiveDataset} />
              <div className="search-box">
                <span>⌕</span>
                <input placeholder="Search slide ID" value={search} onChange={(e) => setSearch(e.target.value)} />
              </div>
              <select value={filter} onChange={(e) => setFilter(e.target.value)}>
                <option value="ALL">All cases</option>
                <option value="URGENT">Urgent</option>
                <option value="FLAGGED">Flagged</option>
                <option value="ROUTINE">Routine</option>
              </select>
            </div>
          </div>

          <div className="case-grid">
            {filteredCases.map((item) => (
              <article className="case-card" key={item.id} onClick={() => openCase(item)}>
                <div className="card-top">
                  <span className="case-code">{item.id}</span>
                  <StatusBadge tier={item.tier} />
                </div>

                {/* Real Per-Slide Attention Map Thumbnail */}
                <SlideMiniHeatmap
                  patchCoordinates={item.patchCoordinates}
                  patchAttentionWeights={item.patchAttentionWeights}
                />

                <div className="card-info">
                  <div><span>MALIGNANCY</span><strong className={item.probability >= 80 ? "red-text" : item.probability >= 20 ? "orange-text" : "green-text"}>{item.probability}%</strong></div>
                  <div><span>UNCERTAINTY</span><strong>{item.uncertainty}</strong></div>
                  <div><span>PATCHES</span><strong>{item.patches}</strong></div>
                </div>
                <div className="card-bottom">
                  <span>{item.note}</span>
                  <button>View →</button>
                </div>
              </article>
            ))}
          </div>
        </section>

        {/* WORKFLOW */}
        <section className="workflow" id="workflow">
          <div className="workflow-heading"><span>MODEL WORKFLOW</span><h2>From slide<br /><em>to insight.</em></h2></div>
          <div className="workflow-steps">
            <div><span>01</span><h3>Whole-slide image</h3><p>A digital pathology slide enters the AI pipeline.</p></div>
            <div><span>02</span><h3>Patch extraction</h3><p>Tissue is divided into manageable image regions.</p></div>
            <div><span>03</span><h3>ABMIL attention</h3><p>The model learns which regions matter most.</p></div>
            <div><span>04</span><h3>Human review</h3><p>Experts receive a prioritized and interpretable result.</p></div>
          </div>
        </section>

        {/* ABOUT */}
        <section className="about-model">
          <div><span>DESIGNED FOR INTERPRETABILITY</span><h2>Model that shows<br /><em>why.</em></h2></div>
          <p>Instead of presenting a single prediction, the system combines malignancy probability with uncertainty and attention-based evidence. This gives reviewers a clearer starting point when prioritizing cases.</p>
        </section>

        {/* DOCTOR REVIEW */}
        <DoctorReview casesData={casesData} reviews={reviews} onSaveReview={handleSaveReview} onOpenCase={openCase} />
      </main>

      <footer>
        <div><strong>HISTOPATHOLOGY</strong><span>TRIAGE-BASED</span></div>
        <span>ABMIL · MC-DROPOUT · MODEL DECISION SUPPORT</span>
        <span>© Cache Me Outside 2026</span>
      </footer>
    </div>
  );
}
