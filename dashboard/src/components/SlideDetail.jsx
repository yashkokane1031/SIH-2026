import { useMemo, useState } from 'react';
import './SlideDetail.css';

/* ===================================================================
   Triage tier explanation — human-readable routing reason
   =================================================================== */
const TIER_EXPLANATION = {
  urgent:    'Routed to Tier 1: malignancy probability above threshold with low model uncertainty.',
  uncertain: 'Routed to Tier 3: model uncertainty above threshold — requires expert pathologist review regardless of probability.',
  routine:   'Routed to Tier 2: malignancy probability below threshold with low model uncertainty.',
};

/**
 * Helper to compute statistical percentile of an array
 */
function getPercentile(arr, p) {
  if (!arr || arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (idx - lower);
}

/**
 * Perceptually rich, high-contrast colormap (Magma/Plasma inspired).
 * Applies non-linear gamma scaling (power 0.45) to expand low-to-mid attention contrast.
 * Stop sequence: Dark Slate → Deep Indigo → Vivid Purple → Golden Amber → Bright Red → White-Hot Yellow
 */
function heatmapColor(value) {
  const raw = Math.max(0, Math.min(1, value));
  // Power transform (gamma ~ 0.45) to expand discrimination in low-mid attention range (0.01 - 0.30)
  const clamp = Math.pow(raw, 0.45);

  const stops = [
    { pos: 0.0,  r: 15,  g: 23,  b: 42  },  // #0F172A (dark slate navy)
    { pos: 0.20, r: 49,  g: 46,  b: 129 },  // #312E81 (deep indigo)
    { pos: 0.45, r: 168, g: 85,  b: 247 },  // #A855F7 (vivid purple/violet)
    { pos: 0.70, r: 245, g: 158, b: 11  },  // #F59E0B (golden amber)
    { pos: 0.90, r: 239, g: 68,  b: 68  },  // #EF4444 (bright red)
    { pos: 1.0,  r: 254, g: 240, b: 138 },  // #FEF08A (white-hot yellow core)
  ];

  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (clamp >= stops[i].pos && clamp <= stops[i + 1].pos) {
      lo = stops[i];
      hi = stops[i + 1];
      break;
    }
  }

  const t = hi.pos === lo.pos ? 0 : (clamp - lo.pos) / (hi.pos - lo.pos);
  const r = Math.round(lo.r + t * (hi.r - lo.r));
  const g = Math.round(lo.g + t * (hi.g - lo.g));
  const b = Math.round(lo.b + t * (hi.b - lo.b));

  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * SlideDetail — Full detail view for a single slide.
 * Shows metadata, attention heatmap grid, routing explanation,
 * and confirm/override action buttons.
 */
export default function SlideDetail({ slide, metadata, action, onAction, onBack }) {
  const [hoveredPatch, setHoveredPatch] = useState(null);

  // Build the normalized heatmap data with slide-specific density-aware bounding box scaling
  const heatmapData = useMemo(() => {
    const coords = slide.patch_coordinates || [];
    const weights = slide.patch_attention_weights || [];
    if (coords.length === 0) return null;

    const xs = coords.map(([x]) => x);
    const ys = coords.map(([, y]) => y);

    const fullMinX = Math.min(...xs);
    const fullMaxX = Math.max(...xs);
    const fullMinY = Math.min(...ys);
    const fullMaxY = Math.max(...ys);

    const spanX = Math.max(1, fullMaxX - fullMinX);
    const spanY = Math.max(1, fullMaxY - fullMinY);

    const isCompactGrid = spanX <= 25 && spanY <= 25;

    let cropMinX = fullMinX;
    let cropMaxX = fullMaxX;
    let cropMinY = fullMinY;
    let cropMaxY = fullMaxY;

    if (!isCompactGrid) {
      // Density-aware crop: use 2nd and 98th percentiles to tighten bounding box around the main tissue mass
      const p02X = getPercentile(xs, 2);
      const p98X = getPercentile(xs, 98);
      const p02Y = getPercentile(ys, 2);
      const p98Y = getPercentile(ys, 98);

      const coreSpanX = Math.max(1, p98X - p02X);
      const coreSpanY = Math.max(1, p98Y - p02Y);

      // Add a subtle 4% buffer around the core cluster so peripheral boundary patches are clear
      cropMinX = p02X - coreSpanX * 0.04;
      cropMaxX = p98X + coreSpanX * 0.04;
      cropMinY = p02Y - coreSpanY * 0.04;
      cropMaxY = p98Y + coreSpanY * 0.04;
    }

    const cropSpanX = Math.max(1, cropMaxX - cropMinX);
    const cropSpanY = Math.max(1, cropMaxY - cropMinY);

    const canvasWidth = 600;
    const canvasHeight = 600;
    const padding = 24;
    const availW = canvasWidth - padding * 2;
    const availH = canvasHeight - padding * 2;

    const scale = Math.min(availW / cropSpanX, availH / cropSpanY);
    const offsetX = padding + (availW - cropSpanX * scale) / 2;
    const offsetY = padding + (availH - cropSpanY * scale) / 2;

    // For compact synthetic grid, tile size matches grid step
    // For real WSI with wider coordinates, size is sized to fill the tissue mass prominently
    const patchSize = isCompactGrid
      ? Math.max(16, scale * 0.92)
      : Math.max(16, Math.min(26, Math.sqrt((availW * availH) / (coords.length * 1.5))));

    const patches = coords.map(([x, y], i) => {
      const px = offsetX + (x - cropMinX) * scale;
      const py = offsetY + (y - cropMinY) * scale;
      const weight = weights[i] ?? 0;
      return {
        index: i,
        origX: x,
        origY: y,
        px,
        py,
        weight,
        color: heatmapColor(weight),
      };
    });

    return {
      canvasWidth,
      canvasHeight,
      patchSize,
      patches,
    };
  }, [slide]);

  const tierClass = slide.tier;
  const prob = (slide.malignancy_probability * 100).toFixed(2);
  const uncert = slide.uncertainty_score.toFixed(4);

  return (
    <div className="slide-detail">
      {/* --- Slide header --- */}
      <div className="slide-detail__header">
        <div className="slide-detail__id-row">
          <h2 className="slide-detail__slide-id">{slide.slide_id}</h2>
          <span className={`tier-badge tier-badge--${tierClass}`}>
            {tierClass === 'urgent' ? 'Urgent' : tierClass === 'uncertain' ? 'Flagged' : 'Routine'}
          </span>
        </div>
        <p className="slide-detail__explanation">
          {TIER_EXPLANATION[slide.tier]}
        </p>
      </div>

      {/* --- Metrics row --- */}
      <div className="slide-detail__metrics">
        <div className="metric-card">
          <span className="metric-card__label">Malignancy Probability</span>
          <span className={`metric-card__value metric-card__value--${
            slide.malignancy_probability > 0.7 ? 'high' :
            slide.malignancy_probability > 0.3 ? 'mid' : 'low'
          }`}>
            {prob}%
          </span>
          <div className="metric-card__bar">
            <div
              className={`metric-card__bar-fill metric-card__bar-fill--${
                slide.malignancy_probability > 0.7 ? 'high' :
                slide.malignancy_probability > 0.3 ? 'mid' : 'low'
              }`}
              style={{ width: `${Math.min(100, slide.malignancy_probability * 100)}%` }}
            />
          </div>
        </div>
        <div className="metric-card">
          <span className="metric-card__label">Uncertainty Score</span>
          <span className="metric-card__value">{uncert}</span>
          <div className="metric-card__bar">
            <div
              className={`metric-card__bar-fill metric-card__bar-fill--${
                slide.uncertainty_score >= (metadata?.triage_thresholds?.uncertain_std ?? 0.15)
                  ? 'high' : 'low'
              }`}
              style={{ width: `${Math.min(100, slide.uncertainty_score * 400)}%` }}
            />
          </div>
        </div>
        <div className="metric-card">
          <span className="metric-card__label">Patches Analyzed</span>
          <span className="metric-card__value">{slide.num_patches}</span>
        </div>
        {slide.true_label !== undefined && (
          <div className="metric-card">
            <span className="metric-card__label">Ground Truth</span>
            <span className={`metric-card__value metric-card__value--${
              slide.true_label === 1 ? 'high' : 'low'
            }`}>
              {slide.true_label === 1 ? 'Malignant' : 'Benign'}
            </span>
          </div>
        )}
      </div>

      {/* --- Heatmap --- */}
      <div className="slide-detail__heatmap-section">
        <div className="slide-detail__section-header">
          <h3 className="slide-detail__section-title">Attention Heatmap</h3>
          <span className="slide-detail__section-sub">
            Patch-level attention weights normalized across tissue coordinates — brighter regions indicate high attention
          </span>
        </div>

        <div className="heatmap-container">
          {heatmapData && (
            <svg
              className="heatmap-svg"
              viewBox={`0 0 ${heatmapData.canvasWidth} ${heatmapData.canvasHeight}`}
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Background */}
              <rect
                x="0"
                y="0"
                width={heatmapData.canvasWidth}
                height={heatmapData.canvasHeight}
                fill="#0D1117"
                rx="4"
              />

              {/* Individual patch tiles */}
              {heatmapData.patches.map((p) => {
                const isHovered = hoveredPatch === p.index;
                const half = heatmapData.patchSize / 2;
                return (
                  <rect
                    key={p.index}
                    className="heatmap-patch"
                    x={p.px - half}
                    y={p.py - half}
                    width={heatmapData.patchSize}
                    height={heatmapData.patchSize}
                    fill={p.color}
                    rx="2"
                    stroke={isHovered ? '#FFFFFF' : 'rgba(0, 0, 0, 0.4)'}
                    strokeWidth={isHovered ? 2.5 : 0.75}
                    onMouseEnter={() => setHoveredPatch(p.index)}
                    onMouseLeave={() => setHoveredPatch(null)}
                  >
                    <title>
                      {`Patch #${p.index} (${p.origX}, ${p.origY})\nAttention: ${p.weight.toFixed(4)}`}
                    </title>
                  </rect>
                );
              })}
            </svg>
          )}

          {/* Colormap legend */}
          <div className="heatmap-legend">
            <span className="heatmap-legend__label">Low</span>
            <div className="heatmap-legend__bar" />
            <span className="heatmap-legend__label">High</span>
          </div>

          {/* Hover info */}
          {hoveredPatch !== null && (
            <div className="heatmap-hover-info">
              <span className="heatmap-hover-info__label">Patch #{hoveredPatch}</span>
              <span className="heatmap-hover-info__value">
                Attention: {slide.patch_attention_weights[hoveredPatch]?.toFixed(4)}
              </span>
              <span className="heatmap-hover-info__coord">
                Coords: ({slide.patch_coordinates[hoveredPatch]?.[0]}, {slide.patch_coordinates[hoveredPatch]?.[1]})
              </span>
            </div>
          )}
        </div>
      </div>

      {/* --- Action buttons --- */}
      <div className="slide-detail__actions">
        <div className="slide-detail__section-header">
          <h3 className="slide-detail__section-title">Pathologist Review</h3>
        </div>
        <div className="slide-detail__action-row">
          {action ? (
            <div className={`action-result action-result--${action}`}>
              {action === 'confirmed'
                ? 'Triage classification confirmed by pathologist'
                : 'Triage classification overridden by pathologist'}
            </div>
          ) : (
            <>
              <button
                className="action-btn action-btn--confirm"
                onClick={() => onAction(slide.slide_id, 'confirmed')}
              >
                Confirm Classification
              </button>
              <button
                className="action-btn action-btn--override"
                onClick={() => onAction(slide.slide_id, 'overridden')}
              >
                Override Classification
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
