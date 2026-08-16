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
 * Interpolate a color on the sequential heatmap colormap.
 * 0.0 = dark slate → 0.33 = blue → 0.66 = amber → 1.0 = red
 */
function heatmapColor(value) {
  const clamp = Math.max(0, Math.min(1, value));

  // 4-stop colormap: slate → blue → amber → red
  const stops = [
    { pos: 0.0, r: 30,  g: 41,  b: 59  },   // #1E293B
    { pos: 0.33, r: 29,  g: 78,  b: 216 },   // #1D4ED8
    { pos: 0.66, r: 245, g: 158, b: 11  },   // #F59E0B
    { pos: 1.0, r: 220, g: 38,  b: 38  },    // #DC2626
  ];

  // Find the two stops we're between
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

  // Build the heatmap grid data
  const gridData = useMemo(() => {
    const coords = slide.patch_coordinates;
    const weights = slide.patch_attention_weights;

    // Find grid bounds
    let maxX = 0, maxY = 0;
    coords.forEach(([x, y]) => {
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    });

    // Build a sparse grid map
    const grid = {};
    coords.forEach(([x, y], i) => {
      grid[`${x},${y}`] = {
        x, y,
        weight: weights[i],
        index: i,
      };
    });

    return { maxX, maxY, grid, totalPatches: coords.length };
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
            Patch-level attention weights — brighter regions indicate areas the model focused on
          </span>
        </div>

        <div className="heatmap-container">
          <div
            className="heatmap-grid"
            style={{
              gridTemplateColumns: `repeat(${gridData.maxX + 1}, 1fr)`,
              gridTemplateRows: `repeat(${gridData.maxY + 1}, 1fr)`,
            }}
          >
            {Array.from({ length: (gridData.maxY + 1) }).map((_, y) =>
              Array.from({ length: (gridData.maxX + 1) }).map((_, x) => {
                const cell = gridData.grid[`${x},${y}`];
                const isEmpty = !cell;
                const weight = cell ? cell.weight : 0;
                const isHovered = hoveredPatch !== null && cell && cell.index === hoveredPatch;

                return (
                  <div
                    key={`${x}-${y}`}
                    className={`heatmap-cell ${isEmpty ? 'heatmap-cell--empty' : ''} ${isHovered ? 'heatmap-cell--hovered' : ''}`}
                    style={!isEmpty ? {
                      backgroundColor: heatmapColor(weight),
                    } : undefined}
                    onMouseEnter={() => cell && setHoveredPatch(cell.index)}
                    onMouseLeave={() => setHoveredPatch(null)}
                    title={cell ? `Patch (${x}, ${y})\nAttention: ${weight.toFixed(4)}` : ''}
                  />
                );
              })
            )}
          </div>

          {/* Colormap legend */}
          <div className="heatmap-legend">
            <span className="heatmap-legend__label">Low</span>
            <div className="heatmap-legend__bar" />
            <span className="heatmap-legend__label">High</span>
          </div>

          {/* Hover info */}
          {hoveredPatch !== null && (
            <div className="heatmap-hover-info">
              <span className="heatmap-hover-info__label">Patch {hoveredPatch}</span>
              <span className="heatmap-hover-info__value">
                Attention: {slide.patch_attention_weights[hoveredPatch]?.toFixed(4)}
              </span>
              <span className="heatmap-hover-info__coord">
                ({slide.patch_coordinates[hoveredPatch]?.[0]}, {slide.patch_coordinates[hoveredPatch]?.[1]})
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
