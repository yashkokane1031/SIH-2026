import { useMemo } from 'react';
import './TriageQueue.css';

/* ===================================================================
   Tier configuration — single source of truth for ordering,
   labels, and CSS class names.
   =================================================================== */
const TIER_CONFIG = {
  urgent:    { order: 0, label: 'Urgent',    statusLabel: 'Requires immediate review' },
  uncertain: { order: 1, label: 'Flagged',   statusLabel: 'Model uncertainty — needs expert review' },
  routine:   { order: 2, label: 'Routine',   statusLabel: 'Likely benign — standard queue' },
};

/**
 * TriageQueue — Main triage table view.
 * Sorts slides by tier priority (urgent first), shows summary counts,
 * and provides row-click navigation to slide detail.
 */
export default function TriageQueue({ slides, actions, onSelectSlide }) {
  // Sort slides: tier priority first, then by descending probability
  const sorted = useMemo(() => {
    return [...slides].sort((a, b) => {
      const tierDiff = (TIER_CONFIG[a.tier]?.order ?? 9) - (TIER_CONFIG[b.tier]?.order ?? 9);
      if (tierDiff !== 0) return tierDiff;
      return b.malignancy_probability - a.malignancy_probability;
    });
  }, [slides]);

  // Tier counts
  const counts = useMemo(() => {
    const c = { urgent: 0, routine: 0, uncertain: 0 };
    slides.forEach((s) => { c[s.tier] = (c[s.tier] || 0) + 1; });
    return c;
  }, [slides]);

  return (
    <div className="triage-queue">
      {/* --- Summary bar --- */}
      <div className="triage-queue__summary">
        <div className="summary-card summary-card--urgent">
          <span className="summary-card__count">{counts.urgent}</span>
          <span className="summary-card__label">Urgent</span>
          <span className="summary-card__tier">Tier 1</span>
        </div>
        <div className="summary-card summary-card--uncertain">
          <span className="summary-card__count">{counts.uncertain}</span>
          <span className="summary-card__label">Flagged</span>
          <span className="summary-card__tier">Tier 3</span>
        </div>
        <div className="summary-card summary-card--routine">
          <span className="summary-card__count">{counts.routine}</span>
          <span className="summary-card__label">Routine</span>
          <span className="summary-card__tier">Tier 2</span>
        </div>
      </div>

      {/* --- Table --- */}
      <div className="triage-queue__table-wrapper">
        <table className="triage-table">
          <thead>
            <tr>
              <th className="triage-table__th">Slide ID</th>
              <th className="triage-table__th">Tier</th>
              <th className="triage-table__th triage-table__th--num">Malig. Prob.</th>
              <th className="triage-table__th triage-table__th--num">Uncertainty</th>
              <th className="triage-table__th">Status</th>
              <th className="triage-table__th">Action</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((slide) => {
              const tierInfo = TIER_CONFIG[slide.tier] || TIER_CONFIG.routine;
              const action = actions[slide.slide_id];

              return (
                <tr
                  key={slide.slide_id}
                  className={`triage-table__row triage-table__row--${slide.tier}`}
                  onClick={() => onSelectSlide(slide.slide_id)}
                  tabIndex={0}
                  role="button"
                  aria-label={`View details for ${slide.slide_id}`}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelectSlide(slide.slide_id);
                    }
                  }}
                >
                  <td className="triage-table__td triage-table__td--id">
                    {slide.slide_id}
                  </td>
                  <td className="triage-table__td">
                    <span className={`tier-badge tier-badge--${slide.tier}`}>
                      {tierInfo.label}
                    </span>
                  </td>
                  <td className="triage-table__td triage-table__td--num">
                    {(slide.malignancy_probability * 100).toFixed(1)}%
                  </td>
                  <td className="triage-table__td triage-table__td--num">
                    {slide.uncertainty_score.toFixed(4)}
                  </td>
                  <td className="triage-table__td triage-table__td--status">
                    {tierInfo.statusLabel}
                  </td>
                  <td className="triage-table__td">
                    {action ? (
                      <span className={`action-tag action-tag--${action}`}>
                        {action === 'confirmed' ? 'Confirmed' : 'Overridden'}
                      </span>
                    ) : (
                      <span className="action-tag action-tag--pending">Pending</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
