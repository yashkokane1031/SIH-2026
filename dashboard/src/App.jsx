import { useState, useEffect, useCallback } from 'react';
import './App.css';
import TriageQueue from './components/TriageQueue';
import SlideDetail from './components/SlideDetail';

/**
 * App — Root component managing navigation between
 * the Triage Queue (list) and Slide Detail (heatmap) views.
 * Loads triage_results.json from /public on mount.
 */
export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSlideId, setSelectedSlideId] = useState(null);
  const [actions, setActions] = useState({}); // { slide_id: 'confirmed' | 'overridden' }

  useEffect(() => {
    fetch('/triage_results.json')
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load data: ${res.status}`);
        return res.json();
      })
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const handleSelectSlide = useCallback((slideId) => {
    setSelectedSlideId(slideId);
  }, []);

  const handleBack = useCallback(() => {
    setSelectedSlideId(null);
  }, []);

  const handleAction = useCallback((slideId, action) => {
    setActions((prev) => ({ ...prev, [slideId]: action }));
    console.log(`[AUDIT] Slide ${slideId}: pathologist action = ${action} at ${new Date().toISOString()}`);
  }, []);

  // --- Loading state ---
  if (loading) {
    return (
      <div className="app-state">
        <div className="app-state__content">
          <div className="app-state__spinner" />
          <p className="app-state__label">Loading triage data...</p>
        </div>
      </div>
    );
  }

  // --- Error state ---
  if (error) {
    return (
      <div className="app-state">
        <div className="app-state__content">
          <p className="app-state__label app-state__label--error">
            Error: {error}
          </p>
          <p className="app-state__sublabel">
            Ensure triage_results.json is in the public/ directory.
          </p>
        </div>
      </div>
    );
  }

  // --- Empty state ---
  if (!data || !data.slides || data.slides.length === 0) {
    return (
      <div className="app-state">
        <div className="app-state__content">
          <p className="app-state__label">No slides in triage queue</p>
          <p className="app-state__sublabel">
            Run the pipeline to generate triage_results.json
          </p>
        </div>
      </div>
    );
  }

  const selectedSlide = selectedSlideId
    ? data.slides.find((s) => s.slide_id === selectedSlideId)
    : null;

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header__left">
          {selectedSlide && (
            <button
              className="app-header__back"
              onClick={handleBack}
              aria-label="Back to triage queue"
            >
              &larr;
            </button>
          )}
          <h1 className="app-header__title">
            Histopathology Triage
          </h1>
        </div>
        <div className="app-header__right">
          <span className="app-header__meta">
            ABMIL + MC-Dropout
          </span>
          <span className="app-header__separator">|</span>
          <span className="app-header__meta">
            {data.slides.length} slides
          </span>
        </div>
      </header>

      <main className="app-main">
        {selectedSlide ? (
          <SlideDetail
            slide={selectedSlide}
            metadata={data.metadata}
            action={actions[selectedSlideId]}
            onAction={handleAction}
            onBack={handleBack}
          />
        ) : (
          <TriageQueue
            slides={data.slides}
            actions={actions}
            onSelectSlide={handleSelectSlide}
          />
        )}
      </main>
    </div>
  );
}
