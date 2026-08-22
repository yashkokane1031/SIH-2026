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
  const [dataset, setDataset] = useState('camelyon16'); // 'camelyon16' | 'synthetic'
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSlideId, setSelectedSlideId] = useState(null);
  const [actions, setActions] = useState({}); // { slide_id: 'confirmed' | 'overridden' }

  const loadDataset = useCallback((targetDataset) => {
    setLoading(true);
    setError(null);
    setSelectedSlideId(null);

    const primaryFile = targetDataset === 'camelyon16'
      ? '/triage_results_camelyon16.json'
      : '/triage_results_synthetic.json';

    fetch(primaryFile)
      .then((res) => {
        if (!res.ok) {
          // Fallback to /triage_results.json
          return fetch('/triage_results.json');
        }
        return res;
      })
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

  useEffect(() => {
    loadDataset(dataset);
  }, [dataset, loadDataset]);

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
          <p className="app-state__label">Loading {dataset === 'camelyon16' ? 'CAMELYON16' : 'Synthetic'} triage data...</p>
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
            Ensure triage JSON files are present in the public/ directory.
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
            Run the pipeline to generate triage results.
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

        <div className="app-header__center" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <div style={{
            display: 'inline-flex',
            background: 'rgba(255, 255, 255, 0.08)',
            padding: '3px',
            borderRadius: '8px',
            border: '1px solid rgba(255, 255, 255, 0.12)'
          }}>
            <button
              style={{
                background: dataset === 'camelyon16' ? '#2563eb' : 'transparent',
                color: dataset === 'camelyon16' ? '#ffffff' : '#94a3b8',
                border: 'none',
                padding: '5px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onClick={() => setDataset('camelyon16')}
            >
              Real CAMELYON16
            </button>
            <button
              style={{
                background: dataset === 'synthetic' ? '#2563eb' : 'transparent',
                color: dataset === 'synthetic' ? '#ffffff' : '#94a3b8',
                border: 'none',
                padding: '5px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onClick={() => setDataset('synthetic')}
            >
              Synthetic Benchmark
            </button>
          </div>
        </div>

        <div className="app-header__right">
          <span className="app-header__meta">
            {data.metadata?.model?.includes('2048') ? '2048-d ResNet' : 'ABMIL + MC-Dropout'}
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
