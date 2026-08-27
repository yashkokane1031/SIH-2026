import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import "./PathoIntro.css";

/**
 * PathoIntro — Cinematic Clinical Decision Support Intro Screen
 * 
 * Features:
 * - Scanning laser sweep & trailing tissue grid
 * - Web Audio API procedural sound engine (no external audio files)
 * - Staggered typography and glowing pulse-bar logo
 * - Responsive, keyboard-accessible (Enter / Space / Esc / M)
 * - Seamless handoff animation into clinical workspace
 */
export default function PathoIntro({ onEnter }) {
  const [muted, setMuted] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const audioCtxRef = useRef(null);
  const introTimersRef = useRef([]);

  // Generate simulated tissue cluster nodes for the laser to illuminate
  const tissueNodes = useMemo(() => {
    const nodes = [];
    const count = 36;
    for (let i = 0; i < count; i++) {
      nodes.push({
        id: i,
        top: `${15 + Math.random() * 70}%`,
        left: `${10 + Math.random() * 80}%`,
        delay: `${0.4 + Math.random() * 0.9}s`,
      });
    }
    return nodes;
  }, []);

  // Web Audio Context Getter
  const getAudioContext = useCallback(() => {
    if (!audioCtxRef.current) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) {
        audioCtxRef.current = new AC();
      }
    }
    if (audioCtxRef.current && audioCtxRef.current.state === "suspended") {
      audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  }, []);

  // Procedural Tone Synthesizer
  const playTone = useCallback(
    ({ freq = 440, delay = 0, dur = 0.3, type = "sine", peak = 0.14, glideTo = null }) => {
      if (muted) return;
      const ctx = getAudioContext();
      if (!ctx) return;
      const t0 = ctx.currentTime + delay;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, t0);
      if (glideTo) {
        osc.frequency.exponentialRampToValueAtTime(glideTo, t0 + dur);
      }

      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.linearRampToValueAtTime(peak, t0 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(t0);
      osc.stop(t0 + dur + 0.05);
    },
    [muted, getAudioContext]
  );

  // Laser Scanner Sweep Sound
  const playSweepSound = useCallback(
    (delay = 0) => {
      if (muted) return;
      const ctx = getAudioContext();
      if (!ctx) return;
      const t0 = ctx.currentTime + delay;
      const dur = 1.15;
      const bufferSize = Math.floor(ctx.sampleRate * dur);
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);

      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * 0.35;
      }

      const noise = ctx.createBufferSource();
      noise.buffer = buffer;

      const filter = ctx.createBiquadFilter();
      filter.type = "bandpass";
      filter.Q.value = 5.5;
      filter.frequency.setValueAtTime(450, t0);
      filter.frequency.exponentialRampToValueAtTime(2400, t0 + dur);

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.linearRampToValueAtTime(0.12, t0 + 0.15);
      gain.gain.linearRampToValueAtTime(0.08, t0 + dur * 0.7);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);

      noise.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);

      noise.start(t0);
      noise.stop(t0 + dur + 0.05);

      playTone({ freq: 180, delay, dur, type: "sine", peak: 0.06, glideTo: 600 });
    },
    [muted, getAudioContext, playTone]
  );

  // Tactile Click Feedback
  const playClickSound = useCallback(() => {
    playTone({ freq: 720, delay: 0, dur: 0.1, type: "triangle", peak: 0.16 });
    playTone({ freq: 1300, delay: 0.03, dur: 0.28, type: "sine", peak: 0.12, glideTo: 2200 });
  }, [playTone]);

  // Schedule cinematic audio cue sequence
  useEffect(() => {
    introTimersRef.current.push(setTimeout(() => playSweepSound(0), 300));
    introTimersRef.current.push(setTimeout(() => playTone({ freq: 523.25, dur: 0.35, peak: 0.12 }), 1100)); // Bar 1 — C5
    introTimersRef.current.push(setTimeout(() => playTone({ freq: 659.25, dur: 0.35, peak: 0.12 }), 1220)); // Bar 2 — E5
    introTimersRef.current.push(setTimeout(() => playTone({ freq: 783.99, dur: 0.4, peak: 0.12 }), 1340));  // Bar 3 — G5
    introTimersRef.current.push(setTimeout(() => playTone({ freq: 1046.5, dur: 0.55, peak: 0.09 }), 2450)); // CTA chord — C6

    return () => {
      introTimersRef.current.forEach(clearTimeout);
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close().catch(() => {});
      }
    };
  }, [playSweepSound, playTone]);

  // Enter Site Trigger
  const handleEnter = useCallback(() => {
    if (leaving) return;
    playClickSound();
    introTimersRef.current.forEach(clearTimeout);
    setLeaving(true);

    setTimeout(() => {
      if (onEnter) onEnter();
    }, 750);
  }, [leaving, playClickSound, onEnter]);

  // Keyboard Shortcuts (Enter/Space to Enter, M to Mute, Esc to Skip)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handleEnter();
      } else if (e.key === "m" || e.key === "M") {
        setMuted((prev) => !prev);
      } else if (e.key === "Escape") {
        if (onEnter) onEnter();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleEnter, onEnter]);

  return (
    <div className={`patho-intro-stage ${leaving ? "leaving" : ""}`} id="stage">
      {/* Whole-Slide Grid Backdrop */}
      <div className="intro-grid" />
      <div className="intro-sweep-trail" />
      <div className="intro-sweep" />

      {/* Simulated Cytological Nodes */}
      <div className="intro-node-field">
        {tissueNodes.map((node) => (
          <div
            key={node.id}
            className="intro-node"
            style={{
              top: node.top,
              left: node.left,
              animationDelay: node.delay,
            }}
          />
        ))}
      </div>

      {/* Audio Mute Toggle */}
      <button
        className="intro-sound-toggle"
        onClick={() => {
          setMuted((prev) => !prev);
          getAudioContext();
        }}
        aria-label={muted ? "Unmute audio" : "Mute audio"}
        title={muted ? "Unmute sound (M)" : "Mute sound (M)"}
      >
        {muted ? (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M11 5 6 9H2v6h4l5 4V5z" />
            <path d="m23 9-6 6" />
            <path d="m17 9 6 6" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M11 5 6 9H2v6h4l5 4V5z" />
            <path d="M15.5 8.5a5 5 0 0 1 0 7" />
            <path d="M18.5 5.5a9 9 0 0 1 0 13" />
          </svg>
        )}
      </button>

      {/* Central Logo Cluster */}
      <div className="intro-logo-wrap">
        <div className="intro-eyebrow">Clinical Decision Support · ABMIL</div>

        <div className="intro-mark-row">
          <div className="intro-icon">
            <div className="intro-bar" style={{ "--h": "16px" }} />
            <div className="intro-bar" style={{ "--h": "30px" }} />
            <div className="intro-bar" style={{ "--h": "22px" }} />
          </div>

          <div className="intro-wordmark">
            <div className="intro-patho">
              <span>P</span>
              <span>A</span>
              <span>T</span>
              <span>H</span>
              <span>O</span>
            </div>
            <div className="intro-intelligence">Intelligence</div>
          </div>
        </div>

        {/* CTA Launch */}
        <div className="intro-cta-wrapper">
          <button className="intro-cta-btn" onClick={handleEnter}>
            <span>Get Started</span>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M13 6l6 6-6 6" />
            </svg>
          </button>
          <span className="intro-hint">Press [Space] or [Enter] to begin</span>
        </div>
      </div>
    </div>
  );
}
