'use client';
import { useState } from 'react';
import { Microscope, ChevronDown, ChevronUp } from 'lucide-react';

const PLOT_INFO = [
  {
    key: 'feature_breakdown',
    title: '📊 Feature-by-Feature Dialect Match',
    desc: 'Breaks down the similarity into 5 acoustic categories — green cells (high %) mean a strong match, red cells (low %) mean a weak match. Each row shows a different acoustic property, so you can see exactly WHERE and WHY your audio matches a specific dialect.',
    highlight: true,
  },
  {
    key: 'pitch_contour',
    title: '🎵 Pitch (F0) Contour with Dialect Ranges',
    desc: 'Shows your file\'s pitch over time. Shaded bands show the typical pitch range (μ±σ) for each dialect. You can see which dialect\'s pitch range your audio falls into.',
  },
];

export default function FeatureVisualizer({ featureData }) {
  const [expanded, setExpanded] = useState(true);

  if (!featureData) {
    return (
      <div className="card">
        <div className="card-title"><Microscope size={18} /> Feature Analysis</div>
        <div className="empty-state">
          <div className="empty-state-icon"><Microscope size={48} /></div>
          <p className="empty-state-text">Analyze a file to see dialect comparison plots</p>
        </div>
      </div>
    );
  }

  const availablePlots = PLOT_INFO.filter(p => featureData[p.key]);

  if (availablePlots.length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <span className="card-title"><Microscope size={18} /> Feature Analysis</span>
        </div>
        <p style={{ color: 'var(--text-dim)', padding: 16 }}>No plots were generated.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header" style={{ cursor: 'pointer' }} onClick={() => setExpanded(!expanded)}>
        <span className="card-title"><Microscope size={18} /> Dialect Comparison Plots</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="card-badge badge-secondary">{availablePlots.length} plots</span>
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </div>

      {expanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24, marginTop: 12 }}>
          {availablePlots.map((plot) => (
            <div key={plot.key} style={{
              background: plot.highlight ? 'linear-gradient(135deg, rgba(124,58,237,0.08), rgba(199,91,57,0.08))' : 'var(--surface)',
              borderRadius: 10,
              padding: plot.highlight ? 20 : 16,
              border: plot.highlight ? '2px solid var(--primary)' : '1px solid var(--border)',
              boxShadow: plot.highlight ? '0 0 20px rgba(124,58,237,0.15)' : 'none',
            }}>
              <h3 style={{
                fontSize: plot.highlight ? '1.1rem' : '0.95rem',
                color: plot.highlight ? 'var(--primary)' : 'var(--text)',
                marginBottom: 6,
                fontWeight: 700,
              }}>
                {plot.title}
              </h3>
              <p style={{
                fontSize: '0.75rem',
                color: 'var(--text-dim)',
                marginBottom: 12,
                lineHeight: 1.5,
              }}>
                {plot.desc}
              </p>
              <div className="chart-container" style={{ borderRadius: 8, overflow: 'hidden' }}>
                <img
                  src={`data:image/png;base64,${featureData[plot.key]}`}
                  alt={plot.title}
                  style={{ width: '100%', display: 'block' }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
