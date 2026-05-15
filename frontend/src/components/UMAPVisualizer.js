'use client';
import { ScatterChart } from 'lucide-react';

const DIALECT_COLORS = {
  'Egyptian Arabic':  '#7C3AED',
  'Gulf Arabic':      '#C75B39',
  'Levantine Arabic': '#7A9E7E',
  'Maghrebi Arabic':  '#E8A838',
};

export default function UMAPVisualizer({ umapData }) {

  if (!umapData) {
    return (
      <div className="card" id="umap-visualizer">
        <div className="card-header">
          <span className="card-title">
            <ScatterChart size={18} /> UMAP Dialect Space
          </span>
          <span className="card-badge badge-secondary">2D Projection</span>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon"><ScatterChart size={48} /></div>
          <p className="empty-state-text">
            Analyze a file to see where it falls in the dialect space
          </p>
        </div>
      </div>
    );
  }

  const dialectColor = DIALECT_COLORS[umapData.predicted_dialect] ?? '#888';

  return (
    <div className="card" id="umap-visualizer">

      {/* ── Header ── */}
      <div className="card-header">
        <span className="card-title">
          <ScatterChart size={18} /> UMAP Dialect Space
        </span>
        <span className="card-badge badge-secondary">2D Projection</span>
      </div>

      {/* ── Dialect badge ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        background: 'var(--surface)', borderRadius: 8,
        padding: '10px 14px', marginBottom: 14,
        border: `1px solid ${dialectColor}44`,
      }}>
        <span style={{
          width: 14, height: 14, borderRadius: '50%',
          background: dialectColor, flexShrink: 0,
          boxShadow: `0 0 8px ${dialectColor}66`,
        }} />
        <div>
          <p style={{ margin: 0, fontSize: '0.72rem', color: 'var(--text-dim)' }}>
            Predicted dialect
          </p>
          <p style={{
            margin: 0, fontWeight: 700, fontSize: '1rem',
            color: dialectColor,
          }}>
            {umapData.predicted_dialect}
          </p>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <p style={{ margin: 0, fontSize: '0.72rem', color: 'var(--text-dim)' }}>
            UMAP coords
          </p>
          <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
            ({umapData.umap_coords[0].toFixed(2)}, {umapData.umap_coords[1].toFixed(2)})
          </p>
        </div>
      </div>

      {/* ── Scatter plot ── */}
      <div>
        <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
          Dialect Space Projection
        </h3>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 8 }}>
          Each dot is a training sample. The ★ shows where your file lands in the
          learned 2D dialect space.
        </p>
        <div className="chart-container" style={{ borderRadius: 8, overflow: 'hidden' }}>
          <img
            src={`data:image/png;base64,${umapData.umap_chart}`}
            alt="UMAP projection"
            style={{ width: '100%', display: 'block', borderRadius: 8 }}
          />
        </div>
      </div>

    </div>
  );
}
