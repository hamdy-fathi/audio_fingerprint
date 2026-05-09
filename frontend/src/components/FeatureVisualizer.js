'use client';
import { Microscope } from 'lucide-react';

export default function FeatureVisualizer({ featureData }) {
  if (!featureData) {
    return (
      <div className="card">
        <div className="card-title"><Microscope size={18} /> Feature Analysis</div>
        <div className="empty-state">
          <div className="empty-state-icon"><Microscope size={48} /></div>
          <p className="empty-state-text">Analyze a file to see feature comparisons</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title"><Microscope size={18} /> Feature Analysis</span>
        <span className="card-badge badge-secondary">Distinguishing Features</span>
      </div>

      {featureData.mfcc_chart && (
        <div>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
            MFCC Profile Comparison
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 8 }}>
            MFCCs capture vocal tract shape differences between dialects. Each bar shows how the uploaded file compares to dialect averages.
          </p>
          <div className="chart-container">
            <img src={`data:image/png;base64,${featureData.mfcc_chart}`} alt="MFCC Comparison" />
          </div>
        </div>
      )}

      {featureData.radar_chart && (
        <div>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
            Spectral Feature Radar
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 8 }}>
            Each axis represents a spectral property. Dialect polygons with different shapes indicate distinct acoustic fingerprints.
          </p>
          <div className="chart-container">
            <img src={`data:image/png;base64,${featureData.radar_chart}`} alt="Spectral Radar" />
          </div>
        </div>
      )}

      {featureData.pitch_chart && (
        <div>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
            Pitch (F0) Contour
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 8 }}>
            Intonation patterns vary significantly across dialects. Shaded regions show typical pitch ranges per dialect.
          </p>
          <div className="chart-container">
            <img src={`data:image/png;base64,${featureData.pitch_chart}`} alt="Pitch Contour" />
          </div>
        </div>
      )}
    </div>
  );
}
