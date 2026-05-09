'use client';
import { useState } from 'react';
import { BarChart3, AudioWaveform, Activity } from 'lucide-react';

export default function SpectrogramViewer({ spectrogramData }) {
  const [view, setView] = useState('spectrogram');

  if (!spectrogramData) {
    return (
      <div className="card">
        <div className="card-title"><BarChart3 size={18} /> Spectrogram</div>
        <div className="empty-state">
          <div className="empty-state-icon"><BarChart3 size={48} /></div>
          <p className="empty-state-text">Select a file to view its spectrogram</p>
        </div>
      </div>
    );
  }

  const images = {
    spectrogram: spectrogramData.spectrogram,
    mel_spectrogram: spectrogramData.mel_spectrogram,
    waveform: spectrogramData.waveform,
  };

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title"><BarChart3 size={18} /> Spectrogram</span>
        <span className="card-badge badge-accent">{spectrogramData.duration}s | {spectrogramData.sample_rate} Hz</span>
      </div>
      <div className="spectrogram-toggle">
        {[
          ['spectrogram', 'Spectrogram'],
          ['mel_spectrogram', 'Mel Spectrogram'],
          ['waveform', 'Waveform'],
        ].map(([key, label]) => (
          <button key={key} className={view === key ? 'active' : ''} onClick={() => setView(key)} id={`spec-toggle-${key}`}>
            {label}
          </button>
        ))}
      </div>
      <div className="spectrogram-container">
        {images[view] && <img src={`data:image/png;base64,${images[view]}`} alt={view} />}
      </div>
    </div>
  );
}
