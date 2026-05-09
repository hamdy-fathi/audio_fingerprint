'use client';
import { FileText } from 'lucide-react';

export default function TranscriptionPanel({ segments, currentTime, isLoading }) {
  if (isLoading) {
    return (
      <div className="card">
        <div className="card-title"><FileText size={18} /> Live Transcription</div>
        <div className="loading-spinner"><div className="spinner" /><span className="loading-text">Transcribing with Whisper...</span></div>
      </div>
    );
  }

  if (!segments || segments.length === 0) {
    return (
      <div className="card">
        <div className="card-title"><FileText size={18} /> Live Transcription</div>
        <div className="empty-state">
          <div className="empty-state-icon"><FileText size={48} /></div>
          <p className="empty-state-text">Transcription will appear here</p>
          <p className="empty-state-subtext">Select a file and click Transcribe</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title"><FileText size={18} /> Live Transcription</span>
        <span className="card-badge badge-primary">{segments.length} segments</span>
      </div>
      <div className="transcription-panel" id="transcription-display">
        {segments.map((seg, i) => {
          let cls = 'upcoming';
          if (currentTime >= seg.end) cls = 'spoken';
          else if (currentTime >= seg.start) cls = 'current';
          return (
            <span key={i} className={`transcript-segment ${cls}`} title={`${seg.start}s - ${seg.end}s`}>
              {seg.text}{' '}
            </span>
          );
        })}
      </div>
      <div style={{ marginTop: 16, padding: 12, background: 'var(--bg)', borderRadius: 'var(--radius)' }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 4 }}>FULL TEXT</div>
        <div style={{ direction: 'rtl', fontFamily: 'Cairo, sans-serif', lineHeight: 1.8, color: 'var(--text-secondary)' }}>
          {segments.map((s) => s.text).join(' ')}
        </div>
      </div>
    </div>
  );
}
