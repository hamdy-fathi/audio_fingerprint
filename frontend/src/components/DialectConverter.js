'use client';
import { useState, useRef, useEffect } from 'react';
import { RefreshCw, Play } from 'lucide-react';
import { convertDialect } from '@/lib/api';

const DIALECTS = [
  { key: 'egyptian', label: 'Egyptian', ar: 'مصري' },
  { key: 'gulf', label: 'Gulf', ar: 'خليجي' },
  { key: 'levantine', label: 'Levantine', ar: 'شامي' },
  { key: 'maghrebi', label: 'Maghrebi', ar: 'مغربي' },
];

export default function DialectConverter({ fileId, sourceDialect }) {
  const [target, setTarget] = useState('');
  const [gender, setGender] = useState('male');
  const [pitch, setPitch] = useState(0);
  const [rate, setRate] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const audioRef = useRef(null);

  useEffect(() => {
    setResult(null);
  }, [fileId]);

  const handleConvert = async () => {
    if (!fileId || !target) return;
    
    const originalText = result?.original_text || null;
    setLoading(true);
    
    try {
      const pitchStr = pitch >= 0 ? `+${pitch}Hz` : `${pitch}Hz`;
      const rateStr = rate >= 0 ? `+${rate}%` : `${rate}%`;
      const data = await convertDialect(fileId, target, sourceDialect, gender, pitchStr, rateStr, originalText);
      setResult(data);
    } catch (e) {
      alert('Conversion failed: ' + e.message);
    }
    setLoading(false);
  };

  const playConverted = () => {
    if (!result?.audio_base64 || !audioRef.current) return;
    audioRef.current.src = `data:audio/mp3;base64,${result.audio_base64}`;
    audioRef.current.play();
  };

  if (!fileId) {
    return (
      <div className="card">
        <div className="card-title"><RefreshCw size={18} /> Dialect Converter</div>
        <div className="empty-state">
          <div className="empty-state-icon"><RefreshCw size={48} /></div>
          <p className="empty-state-text">Select a file first to convert its dialect</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title"><RefreshCw size={18} /> Dialect Converter</span>
      </div>

      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: 8 }}>TARGET DIALECT</div>
        <div className="dialect-selector">
          {DIALECTS.map((d) => (
            <button
              key={d.key}
              className={`dialect-option ${target === d.key ? 'selected' : ''}`}
              onClick={() => setTarget(d.key)}
              id={`convert-to-${d.key}`}
            >
              {d.label} ({d.ar})
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: 6 }}>VOICE</div>
          <select className="select" value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: 6 }}>PITCH ({pitch > 0 ? '+' : ''}{pitch}Hz)</div>
          <input type="range" min="-50" max="50" value={pitch} onChange={(e) => setPitch(parseInt(e.target.value))} style={{ width: '100%' }} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: 6 }}>RATE ({rate > 0 ? '+' : ''}{rate}%)</div>
          <input type="range" min="-50" max="50" value={rate} onChange={(e) => setRate(parseInt(e.target.value))} style={{ width: '100%' }} />
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button className="btn btn-primary" onClick={handleConvert} disabled={!target || loading} id="convert-btn">
            <RefreshCw size={16} /> {loading ? 'Converting...' : 'Convert'}
          </button>
        </div>
      </div>

      {result && (
        <>
          <div className="text-comparison">
            <div>
              <div className="text-box-label">Original ({sourceDialect || 'detected'})</div>
              <div className="text-box">{result.original_text}</div>
            </div>
            <div>
              <div className="text-box-label">Converted ({result.target_dialect})</div>
              <div className="text-box" style={{ borderLeft: '3px solid var(--primary)' }}>{result.converted_text}</div>
            </div>
          </div>
          <div style={{ marginTop: 16 }}>
            <audio ref={audioRef} />
            <button className="btn btn-accent" onClick={playConverted} id="play-converted-btn">
              <Play size={16} /> Play Converted Audio
            </button>
            <span style={{ marginLeft: 12, fontSize: '0.8rem', color: 'var(--text-dim)' }}>
              Voice: {result.voice_used}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
