'use client';
import { useState, useRef, useEffect } from 'react';
import { RefreshCw, Play, Pause, Volume2, ArrowRight, Mic, SlidersHorizontal, Pencil, RotateCcw, Sparkles, BookOpen } from 'lucide-react';
import { convertDialect, synthesizeText } from '@/lib/api';

const DIALECTS = [
  { key: 'egyptian', label: 'Egyptian', ar: 'مصري', color: 'var(--primary)' },
  { key: 'gulf', label: 'Gulf', ar: 'خليجي', color: 'var(--secondary)' },
  { key: 'levantine', label: 'Levantine', ar: 'شامي', color: 'var(--accent)' },
  { key: 'maghrebi', label: 'Maghrebi', ar: 'مغربي', color: '#E8A838' },
];

export default function DialectConverter({ fileId, sourceDialect }) {
  const [target, setTarget] = useState('');
  const [gender, setGender] = useState('male');
  const [pitch, setPitch] = useState(0);
  const [rate, setRate] = useState(0);
  const [mode, setMode] = useState('dictionary');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [editedText, setEditedText] = useState('');
  const [textEdited, setTextEdited] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const audioRef = useRef(null);

  useEffect(() => {
    setResult(null);
  }, [fileId]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onEnded = () => setPlaying(false);
    audio.addEventListener('ended', onEnded);
    return () => audio.removeEventListener('ended', onEnded);
  }, []);

  const handleConvert = async () => {
    if (!fileId || !target) return;

    const originalText = result?.original_text || null;
    setLoading(true);

    try {
      const pitchStr = pitch >= 0 ? `+${pitch}Hz` : `${pitch}Hz`;
      const rateStr = rate >= 0 ? `+${rate}%` : `${rate}%`;
      const data = await convertDialect(fileId, target, sourceDialect, gender, pitchStr, rateStr, originalText, mode);
      setResult(data);
      setEditedText(data.converted_text);
      setTextEdited(false);
    } catch (e) {
      alert('Conversion failed: ' + e.message);
    }
    setLoading(false);
  };

  const togglePlayback = () => {
    if (!result?.audio_base64 || !audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      audioRef.current.src = `data:audio/mp3;base64,${result.audio_base64}`;
      audioRef.current.play();
      setPlaying(true);
    }
  };

  const handleResynthesize = async () => {
    if (!editedText || !target) return;
    setSyncing(true);
    try {
      const pitchStr = pitch >= 0 ? `+${pitch}Hz` : `${pitch}Hz`;
      const rateStr = rate >= 0 ? `+${rate}%` : `${rate}%`;
      const data = await synthesizeText(editedText, target, gender, pitchStr, rateStr);
      setResult(prev => ({
        ...prev,
        converted_text: editedText,
        audio_base64: data.audio_base64,
        voice_used: data.voice_used,
      }));
      setTextEdited(false);
    } catch (e) {
      alert('Re-synthesis failed: ' + e.message);
    }
    setSyncing(false);
  };

  if (!fileId) {
    return (
      <div className="converter-panel">
        <div className="converter-empty">
          <div className="converter-empty-icon"><RefreshCw size={32} /></div>
          <p className="converter-empty-title">Dialect Converter</p>
          <p className="converter-empty-sub">Select an audio file to begin converting between dialects</p>
        </div>
      </div>
    );
  }

  const targetMeta = DIALECTS.find(d => d.key === target);

  return (
    <div className="converter-panel">
      <audio ref={audioRef} />

      {/* Section 1: Target Dialect */}
      <div className="converter-section">
        <div className="converter-section-label">
          <ArrowRight size={12} />
          <span>Target Dialect</span>
        </div>
        <div className="converter-dialect-grid">
          {DIALECTS.map((d) => (
            <button
              key={d.key}
              className={`converter-dialect-card ${target === d.key ? 'selected' : ''}`}
              onClick={() => setTarget(d.key)}
              id={`convert-to-${d.key}`}
              style={{ '--dialect-color': d.color }}
            >
              <span className="converter-dialect-dot" style={{ background: d.color }} />
              <span className="converter-dialect-label">{d.label}</span>
              <span className="converter-dialect-ar">{d.ar}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Section 2: Translation Mode */}
      <div className="converter-section">
        <div className="converter-section-label">
          <Sparkles size={12} />
          <span>Translation Mode</span>
        </div>
        <div className="converter-mode-toggle">
          <button
            className={`mode-btn ${mode === 'dictionary' ? 'active' : ''}`}
            onClick={() => setMode('dictionary')}
            id="mode-dictionary"
          >
            <BookOpen size={14} />
            <span>Dictionary</span>
            <span className="mode-btn-sub">Word-by-word mapping</span>
          </button>
          <button
            className={`mode-btn ${mode === 'ai' ? 'active' : ''}`}
            onClick={() => setMode('ai')}
            id="mode-ai"
          >
            <Sparkles size={14} />
            <span>AI Translation</span>
            <span className="mode-btn-sub">OpenAI GPT-4o</span>
          </button>
        </div>
      </div>

      {/* Section 3: Voice Controls */}
      <div className="converter-section">
        <div className="converter-section-label">
          <SlidersHorizontal size={12} />
          <span>Voice Controls</span>
        </div>
        <div className="converter-controls">
          <div className="converter-voice-row">
            {/* Gender toggle */}
            <div className="converter-gender-toggle">
              <button
                className={`gender-btn ${gender === 'male' ? 'active' : ''}`}
                onClick={() => setGender('male')}
                id="gender-male"
              >
                <Mic size={13} /> Male
              </button>
              <button
                className={`gender-btn ${gender === 'female' ? 'active' : ''}`}
                onClick={() => setGender('female')}
                id="gender-female"
              >
                <Mic size={13} /> Female
              </button>
            </div>

            {/* Convert button */}
            <button
              className="converter-go-btn"
              onClick={handleConvert}
              disabled={!target || loading}
              id="convert-btn"
            >
              {loading ? (
                <><div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> Converting...</>
              ) : (
                <><RefreshCw size={15} /> Convert</>
              )}
            </button>
          </div>

          {/* Sliders */}
          <div className="converter-sliders">
            <div className="converter-slider-group">
              <div className="converter-slider-header">
                <span className="converter-slider-label">Pitch</span>
                <span className="converter-slider-value">{pitch > 0 ? '+' : ''}{pitch}Hz</span>
              </div>
              <div className="converter-slider-track-wrap">
                <input
                  type="range" min="-50" max="50" value={pitch}
                  onChange={(e) => setPitch(parseInt(e.target.value))}
                  className="converter-range"
                  id="pitch-slider"
                />
                <div className="converter-slider-marks">
                  <span>-50</span><span>0</span><span>+50</span>
                </div>
              </div>
            </div>
            <div className="converter-slider-group">
              <div className="converter-slider-header">
                <span className="converter-slider-label">Speed</span>
                <span className="converter-slider-value">{rate > 0 ? '+' : ''}{rate}%</span>
              </div>
              <div className="converter-slider-track-wrap">
                <input
                  type="range" min="-50" max="50" value={rate}
                  onChange={(e) => setRate(parseInt(e.target.value))}
                  className="converter-range"
                  id="rate-slider"
                />
                <div className="converter-slider-marks">
                  <span>-50</span><span>0</span><span>+50</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Section 3: Result */}
      {result && (
        <div className="converter-section">
          <div className="converter-section-label">
            <Volume2 size={12} />
            <span>Result</span>
            {result.voice_used && (
              <span className="converter-voice-badge">{result.voice_used}</span>
            )}
          </div>

          <div className="converter-result-texts">
            <div className="converter-text-card original">
              <div className="converter-text-tag">Original · {sourceDialect || 'detected'}</div>
              <div className="converter-text-content">{result.original_text}</div>
            </div>
            <div className="converter-arrow-divider">
              <ArrowRight size={16} />
            </div>
            <div className="converter-text-card converted">
              <div className="converter-text-tag">
                <span>Converted · {result.target_dialect}</span>
                <Pencil size={10} className="converter-edit-hint" />
              </div>
              <textarea
                className="converter-text-editable"
                value={editedText}
                onChange={(e) => { setEditedText(e.target.value); setTextEdited(true); }}
                id="converted-text-editor"
              />
            </div>
          </div>

          <div className="converter-action-row">
            <button
              className="converter-play-btn"
              onClick={togglePlayback}
              id="play-converted-btn"
            >
              <span className="converter-play-circle">
                {playing ? <Pause size={18} /> : <Play size={18} />}
              </span>
              <span>{playing ? 'Pause Audio' : 'Play Converted Audio'}</span>
            </button>

            {textEdited && (
              <button
                className="converter-resynth-btn"
                onClick={handleResynthesize}
                disabled={syncing}
                id="resynth-btn"
              >
                {syncing ? (
                  <><div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Synthesizing...</>
                ) : (
                  <><RotateCcw size={14} /> Re-synthesize Audio</>
                )}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
