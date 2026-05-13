'use client';
import { useState, useEffect, useRef, useCallback } from 'react';
import { Sliders, FolderOpen, Play, Pause, Plus, Upload } from 'lucide-react';
import { getSamples, mixAndClassify, uploadFile } from '@/lib/api';

const DIALECT_COLORS = {
  egyptian: 'var(--primary)', gulf: 'var(--secondary)',
  levantine: 'var(--accent)', maghrebi: '#E8A838',
};
const DIALECT_NAMES = {
  egyptian: 'Egyptian', gulf: 'Gulf', levantine: 'Levantine', maghrebi: 'Maghrebi',
};

export default function AudioMixer() {
  const [samples, setSamples] = useState({});
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [weight, setWeight] = useState(0.5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showPicker, setShowPicker] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [uploading, setUploading] = useState(null);
  const audioRef = useRef(null);
  const uploadInputRef = useRef(null);
  const activeSlotRef = useRef(null);

  useEffect(() => { getSamples().then(setSamples).catch(() => {}); }, []);

  const handleMix = useCallback(async () => {
    if (!file1 || !file2) return;
    setLoading(true);
    setResult(null);
    try {
      const data = await mixAndClassify(file1.file_id, file2.file_id, weight);
      setResult(data);
    } catch (e) { alert('Mix failed: ' + e.message); }
    setLoading(false);
  }, [file1, file2, weight]);

  const togglePlayback = () => {
    if (!result?.mixed_audio || !audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      if (!audioRef.current.src || audioRef.current.ended) {
        audioRef.current.src = `data:audio/wav;base64,${result.mixed_audio}`;
      }
      audioRef.current.play();
      setPlaying(true);
    }
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onEnded = () => setPlaying(false);
    audio.addEventListener('ended', onEnded);
    return () => audio.removeEventListener('ended', onEnded);
  }, []);

  const pickFile = (slot, fileId, filename, dialect) => {
    const obj = { file_id: fileId, filename, dialect };
    if (slot === 1) setFile1(obj);
    else setFile2(obj);
    setShowPicker(null);
  };

  const handleUploadForSlot = async (slot, file) => {
    if (!file) return;
    setUploading(slot);
    try {
      const result = await uploadFile(file);
      pickFile(slot, result.file_id, result.filename, null);
    } catch (e) {
      alert('Upload failed: ' + e.message);
    }
    setUploading(null);
  };

  const triggerUpload = (slot) => {
    activeSlotRef.current = slot;
    uploadInputRef.current?.click();
  };

  const onUploadChange = (e) => {
    const file = e.target.files[0];
    if (file && activeSlotRef.current) {
      handleUploadForSlot(activeSlotRef.current, file);
    }
    e.target.value = '';
  };

  const renderPicker = (slot) => (
    <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 10, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', maxHeight: 300, overflow: 'auto', padding: 8 }}>
      {/* Upload option */}
      <div
        className="mixer-upload-option"
        onClick={() => triggerUpload(slot)}
      >
        {uploading === slot ? (
          <><div className="spinner" style={{ width: 14, height: 14 }} /> <span>Uploading...</span></>
        ) : (
          <><Upload size={14} /> <span>Upload External File</span></>
        )}
      </div>
      <div style={{ borderBottom: '1px solid var(--border)', margin: '6px 0' }} />
      {Object.entries(samples).map(([dialect, data]) => (
        <div key={dialect}>
          <div className="sample-group-title">{data.label}</div>
          {data.files?.map((f) => (
            <div key={f.file_id} className="sample-item" onClick={() => pickFile(slot, f.file_id, f.filename, dialect)}>
              <span className={`sample-dialect-dot dot-${dialect}`} />
              <span>{f.filename}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title"><Sliders size={18} /> Audio Mixer</span>
        <span className="card-badge badge-warning">Weighted Average</span>
      </div>

      <input ref={uploadInputRef} type="file" accept="audio/*" hidden onChange={onUploadChange} />

      <div className="mixer-slots">
        <div className={`mixer-slot ${file1 ? 'filled' : ''}`} style={{ position: 'relative' }}>
          {file1 ? (
            <><div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{file1.filename}</div>
            <span className="card-badge badge-primary">{DIALECT_NAMES[file1.dialect] || 'Uploaded'}</span>
            <button className="btn btn-sm btn-secondary" onClick={() => setFile1(null)}>Change</button></>
          ) : (
            <button className="btn btn-secondary" onClick={() => setShowPicker(showPicker === 1 ? null : 1)} id="mixer-pick-1">
              <FolderOpen size={16} /> Select File 1
            </button>
          )}
          {showPicker === 1 && renderPicker(1)}
        </div>

        <div className="mixer-connector"><Plus size={24} /></div>

        <div className={`mixer-slot ${file2 ? 'filled' : ''}`} style={{ position: 'relative' }}>
          {file2 ? (
            <><div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{file2.filename}</div>
            <span className="card-badge badge-secondary">{DIALECT_NAMES[file2.dialect] || 'Uploaded'}</span>
            <button className="btn btn-sm btn-secondary" onClick={() => setFile2(null)}>Change</button></>
          ) : (
            <button className="btn btn-secondary" onClick={() => setShowPicker(showPicker === 2 ? null : 2)} id="mixer-pick-2">
              <FolderOpen size={16} /> Select File 2
            </button>
          )}
          {showPicker === 2 && renderPicker(2)}
        </div>
      </div>

      <div className="weight-slider-container">
        <div className="weight-slider-labels">
          <span>100% File 1</span>
          <span>100% File 2</span>
        </div>
        <input type="range" className="weight-slider" min="0" max="1" step="0.01" value={weight} onChange={(e) => setWeight(parseFloat(e.target.value))} id="mix-weight-slider" />
        <div className="weight-value">
          File 1: {((1 - weight) * 100).toFixed(0)}% | File 2: {(weight * 100).toFixed(0)}%
        </div>
      </div>

      <div style={{ marginTop: 16, textAlign: 'center' }}>
        <button className="btn btn-primary" onClick={handleMix} disabled={!file1 || !file2 || loading} id="mix-btn">
          <Sliders size={16} /> {loading ? 'Mixing & Classifying...' : 'Mix & Classify'}
        </button>
      </div>

      {result && (
        <div style={{ marginTop: 20 }}>
          <audio ref={audioRef} />
          <div style={{ marginBottom: 16 }}>
            <button className="btn btn-accent" onClick={togglePlayback}>
              {playing ? <Pause size={16} /> : <Play size={16} />}
              {playing ? 'Pause' : 'Play Mixed Audio'}
            </button>
          </div>

          <div className="result-dialect" style={{ padding: 16 }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: 4 }}>CLOSEST DIALECT</div>
            <div className="result-dialect-name" style={{ color: DIALECT_COLORS[result.classification] || 'var(--text)', fontSize: '1.4rem' }}>
              {DIALECT_NAMES[result.classification] || result.classification}
            </div>
            <div className="result-confidence" style={{ background: 'var(--success-dim)', color: 'var(--success)' }}>
              {(result.confidence * 100).toFixed(1)}% confidence
            </div>
          </div>

          {result.probabilities && (
            <div className="prob-bars">
              {Object.entries(result.probabilities).sort(([,a],[,b]) => b - a).map(([d, p]) => (
                <div className="prob-bar-row" key={d}>
                  <span className="prob-bar-label">{DIALECT_NAMES[d] || d}</span>
                  <div className="prob-bar-track">
                    <div className="prob-bar-fill" style={{ width: `${p*100}%`, background: DIALECT_COLORS[d] || '#888' }}>
                      <span className="prob-bar-value">{(p*100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {result.spectrogram && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: 6 }}>MIXED SPECTROGRAM</div>
              <div className="spectrogram-container">
                <img src={`data:image/png;base64,${result.spectrogram}`} alt="Mixed Spectrogram" />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
