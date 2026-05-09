'use client';
import { useRef, useState, useEffect, useCallback } from 'react';
import { FolderOpen, Mic, Upload, Clock } from 'lucide-react';
import { uploadFile, getSamples } from '@/lib/api';

export default function FileUploader({ onFileSelect, selectedFileId }) {
  const inputRef = useRef(null);
  const [samples, setSamples] = useState({});
  const [uploading, setUploading] = useState(false);
  const [dragover, setDragover] = useState(false);

  useEffect(() => {
    getSamples().then(setSamples).catch(() => {});
  }, []);

  const handleUpload = useCallback(async (file) => {
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadFile(file);
      onFileSelect?.(result.file_id, result.filename, null);
    } catch (e) {
      alert('Upload failed: ' + e.message);
    }
    setUploading(false);
  }, [onFileSelect]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragover(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title"><FolderOpen size={18} /> Audio Source</span>
      </div>
      <div
        className={`upload-zone ${dragover ? 'dragover' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
        onDragLeave={() => setDragover(false)}
        onDrop={onDrop}
        id="upload-zone"
      >
        <input ref={inputRef} type="file" accept="audio/*" hidden onChange={(e) => handleUpload(e.target.files[0])} />
        {uploading ? (
          <div className="loading-spinner"><div className="spinner" /><span className="loading-text">Uploading...</span></div>
        ) : (
          <>
            <div className="upload-zone-icon"><Mic size={40} /></div>
            <p className="upload-zone-text"><strong>Click or drag</strong> an audio file here</p>
            <p className="upload-zone-text" style={{ fontSize: '0.8rem', marginTop: 4 }}>WAV, MP3, OGG supported</p>
          </>
        )}
      </div>

      <div style={{ marginTop: 20 }}>
        <div className="card-title" style={{ marginBottom: 12 }}><FolderOpen size={16} /> Sample Files</div>
        {Object.entries(samples).map(([dialect, data]) => (
          <div className="sample-group" key={dialect}>
            <div className="sample-group-title">{data.label}</div>
            {data.files?.map((f) => (
              <div
                key={f.file_id}
                className={`sample-item ${selectedFileId === f.file_id ? 'active' : ''}`}
                onClick={() => onFileSelect?.(f.file_id, f.filename, dialect)}
                id={`sample-${f.file_id}`}
              >
                <span className={`sample-dialect-dot dot-${dialect}`} />
                <span>{f.filename}</span>
                <span className="sample-duration">{f.duration}s</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
