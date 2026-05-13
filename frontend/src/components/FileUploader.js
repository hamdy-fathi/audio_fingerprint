'use client';
import { useRef, useState, useEffect, useCallback } from 'react';
import { Upload, Mic, Play, ChevronDown, ChevronRight, Clock, FileAudio, Check } from 'lucide-react';
import { uploadFile, getSamples } from '@/lib/api';

const DIALECT_META = {
  egyptian:  { label: 'Egyptian',  ar: 'مصري',  color: 'var(--primary)' },
  gulf:      { label: 'Gulf',      ar: 'خليجي', color: 'var(--secondary)' },
  levantine: { label: 'Levantine', ar: 'شامي',  color: 'var(--accent)' },
  maghrebi:  { label: 'Maghrebi',  ar: 'مغربي', color: '#E8A838' },
};

export default function FileUploader({ onFileSelect, selectedFileId }) {
  const inputRef = useRef(null);
  const [samples, setSamples] = useState({});
  const [uploading, setUploading] = useState(false);
  const [dragover, setDragover] = useState(false);
  const [expandedDialect, setExpandedDialect] = useState(null);

  useEffect(() => {
    getSamples().then((data) => {
      setSamples(data);
      // auto-expand the first dialect with files
      const firstKey = Object.keys(data).find(k => data[k]?.files?.length > 0);
      if (firstKey) setExpandedDialect(firstKey);
    }).catch(() => {});
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

  const toggleDialect = (dialect) => {
    setExpandedDialect(prev => prev === dialect ? null : dialect);
  };

  const formatDuration = (d) => {
    const s = Math.floor(d);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}:${sec.toString().padStart(2, '0')}` : `0:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className="audio-source-panel">
      {/* Upload Area */}
      <div
        className={`upload-drop ${dragover ? 'dragover' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
        onDragLeave={() => setDragover(false)}
        onDrop={onDrop}
        id="upload-zone"
      >
        <input ref={inputRef} type="file" accept="audio/*" hidden onChange={(e) => handleUpload(e.target.files[0])} />
        {uploading ? (
          <div className="upload-drop-inner">
            <div className="upload-spinner"><div className="spinner" /></div>
            <span className="upload-drop-label">Uploading...</span>
          </div>
        ) : (
          <div className="upload-drop-inner">
            <div className="upload-drop-icon">
              <Upload size={20} />
            </div>
            <div>
              <span className="upload-drop-label">Upload Audio</span>
              <span className="upload-drop-hint">or drag & drop · WAV, MP3, OGG</span>
            </div>
          </div>
        )}
      </div>

      {/* Sample Library */}
      <div className="sample-library">
        <div className="sample-library-title">
          <FileAudio size={14} />
          <span>Sample Library</span>
          <span className="sample-count">{Object.values(samples).reduce((acc, d) => acc + (d.files?.length || 0), 0)}</span>
        </div>

        {Object.entries(samples).map(([dialect, data]) => {
          const meta = DIALECT_META[dialect] || { label: dialect, ar: '', color: 'var(--text-dim)' };
          const isExpanded = expandedDialect === dialect;
          const fileCount = data.files?.length || 0;
          const hasSelected = data.files?.some(f => f.file_id === selectedFileId);

          return (
            <div className="dialect-group" key={dialect}>
              <button
                className={`dialect-group-header ${isExpanded ? 'expanded' : ''} ${hasSelected ? 'has-selected' : ''}`}
                onClick={() => toggleDialect(dialect)}
                id={`dialect-group-${dialect}`}
              >
                <span className="dialect-group-dot" style={{ background: meta.color }} />
                <span className="dialect-group-name">{meta.label}</span>
                <span className="dialect-group-ar">{meta.ar}</span>
                <span className="dialect-group-count">{fileCount}</span>
                {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </button>

              {isExpanded && (
                <div className="dialect-group-files">
                  {data.files?.map((f) => {
                    const isActive = selectedFileId === f.file_id;
                    return (
                      <button
                        key={f.file_id}
                        className={`file-row ${isActive ? 'active' : ''}`}
                        onClick={() => onFileSelect?.(f.file_id, f.filename, dialect)}
                        id={`sample-${f.file_id}`}
                      >
                        <span className="file-row-icon">
                          {isActive ? <Check size={12} /> : <Play size={12} />}
                        </span>
                        <span className="file-row-name">{f.filename}</span>
                        <span className="file-row-dur">
                          <Clock size={10} />
                          {formatDuration(f.duration)}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
