'use client';
import { useState, useCallback } from 'react';
import { BarChart3, FileText, RefreshCw, Sliders, Search, Paperclip, AudioLines } from 'lucide-react';
import FileUploader from '@/components/FileUploader';
import AudioPlayer from '@/components/AudioPlayer';
import SpectrogramViewer from '@/components/SpectrogramViewer';
import ClassificationResult from '@/components/ClassificationResult';
import UMAPVisualizer from '@/components/UMAPVisualizer';
import TranscriptionPanel from '@/components/TranscriptionPanel';
import DialectConverter from '@/components/DialectConverter';
import AudioMixer from '@/components/AudioMixer';
import { getSpectrogram, classifyDialect, getUMAPProjection, transcribeFile } from '@/lib/api';

const TABS = [
  { key: 'analyze', icon: BarChart3, label: 'Analyze & Classify' },
  { key: 'transcribe', icon: FileText, label: 'Transcription' },
  { key: 'convert', icon: RefreshCw, label: 'Dialect Converter' },
  { key: 'mix', icon: Sliders, label: 'Audio Mixer' },
];

export default function Home() {
  const [tab, setTab] = useState('analyze');
  const [fileId, setFileId] = useState(null);
  const [fileName, setFileName] = useState('');
  const [fileDialect, setFileDialect] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);

  const [specData, setSpecData] = useState(null);
  const [classData, setClassData] = useState(null);
  const [umapData, setUmapData] = useState(null);
  const [transSegments, setTransSegments] = useState([]);
  const [loadingSpec, setLoadingSpec] = useState(false);
  const [loadingTrans, setLoadingTrans] = useState(false);

  const handleFileSelect = useCallback((id, name, dialect) => {
    setFileId(id);
    setFileName(name);
    setFileDialect(dialect);
    setSpecData(null);
    setClassData(null);
    setUmapData(null);
    setTransSegments([]);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!fileId) return;
    setLoadingSpec(true);
    try {
      const [spec, cls, umap] = await Promise.all([
        getSpectrogram(fileId),
        classifyDialect(fileId),
        getUMAPProjection(fileId),
      ]);
      setSpecData(spec);
      setClassData(cls);
      setUmapData(umap);
    } catch (e) {
      alert('Analysis error: ' + e.message);
    }
    setLoadingSpec(false);
  }, [fileId]);

  const handleTranscribe = useCallback(async () => {
    if (!fileId) return;
    setLoadingTrans(true);
    setTransSegments([]);
    try {
      const result = await transcribeFile(fileId);
      setTransSegments(result.segments || []);
    } catch (e) {
      alert('Transcription error: ' + e.message);
    }
    setLoadingTrans(false);
  }, [fileId]);

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-title">
          <div className="logo-icon"><AudioLines size={22} /></div>
          <div>
            <h1>Arabic Dialect Fingerprint</h1>
            <p className="header-subtitle">Detect, Analyze & Convert Arabic Dialects</p>
          </div>
        </div>
        {fileName && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Paperclip size={14} style={{ color: 'var(--text-secondary)' }} />
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{fileName}</span>
            {fileDialect && <span className="card-badge badge-primary">{fileDialect}</span>}
          </div>
        )}
      </header>

      <nav className="tabs" id="main-tabs">
        {TABS.map((t) => {
          const Icon = t.icon;
          return (
            <button key={t.key} className={`tab ${tab === t.key ? 'active' : ''}`} onClick={() => setTab(t.key)} id={`tab-${t.key}`}>
              <span className="tab-icon"><Icon size={16} /></span> {t.label}
            </button>
          );
        })}
      </nav>

      <div className="grid-sidebar">
        <aside>
          <FileUploader onFileSelect={handleFileSelect} selectedFileId={fileId} />
          {fileId && (
            <div style={{ marginTop: 12 }}>
              <AudioPlayer fileId={fileId} onTimeUpdate={setCurrentTime} />
            </div>
          )}
        </aside>

        <main>
          {tab === 'analyze' && (
            <>
              <div style={{ marginBottom: 16 }}>
                <button className="btn btn-primary" onClick={handleAnalyze} disabled={!fileId || loadingSpec} id="analyze-btn">
                  <Search size={16} /> {loadingSpec ? 'Analyzing...' : 'Analyze & Classify'}
                </button>
              </div>

              {loadingSpec ? (
                <div className="card"><div className="loading-spinner"><div className="spinner" /><span className="loading-text">Generating spectrograms & extracting features...</span></div></div>
              ) : (
                <>
                  <SpectrogramViewer spectrogramData={specData} />
                  <ClassificationResult classificationData={classData} />
                  <UMAPVisualizer umapData={umapData} />
                </>
              )}
            </>
          )}

          {tab === 'transcribe' && (
            <>
              <div style={{ marginBottom: 16 }}>
                <button className="btn btn-primary" onClick={handleTranscribe} disabled={!fileId || loadingTrans} id="transcribe-btn">
                  <FileText size={16} /> {loadingTrans ? 'Transcribing...' : 'Transcribe Audio'}
                </button>
              </div>
              <TranscriptionPanel segments={transSegments} currentTime={currentTime} isLoading={loadingTrans} />
            </>
          )}

          {tab === 'convert' && (
            <DialectConverter fileId={fileId} sourceDialect={classData?.predicted_dialect || fileDialect} />
          )}

          {tab === 'mix' && <AudioMixer />}
        </main>
      </div>
    </div>
  );
}
