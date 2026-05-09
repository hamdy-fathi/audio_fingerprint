const API_BASE = 'http://localhost:8001/api';
const WS_BASE = 'ws://localhost:8001/api';

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

export async function getSamples() {
  const res = await fetch(`${API_BASE}/samples`);
  if (!res.ok) throw new Error('Failed to load samples');
  return res.json();
}

export function getFileUrl(fileId) {
  return `${API_BASE}/file/${fileId}`;
}

export async function getSpectrogram(fileId) {
  const res = await fetch(`${API_BASE}/spectrogram`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId }),
  });
  if (!res.ok) throw new Error('Failed to get spectrogram');
  return res.json();
}

export async function classifyDialect(fileId) {
  const res = await fetch(`${API_BASE}/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId }),
  });
  if (!res.ok) throw new Error('Classification failed');
  return res.json();
}

export async function getFeatures(fileId) {
  const res = await fetch(`${API_BASE}/features`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId }),
  });
  if (!res.ok) throw new Error('Feature extraction failed');
  return res.json();
}

export async function transcribeFile(fileId) {
  const res = await fetch(`${API_BASE}/transcribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId }),
  });
  if (!res.ok) throw new Error('Transcription failed');
  return res.json();
}

export function connectTranscriptionWS(fileId, onSegment, onDone, onError) {
  const ws = new WebSocket(`${WS_BASE}/transcribe/ws`);
  ws.onopen = () => ws.send(JSON.stringify({ file_id: fileId }));
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.done) { onDone?.(); ws.close(); }
    else if (data.error) { onError?.(data.error); ws.close(); }
    else onSegment?.(data);
  };
  ws.onerror = () => onError?.('WebSocket error');
  return ws;
}

export async function convertDialect(fileId, targetDialect, sourceDialect, gender) {
  const res = await fetch(`${API_BASE}/convert-dialect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      file_id: fileId,
      target_dialect: targetDialect,
      source_dialect: sourceDialect || null,
      gender: gender || 'male',
    }),
  });
  if (!res.ok) throw new Error('Conversion failed');
  return res.json();
}

export async function getDialectVoices() {
  const res = await fetch(`${API_BASE}/dialect-voices`);
  if (!res.ok) throw new Error('Failed to get voices');
  return res.json();
}

export async function mixAndClassify(fileId1, fileId2, weight) {
  const res = await fetch(`${API_BASE}/mix-and-classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      file_id_1: fileId1,
      file_id_2: fileId2,
      weight: weight,
    }),
  });
  if (!res.ok) throw new Error('Mix and classify failed');
  return res.json();
}
