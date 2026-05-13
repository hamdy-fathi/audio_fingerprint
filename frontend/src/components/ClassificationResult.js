'use client';
import { useState } from 'react';
import { Target, CheckCircle, BarChart3, ChevronDown, ChevronUp, Info, BookOpen } from 'lucide-react';

const DIALECT_NAMES = {
  egyptian: { en: 'Egyptian', ar: 'مصري', color: 'var(--primary)' },
  gulf: { en: 'Gulf', ar: 'خليجي', color: 'var(--secondary)' },
  levantine: { en: 'Levantine', ar: 'شامي', color: 'var(--accent)' },
  maghrebi: { en: 'Maghrebi', ar: 'مغربي', color: '#E8A838' },
};

const FEATURE_CATEGORIES = {
  mfcc: { label: 'MFCC', color: 'var(--primary)', desc: 'Mel-frequency cepstral coefficients — capture vocal tract shape' },
  delta: { label: 'Delta', color: 'var(--accent)', desc: 'Rate of change of MFCCs — capture speech dynamics' },
  delta2: { label: 'Delta²', color: '#6DD4A0', desc: 'Acceleration of MFCCs — capture rapid articulation shifts' },
  contrast: { label: 'Spectral Contrast', color: 'var(--secondary)', desc: 'Frequency peak vs valley difference — detect tonal patterns' },
  chroma: { label: 'Chroma', color: '#E8A838', desc: 'Pitch class distribution — capture melodic dialect traits' },
  rms: { label: 'RMS Energy', color: '#D97706', desc: 'Signal loudness — measure speech intensity patterns' },
  zcr: { label: 'ZCR', color: '#E066A0', desc: 'Zero crossing rate — detect fricative/plosive sounds' },
};

const TOP_FEATURES = [
  { feature: 'delta2_2_std', importance: 0.0342 },
  { feature: 'contrast_4_mean', importance: 0.0306 },
  { feature: 'delta_4_std', importance: 0.0221 },
  { feature: 'delta2_9_std', importance: 0.0209 },
  { feature: 'mfcc_12_mean', importance: 0.0159 },
  { feature: 'delta2_1_std', importance: 0.0155 },
  { feature: 'mfcc_4_mean', importance: 0.0140 },
  { feature: 'rms_1_std', importance: 0.0139 },
  { feature: 'delta2_14_std', importance: 0.0136 },
  { feature: 'mfcc_6_mean', importance: 0.0135 },
  { feature: 'contrast_7_std', importance: 0.0128 },
  { feature: 'contrast_3_mean', importance: 0.0127 },
  { feature: 'contrast_1_mean', importance: 0.0124 },
  { feature: 'mfcc_1_mean', importance: 0.0122 },
  { feature: 'mfcc_15_std', importance: 0.0118 },
  { feature: 'delta2_3_std', importance: 0.0115 },
  { feature: 'delta2_7_std', importance: 0.0108 },
  { feature: 'mfcc_5_mean', importance: 0.0105 },
  { feature: 'contrast_4_std', importance: 0.0104 },
  { feature: 'contrast_6_mean', importance: 0.0103 },
];

const FEATURE_EXPLANATIONS = {
  'delta2_2_std': 'Variation in acceleration of MFCC coefficient 2 over time. Strong cue for speech rhythm and changing pronunciation dynamics.',
  'contrast_4_mean': 'Average spectral contrast in frequency band 4. Captures speech texture and peak/valley energy differences.',
  'delta_4_std': 'Variation in the first derivative of MFCC coefficient 4. Measures how pronunciation changes over time.',
  'delta2_9_std': 'Variation in acceleration of MFCC coefficient 9. Captures dynamic articulation differences.',
  'mfcc_12_mean': 'Average MFCC coefficient 12. Represents a stable spectral/pronunciation property of the speaker audio.',
  'delta2_1_std': 'Variation in acceleration of MFCC coefficient 1. Important for overall speech-dynamic changes.',
  'mfcc_4_mean': 'Average MFCC coefficient 4. Often linked to broad vocal tract and vowel/consonant acoustic patterns.',
  'rms_1_std': 'Variation in loudness/energy. Can reflect stress, emphasis, and speaking intensity changes.',
  'delta2_14_std': 'Variation in acceleration of MFCC coefficient 14. Captures subtle high-detail spectral dynamics.',
  'mfcc_6_mean': 'Average MFCC coefficient 6. Represents another stable spectral envelope cue.',
  'contrast_7_std': 'Variation in spectral contrast in frequency band 7. Can reflect changes in sharpness and high-frequency texture.',
  'contrast_3_mean': 'Average spectral contrast in band 3. Captures mid-frequency texture differences.',
  'contrast_1_mean': 'Average spectral contrast in band 1. Related to lower-frequency resonance and energy distribution.',
  'mfcc_1_mean': 'Average MFCC coefficient 1. Often related to overall spectral tilt and general voice resonance.',
  'mfcc_15_std': 'Variation in MFCC coefficient 15. Captures subtle spectral changes across the audio clip.',
  'delta2_3_std': 'Variation in acceleration of MFCC coefficient 3. Captures pronunciation dynamics.',
  'delta2_7_std': 'Variation in acceleration of MFCC coefficient 7. Captures dynamic articulation patterns.',
  'mfcc_5_mean': 'Average MFCC coefficient 5. Linked to vocal tract and consonant acoustic patterns.',
  'contrast_4_std': 'Variation in spectral contrast in band 4. Captures texture changes in speech.',
  'contrast_6_mean': 'Average spectral contrast in band 6. Captures high-frequency texture differences.',
};

const FAMILY_EXPLANATIONS = [
  { family: 'MFCC', measures: 'Summarize the spectral envelope of speech, related to vocal tract shape and pronunciation.', why: 'Dialect differences appear in vowel quality, consonant articulation, and pronunciation style.' },
  { family: 'Delta MFCC', measures: 'First time-derivative of MFCCs — how MFCC values change over time.', why: 'Captures speech transitions, rhythm, and phoneme-to-phoneme movement.' },
  { family: 'Delta² MFCC', measures: 'Second time-derivative of MFCCs — acceleration of acoustic changes.', why: 'Captures dynamic speaking patterns and rhythm changes between dialects.' },
  { family: 'Spectral Contrast', measures: 'Difference between spectral peaks and valleys in frequency bands.', why: 'Reflects speech texture, consonant sharpness, and energy distribution.' },
  { family: 'Chroma', measures: 'Energy distribution across pitch classes.', why: 'May capture pitch and prosody-related patterns.' },
  { family: 'ZCR', measures: 'How often the waveform crosses zero.', why: 'Indicates noisiness, sharpness, or unvoiced consonant characteristics.' },
  { family: 'RMS', measures: 'Root Mean Square energy — signal loudness.', why: 'Captures stress, emphasis, and energy variation across speech.' },
];

function getFeatureCategory(name) {
  if (name.startsWith('delta2')) return 'delta2';
  if (name.startsWith('delta')) return 'delta';
  if (name.startsWith('mfcc')) return 'mfcc';
  if (name.startsWith('contrast')) return 'contrast';
  if (name.startsWith('chroma')) return 'chroma';
  if (name.startsWith('rms')) return 'rms';
  if (name.startsWith('zcr')) return 'zcr';
  return 'mfcc';
}

function formatFeatureName(raw) {
  // e.g. "delta2_2_std" → "Delta² #2 (σ)"
  //      "mfcc_12_mean" → "MFCC #12 (μ)"
  const cat = getFeatureCategory(raw);
  const catInfo = FEATURE_CATEGORIES[cat] || { label: cat };
  const parts = raw.replace(cat + '_', '').split('_');
  const num = parts[0];
  const stat = parts[1] === 'mean' ? 'μ' : 'σ';
  return { label: `${catInfo.label} #${num}`, stat, catKey: cat };
}

export default function ClassificationResult({ classificationData }) {
  const [showAll, setShowAll] = useState(false);
  const [showExplain, setShowExplain] = useState(false);

  if (!classificationData) {
    return (
      <div className="card">
        <div className="card-title"><Target size={18} /> Classification</div>
        <div className="empty-state">
          <div className="empty-state-icon"><Target size={48} /></div>
          <p className="empty-state-text">Classify a file to see results</p>
        </div>
      </div>
    );
  }

  const { predicted_dialect, confidence, probabilities, probability_chart } = classificationData;
  const dialectInfo = DIALECT_NAMES[predicted_dialect] || { en: predicted_dialect, ar: '', color: 'var(--text)' };

  const maxImportance = TOP_FEATURES[0].importance;
  const visibleFeatures = showAll ? TOP_FEATURES : TOP_FEATURES.slice(0, 10);

  // Group features by category for the summary
  const categoryCounts = {};
  TOP_FEATURES.forEach(f => {
    const cat = getFeatureCategory(f.feature);
    categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
  });

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title"><Target size={18} /> Classification Result</span>
        <span className="card-badge badge-primary">XGBoost</span>
      </div>

      <div className="cls-layout">
        {/* Left: Dialect Result + Probabilities */}
        <div className="cls-left">
          <div className="result-dialect">
            <div className="result-dialect-name" style={{ color: dialectInfo.color }}>
              {dialectInfo.en}
            </div>
            <div className="result-dialect-arabic">{dialectInfo.ar}</div>
            <div className="result-confidence" style={{ background: 'var(--success-dim)', color: 'var(--success)' }}>
              <CheckCircle size={16} /> {(confidence * 100).toFixed(1)}% Confidence
            </div>
          </div>

          {probabilities && (
            <div className="prob-bars">
              {Object.entries(probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([dialect, prob]) => {
                  const info = DIALECT_NAMES[dialect] || { en: dialect, color: '#888' };
                  return (
                    <div className="prob-bar-row" key={dialect}>
                      <span className="prob-bar-label">{info.en}</span>
                      <div className="prob-bar-track">
                        <div className="prob-bar-fill" style={{ width: `${prob * 100}%`, background: info.color }}>
                          <span className="prob-bar-value">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </div>

        {/* Right: Feature Importances */}
        <div className="cls-right">
          <div className="fi-header">
            <div className="fi-title">
              <BarChart3 size={15} />
              <span>Top Feature Importances</span>
            </div>
            <div className="fi-tags">
              {Object.entries(categoryCounts).map(([cat, count]) => {
                const catInfo = FEATURE_CATEGORIES[cat];
                return (
                  <span key={cat} className="fi-tag" style={{ color: catInfo?.color, background: `color-mix(in srgb, ${catInfo?.color || '#888'} 12%, transparent)` }}>
                    {catInfo?.label || cat} ×{count}
                  </span>
                );
              })}
            </div>
          </div>

          <div className="fi-bars">
            {visibleFeatures.map((f, i) => {
              const { label, stat, catKey } = formatFeatureName(f.feature);
              const catInfo = FEATURE_CATEGORIES[catKey];
              const pct = (f.importance / maxImportance) * 100;
              return (
                <div className="fi-row" key={f.feature}>
                  <span className="fi-rank">#{i + 1}</span>
                  <div className="fi-label-wrap">
                    <span className="fi-label">{label}</span>
                    <span className="fi-stat" style={{ color: catInfo?.color }}>{stat}</span>
                  </div>
                  <div className="fi-track">
                    <div
                      className="fi-fill"
                      style={{
                        width: `${pct}%`,
                        background: catInfo?.color || 'var(--primary)',
                      }}
                    />
                  </div>
                  <span className="fi-val">{(f.importance * 100).toFixed(2)}%</span>
                </div>
              );
            })}
          </div>

          <button className="fi-toggle" onClick={() => setShowAll(!showAll)}>
            {showAll ? <><ChevronUp size={14} /> Show Top 10</> : <><ChevronDown size={14} /> Show All 20</>}
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="fi-legend">
        {Object.entries(FEATURE_CATEGORIES).map(([key, info]) => (
          <div key={key} className="fi-legend-item" title={info.desc}>
            <span className="fi-legend-dot" style={{ background: info.color }} />
            <span>{info.label}</span>
          </div>
        ))}
      </div>

      {/* Feature Explanation Tab */}
      <button className="fi-explain-toggle" onClick={() => setShowExplain(!showExplain)}>
        {showExplain ? <ChevronUp size={14} /> : <Info size={14} />}
        <span>{showExplain ? 'Hide Explanation' : 'What do these features mean?'}</span>
      </button>

      {showExplain && (
        <div className="fi-explain">
          <div className="fi-explain-section">
            <div className="fi-explain-heading">
              <BookOpen size={14} />
              <span>Top Features — Explained</span>
            </div>
            <p className="fi-explain-intro">
              These are the most important features used by the XGBoost model. Higher importance means the model used that feature more often to separate dialect classes.
            </p>
            <div className="fi-explain-grid">
              {TOP_FEATURES.slice(0, 15).map((f) => {
                const { label, stat, catKey } = formatFeatureName(f.feature);
                const catInfo = FEATURE_CATEGORIES[catKey];
                return (
                  <div className="fi-explain-card" key={f.feature}>
                    <div className="fi-explain-card-head">
                      <span className="fi-explain-card-dot" style={{ background: catInfo?.color }} />
                      <span className="fi-explain-card-name">{label} <span style={{ color: catInfo?.color }}>{stat}</span></span>
                      <span className="fi-explain-card-pct">{(f.importance * 100).toFixed(2)}%</span>
                    </div>
                    <p className="fi-explain-card-desc">{FEATURE_EXPLANATIONS[f.feature]}</p>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="fi-explain-section">
            <div className="fi-explain-heading">
              <BarChart3 size={14} />
              <span>Feature Families</span>
            </div>
            <div className="fi-family-table">
              <div className="fi-family-header">
                <span>Family</span>
                <span>What it Measures</span>
                <span>Why it Helps</span>
              </div>
              {FAMILY_EXPLANATIONS.map((fam) => (
                <div className="fi-family-row" key={fam.family}>
                  <span className="fi-family-name">{fam.family}</span>
                  <span className="fi-family-desc">{fam.measures}</span>
                  <span className="fi-family-why">{fam.why}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
