'use client';
import { Target, CheckCircle } from 'lucide-react';

const DIALECT_NAMES = {
  egyptian: { en: 'Egyptian', ar: 'مصري', color: 'var(--primary)' },
  gulf: { en: 'Gulf', ar: 'خليجي', color: 'var(--secondary)' },
  levantine: { en: 'Levantine', ar: 'شامي', color: 'var(--accent)' },
  maghrebi: { en: 'Maghrebi', ar: 'مغربي', color: '#E8A838' },
};

export default function ClassificationResult({ classificationData }) {
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

  const { predicted_dialect, confidence, probabilities, probability_chart, importance_chart, svm_prediction } = classificationData;
  const dialectInfo = DIALECT_NAMES[predicted_dialect] || { en: predicted_dialect, ar: '', color: 'var(--text)' };

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title"><Target size={18} /> Classification Result</span>
        <span className="card-badge badge-success">RF + SVM</span>
      </div>

      <div className="result-dialect">
        <div className="result-dialect-name" style={{ color: dialectInfo.color }}>
          {dialectInfo.en}
        </div>
        <div className="result-dialect-arabic">{dialectInfo.ar}</div>
        <div className="result-confidence" style={{ background: 'var(--success-dim)', color: 'var(--success)' }}>
          <CheckCircle size={16} /> {(confidence * 100).toFixed(1)}% Confidence
        </div>
        {svm_prediction && (
          <div style={{ marginTop: 8, fontSize: '0.8rem', color: 'var(--text-dim)' }}>
            SVM agrees: {DIALECT_NAMES[svm_prediction]?.en || svm_prediction}
          </div>
        )}
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

      {importance_chart && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
            Feature Importances
          </h3>
          <div className="chart-container">
            <img src={`data:image/png;base64,${importance_chart}`} alt="Feature Importance" />
          </div>
        </div>
      )}
    </div>
  );
}
