"""
Feature visualization module.
Generates comparison charts showing how features distinguish dialects.
"""
import io, base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COLORS = {
    'bg': '#1A1410',
    'surface': '#2A201A',
    'primary': '#7C3AED',
    'secondary': '#C75B39',
    'accent': '#7A9E7E',
    'text': '#F5E6D3',
    'border': '#3E342C'
}

DIALECT_COLORS = {
    'egyptian': '#7C3AED',
    'gulf': '#C75B39',
    'levantine': '#7A9E7E',
    'maghrebi': '#E8A838'
}

DIALECT_LABELS = {
    'egyptian': 'Egyptian (مصري)',
    'gulf': 'Gulf (خليجي)',
    'levantine': 'Levantine (شامي)',
    'maghrebi': 'Maghrebi (مغربي)'
}


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=COLORS['bg'], dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_mfcc_comparison(file_features, dialect_avg_features):
    """Bar chart comparing MFCC means of uploaded file vs each dialect average."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    n_mfcc = 13
    x = np.arange(n_mfcc)
    width = 0.15

    # File's MFCCs
    file_mfccs = [file_features.get(f'mfcc_{i+1}_mean', 0) for i in range(n_mfcc)]
    ax.bar(x - 2*width, file_mfccs, width, label='Uploaded File', color='#E2E8F0', edgecolor='#E2E8F0', alpha=0.9)

    # Each dialect's average MFCCs
    for idx, (dialect, color) in enumerate(DIALECT_COLORS.items()):
        if dialect in dialect_avg_features:
            d_mfccs = [dialect_avg_features[dialect].get(f'mfcc_{i+1}_mean', 0) for i in range(n_mfcc)]
            ax.bar(x + (idx-1)*width, d_mfccs, width, label=DIALECT_LABELS[dialect], color=color, alpha=0.8)

    ax.set_xlabel('MFCC Coefficient', color=COLORS['text'], fontsize=12)
    ax.set_ylabel('Mean Value', color=COLORS['text'], fontsize=12)
    ax.set_title('MFCC Profile Comparison', color=COLORS['text'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M{i+1}' for i in range(n_mfcc)], color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.legend(loc='upper right', facecolor=COLORS['surface'], edgecolor=COLORS['border'], labelcolor=COLORS['text'])
    ax.grid(axis='y', alpha=0.1, color=COLORS['text'])

    return _fig_to_base64(fig)


def plot_spectral_radar(file_features, dialect_avg_features):
    """Radar chart of spectral features with each dialect overlaid."""
    feature_keys = [
        'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean',
        'zcr_mean', 'rms_mean', 'spectral_flatness_mean',
        'spectral_contrast_1_mean', 'spectral_contrast_4_mean'
    ]
    feature_labels = [
        'Centroid', 'Bandwidth', 'Rolloff', 'ZCR', 'RMS', 'Flatness', 'Contrast L', 'Contrast H'
    ]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    angles = np.linspace(0, 2 * np.pi, len(feature_keys), endpoint=False).tolist()
    angles += angles[:1]

    def normalize_values(vals, all_vals_list):
        min_v = min(min(v) for v in all_vals_list if len(v) > 0)
        max_v = max(max(v) for v in all_vals_list if len(v) > 0)
        rng = max_v - min_v if max_v != min_v else 1
        return [(v - min_v) / rng for v in vals]

    all_raw = []
    file_vals = [file_features.get(k, 0) for k in feature_keys]
    all_raw.append(file_vals)
    for dialect in DIALECT_COLORS:
        if dialect in dialect_avg_features:
            all_raw.append([dialect_avg_features[dialect].get(k, 0) for k in feature_keys])

    # Plot file
    vals = normalize_values(file_vals, all_raw) + [normalize_values(file_vals, all_raw)[0]]
    ax.plot(angles, vals, 'o-', linewidth=2.5, label='Uploaded File', color='#E2E8F0')
    ax.fill(angles, vals, alpha=0.1, color='#E2E8F0')

    # Plot dialects
    for dialect, color in DIALECT_COLORS.items():
        if dialect in dialect_avg_features:
            d_vals = [dialect_avg_features[dialect].get(k, 0) for k in feature_keys]
            d_norm = normalize_values(d_vals, all_raw) + [normalize_values(d_vals, all_raw)[0]]
            ax.plot(angles, d_norm, 'o-', linewidth=1.5, label=DIALECT_LABELS[dialect], color=color, alpha=0.7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, color=COLORS['text'], fontsize=9)
    ax.tick_params(colors=COLORS['text'])
    ax.set_title('Spectral Feature Radar', color=COLORS['text'], fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor=COLORS['surface'],
              edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
    ax.grid(color=COLORS['border'], alpha=0.3)

    return _fig_to_base64(fig)


def plot_feature_importance(importances, feature_names, top_n=15):
    """Horizontal bar chart of top feature importances from the classifier."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    colors = []
    for i in indices:
        name = feature_names[i]
        if 'mfcc' in name:
            colors.append(COLORS['primary'])
        elif 'spectral' in name:
            colors.append(COLORS['secondary'])
        elif 'pitch' in name or 'f0' in name:
            colors.append(COLORS['accent'])
        else:
            colors.append('#F59E0B')

    ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor=COLORS['border'])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], color=COLORS['text'], fontsize=9)
    ax.set_xlabel('Importance', color=COLORS['text'], fontsize=12)
    ax.set_title('Top Feature Importances (Random Forest)', color=COLORS['text'], fontsize=14, fontweight='bold')
    ax.tick_params(colors=COLORS['text'])
    ax.grid(axis='x', alpha=0.1, color=COLORS['text'])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], label='MFCC'),
        Patch(facecolor=COLORS['secondary'], label='Spectral'),
        Patch(facecolor=COLORS['accent'], label='Pitch'),
        Patch(facecolor='#F59E0B', label='Other'),
    ]
    ax.legend(handles=legend_elements, facecolor=COLORS['surface'], edgecolor=COLORS['border'],
              labelcolor=COLORS['text'], loc='lower right')

    return _fig_to_base64(fig)


def plot_pitch_contour(y, sr, dialect_pitch_ranges=None):
    """Pitch contour with dialect range overlays."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    import librosa
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    times = librosa.times_like(f0, sr=sr)

    ax.plot(times, f0, color=COLORS['text'], linewidth=1.5, label='File Pitch', alpha=0.9)

    if dialect_pitch_ranges:
        for dialect, (low, high) in dialect_pitch_ranges.items():
            color = DIALECT_COLORS.get(dialect, '#888')
            ax.axhspan(low, high, alpha=0.15, color=color, label=f'{DIALECT_LABELS.get(dialect, dialect)} range')

    ax.set_xlabel('Time (s)', color=COLORS['text'], fontsize=12)
    ax.set_ylabel('Frequency (Hz)', color=COLORS['text'], fontsize=12)
    ax.set_title('Pitch (F0) Contour', color=COLORS['text'], fontsize=14, fontweight='bold')
    ax.tick_params(colors=COLORS['text'])
    ax.legend(facecolor=COLORS['surface'], edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
    ax.grid(alpha=0.1, color=COLORS['text'])

    return _fig_to_base64(fig)


def plot_dialect_probabilities(probabilities):
    """Bar chart of classification probabilities for each dialect."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    dialects = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [DIALECT_COLORS.get(d, '#888') for d in dialects]
    labels = [DIALECT_LABELS.get(d, d) for d in dialects]

    bars = ax.barh(labels, probs, color=colors, edgecolor=COLORS['border'], height=0.5)

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', color=COLORS['text'], fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Probability', color=COLORS['text'], fontsize=12)
    ax.set_title('Dialect Classification Probabilities', color=COLORS['text'], fontsize=14, fontweight='bold')
    ax.tick_params(colors=COLORS['text'])
    ax.grid(axis='x', alpha=0.1, color=COLORS['text'])

    return _fig_to_base64(fig)
