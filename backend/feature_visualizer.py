"""
Feature visualization module.
Generates comparison charts showing how features distinguish dialects.
All plots return base64-encoded PNG strings.
"""
import io, os, base64, glob
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Theme ─────────────────────────────────────────────────────────────────────
COLORS = {
    'bg': '#1A1410', 'surface': '#2A201A', 'primary': '#7C3AED',
    'secondary': '#C75B39', 'accent': '#7A9E7E', 'text': '#F5E6D3',
    'border': '#3E342C',
}
DIALECT_COLORS = {
    'egyptian': '#7C3AED', 'gulf': '#C75B39',
    'levantine': '#7A9E7E', 'maghrebi': '#E8A838',
}
DIALECT_LABELS = {
    'egyptian': 'Egyptian (مصري)', 'gulf': 'Gulf (خليجي)',
    'levantine': 'Levantine (شامي)', 'maghrebi': 'Maghrebi (مغربي)',
}

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=COLORS['bg'], dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── Reference-sample helpers ─────────────────────────────────────────────────
_dialect_avg_cache = None

def _extract_simple_features(y, sr):
    """Lightweight feature extraction for reference averaging."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    f0, _, _ = librosa.pyin(y, fmin=75, fmax=500, sr=sr)
    f0v = f0[np.isfinite(f0) & (f0 > 0)] if f0 is not None else np.array([0.0])
    if len(f0v) == 0:
        f0v = np.array([0.0])

    # rhythm
    rms_env = librosa.feature.rms(y=y, frame_length=2048, hop_length=256)[0]
    env_smooth = np.convolve(rms_env, np.ones(5)/5, mode='same')
    peaks = sum(1 for i in range(1, len(env_smooth)-1)
                if env_smooth[i] > env_smooth[i-1] and env_smooth[i] > env_smooth[i+1]
                and env_smooth[i] > np.max(env_smooth)*0.3)
    dur = len(y) / sr
    speech_rate = peaks / max(dur, 0.01)
    thresh = np.percentile(rms_env, 20)
    pause_ratio = float(np.mean(rms_env < thresh))
    zcr2 = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=256)[0]
    voiced_ratio = float(np.mean(zcr2 < np.median(zcr2)))

    return {
        'mfcc_mean': np.mean(mfcc, axis=1),
        'mfcc_std': np.std(mfcc, axis=1),
        'delta_mean': np.mean(delta, axis=1),
        'delta_std': np.std(delta, axis=1),
        'delta2_mean': np.mean(delta2, axis=1),
        'delta2_std': np.std(delta2, axis=1),
        'spectral_centroid': float(np.mean(cent)),
        'spectral_bandwidth': float(np.mean(bw)),
        'spectral_rolloff': float(np.mean(rolloff)),
        'spectral_flatness': float(np.mean(flatness)),
        'rms': float(np.mean(rms)),
        'zcr': float(np.mean(zcr)),
        'contrast_mean': np.mean(contrast, axis=1),
        'chroma_mean': np.mean(chroma, axis=1),
        'pitch_mean': float(np.mean(f0v)),
        'pitch_std': float(np.std(f0v)),
        'pitch_min': float(np.min(f0v)),
        'pitch_max': float(np.max(f0v)),
        'speech_rate': speech_rate,
        'pause_ratio': pause_ratio,
        'voiced_ratio': voiced_ratio,
        'energy_mean_db': float(np.mean(20*np.log10(rms_env + 1e-12))),
    }


def get_dialect_averages():
    """Return dialect average features from the TRAINING DATA.
    Feature names are mapped to audio_processor keys (0-indexed)."""
    global _dialect_avg_cache
    if _dialect_avg_cache is not None:
        return _dialect_avg_cache

    # Training data averages (from the ML training set)
    # Keys: mfcc_mean_0..12, mfcc_std_0..12, delta_mfcc_mean/std, delta2_mfcc_mean/std,
    #        spectral_contrast_mean, zero_crossing_rate_mean/std, rms_energy_mean/std
    _train = {
        'egyptian': {
            'mfcc_mean':  [-234.9877,101.128,-21.6215,15.0299,-17.9016,-8.1176,-21.4191,-6.0648,-15.9118,-3.0094,-12.3309,-4.2432,-9.7069],
            'mfcc_std':   [100.5588,39.7021,35.433,27.3805,21.7444,17.5728,16.9629,14.9841,14.2484,12.9664,11.9701,11.0635,11.0833],
            'delta_mean': [-0.0518,-0.0181,0.0276,0.0109,0.0064,0.002,-0.0027,0.007,0.0055,-0.0052,0.0001,0.0001,-0.0004],
            'delta_std':  [16.4744,7.3881,6.6635,5.0205,3.9141,3.2375,3.0496,2.7687,2.549,2.4162,2.1637,2.039,1.9877],
            'delta2_mean':[-0.1358,-0.0485,0.0376,0.0049,0.0273,0.0219,0.0236,0.0144,0.0254,0.009,0.0143,0.012,0.01],
            'delta2_std': [9.1672,4.6654,4.1057,2.958,2.4057,2.0338,1.9066,1.7874,1.5938,1.5656,1.3845,1.3493,1.2758],
            'zcr_mean': 0.1258, 'zcr_std': 0.0709,
            'rms_mean': 0.1104, 'rms_std': 0.0686,
            'contrast_mean': [22.0784,18.1149,20.8775,20.2686,21.8017,23.4419,30.4623],
            'contrast_std':  [5.6873,5.3751,5.401,4.6907,4.5038,5.6757,7.1189],
            'chroma_mean': [0.336,0.3378,0.344,0.3404,0.3363,0.3412,0.3528,0.3614,0.3578,0.3526,0.343,0.3377],
            'chroma_std':  [0.3024,0.2998,0.3028,0.3003,0.2999,0.3018,0.3063,0.3101,0.309,0.3067,0.3028,0.3012],
        },
        'gulf': {
            'mfcc_mean':  [-284.973,102.5619,-23.4555,15.9037,-19.6683,-12.9806,-19.1593,-10.5355,-14.3233,-2.8106,-11.92,-3.0584,-8.7438],
            'mfcc_std':   [98.2338,44.5599,40.0682,30.948,23.3252,20.4052,18.8184,16.9219,15.5387,13.9244,12.7697,11.8405,11.8973],
            'delta_mean': [-0.1851,-0.0296,0.0704,0.0036,0.0266,0.0032,0.0001,0.0067,0.0122,0.0,0.0023,0.0072,0.0027],
            'delta_std':  [16.6984,8.2636,7.63,5.9103,4.3685,3.777,3.4359,3.121,2.8694,2.6126,2.3308,2.1988,2.1806],
            'delta2_mean':[-0.0759,-0.0031,0.0321,-0.002,0.0274,0.0122,0.0139,0.0129,0.0154,0.0059,0.008,0.0076,0.0107],
            'delta2_std': [9.6494,5.1494,4.8529,3.4816,2.7154,2.3537,2.1458,1.9961,1.7979,1.6825,1.5003,1.4617,1.4003],
            'zcr_mean': 0.1273, 'zcr_std': 0.0748,
            'rms_mean': 0.0786, 'rms_std': 0.0471,
            'contrast_mean': [22.3925,18.7246,21.532,20.9534,22.0396,25.2578,29.3417],
            'contrast_std':  [5.7304,5.6956,5.7659,5.013,4.7776,6.396,6.4921],
            'chroma_mean': [0.3344,0.3297,0.3285,0.3249,0.3314,0.3372,0.336,0.3355,0.3349,0.3427,0.3428,0.3408],
            'chroma_std':  [0.3091,0.3058,0.3043,0.3027,0.3057,0.3078,0.3076,0.3105,0.3087,0.3127,0.311,0.3113],
        },
        'levantine': {
            'mfcc_mean':  [-268.0652,103.4429,-15.8274,14.0133,-22.9134,-9.932,-21.933,-10.2245,-14.8519,-2.2499,-12.8191,-3.8867,-8.883],
            'mfcc_std':   [94.0085,39.9631,36.8813,29.3817,22.1889,18.4578,17.3239,15.0617,14.2726,12.8043,12.2289,11.3598,11.0364],
            'delta_mean': [-0.0691,0.0093,0.0003,-0.013,0.0038,-0.0083,0.0008,-0.0102,0.0052,0.0003,-0.0009,-0.0037,0.0022],
            'delta_std':  [15.0854,7.4351,6.8568,5.4213,3.9646,3.4127,3.1051,2.7612,2.5792,2.3584,2.1946,2.0777,1.9935],
            'delta2_mean':[-0.0879,-0.0185,0.0223,-0.0051,0.0276,0.0141,0.018,0.01,0.0125,0.0035,0.0111,0.0077,0.01],
            'delta2_std': [8.4071,4.6326,4.1989,3.1334,2.4414,2.1259,1.9145,1.7657,1.5973,1.5112,1.3872,1.3499,1.2735],
            'zcr_mean': 0.1206, 'zcr_std': 0.0706,
            'rms_mean': 0.0864, 'rms_std': 0.0532,
            'contrast_mean': [21.85,18.2184,21.0739,20.7763,21.715,24.5399,29.8049],
            'contrast_std':  [5.5924,5.2966,5.3822,4.7586,4.515,6.005,5.7851],
            'chroma_mean': [0.3151,0.3201,0.3204,0.3188,0.3193,0.3248,0.3299,0.3336,0.3302,0.3231,0.3227,0.3213],
            'chroma_std':  [0.3007,0.3045,0.3052,0.3032,0.3023,0.3045,0.3072,0.3099,0.3085,0.304,0.3048,0.305],
        },
        'maghrebi': {
            'mfcc_mean':  [-265.6686,91.8497,-7.8529,28.165,-10.7863,-0.596,-18.702,-5.7149,-15.9381,-0.8488,-12.9913,-1.2084,-10.6949],
            'mfcc_std':   [97.709,48.0911,35.7779,31.5306,21.4524,18.692,18.1513,15.6957,14.9407,13.6871,12.7163,11.7363,11.7757],
            'delta_mean': [0.0253,-0.0101,-0.0093,0.0042,-0.0005,0.0038,0.0003,-0.0014,0.0037,0.0027,-0.0045,0.001,0.0002],
            'delta_std':  [16.9761,9.2596,6.8194,5.898,3.9648,3.4779,3.3312,2.9088,2.7005,2.5718,2.314,2.1682,2.1396],
            'delta2_mean':[-0.2007,-0.061,-0.0018,-0.017,0.0169,0.009,0.0132,0.0072,0.0171,0.0043,0.0061,0.0064,0.012],
            'delta2_std': [10.0669,5.8978,4.4105,3.4708,2.5614,2.24,2.1299,1.9032,1.6974,1.6684,1.4882,1.4353,1.3695],
            'zcr_mean': 0.1308, 'zcr_std': 0.0885,
            'rms_mean': 0.0714, 'rms_std': 0.0387,
            'contrast_mean': [22.6067,17.3889,20.5253,18.9188,20.166,22.4722,33.8444],
            'contrast_std':  [5.6084,5.2149,5.2701,4.0652,3.9839,5.2717,6.1455],
            'chroma_mean': [0.3063,0.3091,0.315,0.3227,0.3335,0.3379,0.3473,0.3443,0.341,0.3322,0.3114,0.3109],
            'chroma_std':  [0.3012,0.3027,0.3055,0.3056,0.3115,0.3115,0.3154,0.315,0.3158,0.3152,0.3035,0.3064],
        },
    }

    # Flatten arrays into audio_processor-style keys (0-indexed)
    averages = {}
    for dialect, raw in _train.items():
        flat = {}
        for i in range(13):
            flat[f'mfcc_mean_{i}'] = raw['mfcc_mean'][i]
            flat[f'mfcc_std_{i}'] = raw['mfcc_std'][i]
            flat[f'delta_mfcc_mean_{i}'] = raw['delta_mean'][i]
            flat[f'delta_mfcc_std_{i}'] = raw['delta_std'][i]
            flat[f'delta2_mfcc_mean_{i}'] = raw['delta2_mean'][i]
            flat[f'delta2_mfcc_std_{i}'] = raw['delta2_std'][i]
        flat['zero_crossing_rate_mean'] = raw['zcr_mean']
        flat['zero_crossing_rate_std'] = raw['zcr_std']
        flat['rms_energy_mean'] = raw['rms_mean']
        flat['rms_energy_std'] = raw['rms_std']
        flat['spectral_contrast_mean'] = float(np.mean(raw['contrast_mean']))
        flat['spectral_contrast_std'] = float(np.mean(raw['contrast_std']))
        # Per-band spectral contrast (7 bands)
        for i in range(7):
            flat[f'contrast_mean_{i}'] = raw['contrast_mean'][i]
            flat[f'contrast_std_{i}'] = raw['contrast_std'][i]
        # Chroma features (12 pitch classes)
        for i in range(12):
            flat[f'chroma_mean_{i}'] = raw['chroma_mean'][i]
            flat[f'chroma_std_{i}'] = raw['chroma_std'][i]
        averages[dialect] = flat

    _dialect_avg_cache = averages
    print(f"[OK] Dialect averages loaded from training data for: {list(averages.keys())}")
    return averages


def extract_extra_features(y, sr):
    """Extract chroma and per-band spectral contrast from audio.
    Returns a dict with keys matching the training data format."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    extra = {}
    for i in range(12):
        extra[f'chroma_mean_{i}'] = float(np.mean(chroma[i]))
        extra[f'chroma_std_{i}'] = float(np.std(chroma[i]))
    for i in range(min(7, contrast.shape[0])):
        extra[f'contrast_mean_{i}'] = float(np.mean(contrast[i]))
        extra[f'contrast_std_{i}'] = float(np.std(contrast[i]))
    return extra


# ── Similarity computation (uses training data as reference) ─────────────────
def compute_dialect_similarity(file_feats, dialect_avgs):
    """Compute per-category and overall similarity (0-100%) comparing file
    features against training data averages, weighted by XGBoost feature importances."""
    import joblib
    import os
    
    # Try to load XGBoost importances to weight the distances (Option 2)
    importances = {}
    try:
        model_path = os.path.join(os.path.dirname(__file__), "models", "arabic_dialect_xgboost_model.pkl")
        model = joblib.load(model_path)
        fi = model.steps[-1][1].feature_importances_
        
        for i in range(20):
            importances[f'mfcc_mean_{i}'] = fi[i]
            importances[f'mfcc_std_{i}'] = fi[20 + i]
            importances[f'delta_mfcc_mean_{i}'] = fi[40 + i]
            importances[f'delta_mfcc_std_{i}'] = fi[60 + i]
            importances[f'delta2_mfcc_mean_{i}'] = fi[80 + i]
            importances[f'delta2_mfcc_std_{i}'] = fi[100 + i]
        for i in range(12):
            importances[f'chroma_mean_{i}'] = fi[120 + i]
            importances[f'chroma_std_{i}'] = fi[132 + i]
        for i in range(7):
            importances[f'contrast_mean_{i}'] = fi[144 + i]
            importances[f'contrast_std_{i}'] = fi[151 + i]
    except Exception:
        pass # fallback to unweighted if model fails to load

    categories = {
        'Vocal Tract (MFCC)': [f'mfcc_mean_{i}' for i in range(13)]
                             + [f'mfcc_std_{i}' for i in range(13)],
        'Speech Dynamics (Delta)': [f'delta_mfcc_mean_{i}' for i in range(13)]
                                  + [f'delta_mfcc_std_{i}' for i in range(13)]
                                  + [f'delta2_mfcc_mean_{i}' for i in range(13)]
                                  + [f'delta2_mfcc_std_{i}' for i in range(13)],
        'Spectral Shape': ['spectral_contrast_mean', 'spectral_contrast_std',
                           'rms_energy_mean', 'rms_energy_std',
                           'zero_crossing_rate_mean', 'zero_crossing_rate_std'],
        'Tonal Profile (Chroma)': [f'chroma_mean_{i}' for i in range(12)]
                                 + [f'chroma_std_{i}' for i in range(12)],
        'Band Contrast': [f'contrast_mean_{i}' for i in range(7)]
                        + [f'contrast_std_{i}' for i in range(7)],
    }
    
    results = {}
    for dialect, avg in dialect_avgs.items():
        cat_scores = {}
        all_dists = []
        all_weights = []
        
        for cat_name, keys in categories.items():
            dists = []
            weights = []
            for k in keys:
                if k not in avg or k not in file_feats:
                    continue
                fv = float(file_feats.get(k, 0))
                av = float(avg.get(k, 0))
                d = abs(fv - av)
                scale = abs(av) + 1e-8
                
                # Weight by importance, default to 1.0 if not found
                w = importances.get(k, 1.0)
                
                dists.append(min(d / scale, 1.0) * w)
                weights.append(w)
                
            if dists and sum(weights) > 0:
                # Weighted average distance
                avg_dist = sum(dists) / sum(weights)
                score = max(0, 100 * (1 - avg_dist))
            else:
                score = 0
            cat_scores[cat_name] = round(score, 1)
            all_dists.extend(dists)
            all_weights.extend(weights)
            
        if all_dists and sum(all_weights) > 0:
            overall_dist = sum(all_dists) / sum(all_weights)
            cat_scores['overall'] = round(max(0, 100 * (1 - overall_dist)), 1)
        else:
            cat_scores['overall'] = 0
            
        results[dialect] = cat_scores
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 0A: Dialect Similarity Dashboard  (THE KEY PLOT)
# ══════════════════════════════════════════════════════════════════════════════
def plot_dialect_similarity(file_feats, dialect_avgs):
    """Big horizontal gauges showing % similarity to each dialect — the user
    can immediately see which dialect the audio matches."""
    sim = compute_dialect_similarity(file_feats, dialect_avgs)
    dialects = sorted(sim.keys(), key=lambda d: sim[d]['overall'], reverse=True)
    best = dialects[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    y_positions = list(range(len(dialects)))[::-1]
    for i, dialect in enumerate(dialects):
        yp = y_positions[i]
        score = sim[dialect]['overall']
        color = DIALECT_COLORS.get(dialect, '#888')
        label = DIALECT_LABELS.get(dialect, dialect)

        # Background track
        ax.barh(yp, 100, height=0.6, color=COLORS['surface'], edgecolor=COLORS['border'])
        # Fill bar
        bar_alpha = 1.0 if dialect == best else 0.6
        ax.barh(yp, score, height=0.6, color=color, alpha=bar_alpha, edgecolor='none')
        # Label on left
        ax.text(-2, yp, label, ha='right', va='center', color=color,
                fontsize=12, fontweight='bold' if dialect == best else 'normal')
        # Percentage on bar
        ax.text(score + 1.5, yp, f'{score:.0f}%', ha='left', va='center',
                color=COLORS['text'], fontsize=14, fontweight='bold')
        # Star for best match
        if dialect == best:
            ax.text(score + 10, yp, '  BEST MATCH', ha='left', va='center',
                    color='#4ADE80', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 115)
    ax.set_ylim(-0.5, len(dialects) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color=COLORS['text'], fontsize=10)
    ax.tick_params(colors=COLORS['text'])
    ax.set_title('How closely does your audio match each dialect?',
                 color=COLORS['text'], fontsize=15, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['border'])
    ax.grid(axis='x', alpha=0.15, color=COLORS['text'])
    fig.tight_layout()
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 0B: Feature Category Breakdown Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def get_162_feature_names():
    names = []
    for f_name, size in [("MFCC", 20), ("Delta", 20), ("Delta²", 20), ("Chroma", 12), ("Contrast", 7), ("ZCR", 1), ("RMS", 1)]:
        for i in range(size):
            names.append(f"{f_name} {i+1} Mean")
        for i in range(size):
            names.append(f"{f_name} {i+1} Std")
    return names

def plot_feature_breakdown(arr_162, file_feats, dialect_avgs):
    """Dual-plot: Left=SHAP Top Features, Right=Category Heatmap (SHAP driven)."""
    import joblib
    import shap
    import os

    dialects = ['egyptian', 'gulf', 'levantine', 'maghrebi']
    categories = ['Vocal Tract (MFCC)', 'Speech Dynamics (Delta)', 'Tonal Profile (Chroma)', 'Spectral Shape & Contrast']
    
    # Fallback data if SHAP fails
    data = np.zeros((4, 4))
    top_5_names = []
    top_5_vals = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['bg'])

    try:
        model_path = os.path.join(os.path.dirname(__file__), "models", "arabic_dialect_xgboost_model.pkl")
        model = joblib.load(model_path)
        clf = model.steps[-1][1]

        transformed_input = arr_162.reshape(1, -1)
        for name, step in model.steps[:-1]:
            transformed_input = step.transform(transformed_input)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(transformed_input) # (1, 162, 4)

        # Get predicted class
        pred_num = model.predict(arr_162.reshape(1, -1))[0]
        shap_for_class = shap_values[0, :, pred_num]

        # --- PLOT 1 DATA ---
        top_5_idx = np.argsort(np.abs(shap_for_class))[-7:]
        feature_names = get_162_feature_names()
        top_5_names = [feature_names[i] for i in top_5_idx]
        top_5_vals = shap_for_class[top_5_idx]

        # --- PLOT 2 DATA: SHAP Category Impact ---
        categories_dict = {
            'Vocal Tract (MFCC)': list(range(0, 40)),
            'Speech Dynamics (Delta)': list(range(40, 120)),
            'Tonal Profile (Chroma)': list(range(120, 144)),
            'Spectral Shape & Contrast': list(range(144, 158)) + list(range(158, 162))
        }
        
        heatmap_data = []
        for cat_name in categories:
            indices = categories_dict[cat_name]
            cat_impacts = []
            for d_idx in range(4): # 4 dialects
                impact = np.sum(shap_values[0, indices, d_idx])
                cat_impacts.append(impact)
                
            # Softmax
            exp_impacts = np.exp(cat_impacts - np.max(cat_impacts))
            softmax = exp_impacts / np.sum(exp_impacts)
            percentages = [round(float(p) * 100) for p in softmax]
            heatmap_data.append(percentages)
            
        data = np.array(heatmap_data)

        # --- PLOT 1 RENDERING ---
        ax1.set_facecolor(COLORS['bg'])
        y_pos = np.arange(len(top_5_names))
        colors = ['#4ADE80' if v > 0 else '#EF4444' for v in top_5_vals]
        ax1.barh(y_pos, top_5_vals, color=colors, height=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_5_names, color=COLORS['text'], fontsize=11, fontweight='bold')
        ax1.tick_params(colors=COLORS['text'])
        ax1.set_title("Top 7 Features Driving Prediction (SHAP)", color=COLORS['text'], fontsize=14, fontweight='bold', pad=12)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_color(COLORS['border'])
        
    except Exception as e:
        print(f"SHAP failed: {e}")
        ax1.text(0.5, 0.5, "SHAP Explainability Unavailable", color=COLORS['text'], ha='center', va='center')
        ax1.axis('off')

    # --- PLOT 2 RENDERING ---
    ax2.set_facecolor(COLORS['surface'])
    im = ax2.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    for i in range(len(categories)):
        best_j = np.argmax(data[i])
        for j in range(len(dialects)):
            val = data[i, j]
            weight = 'bold' if j == best_j else 'normal'
            txt_color = 'white' if val < 50 else 'black'
            ax2.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    color=txt_color, fontsize=12, fontweight=weight)

    ax2.set_xticks(range(len(dialects)))
    ax2.set_xticklabels([DIALECT_LABELS[d].split(' (')[0] for d in dialects],
                       color=COLORS['text'], fontsize=11, fontweight='bold')
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels(categories, color=COLORS['text'], fontsize=10)
    ax2.tick_params(colors=COLORS['text'], length=0)
    ax2.set_title('Feature Category Impact (%)',
                 color=COLORS['text'], fontsize=14, fontweight='bold', pad=12)

    fig.tight_layout()
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: MFCC Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_mfcc_heatmap(y, sr):
    """Heatmap of MFCC coefficients over time — the acoustic fingerprint."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])
    img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+.1f')
    ax.set_ylabel('MFCC Coefficient', color=COLORS['text'], fontsize=11)
    ax.set_title('MFCC Heatmap — Acoustic Fingerprint', color=COLORS['text'], fontsize=13, fontweight='bold')
    ax.tick_params(colors=COLORS['text'])
    ax.xaxis.label.set_color(COLORS['text'])
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: MFCC Comparison Bars
# ══════════════════════════════════════════════════════════════════════════════
def plot_mfcc_comparison(file_feats, dialect_avgs):
    """Grouped bar chart: uploaded file MFCC means vs each dialect average."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    n = 13
    x = np.arange(n)
    width = 0.15
    file_mfccs = file_feats['mfcc_mean']
    ax.bar(x - 2*width, file_mfccs, width, label='Your File', color='#E2E8F0', edgecolor='#CBD5E1', alpha=0.95)

    for idx, (dialect, color) in enumerate(DIALECT_COLORS.items()):
        if dialect in dialect_avgs:
            ax.bar(x + (idx-1)*width, dialect_avgs[dialect]['mfcc_mean'], width,
                   label=DIALECT_LABELS[dialect], color=color, alpha=0.8)

    ax.set_xlabel('MFCC Coefficient', color=COLORS['text'], fontsize=11)
    ax.set_ylabel('Mean Value', color=COLORS['text'], fontsize=11)
    ax.set_title('MFCC Profile — Your File vs Dialect Averages', color=COLORS['text'], fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M{i+1}' for i in range(n)], color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.legend(facecolor=COLORS['surface'], edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
    ax.grid(axis='y', alpha=0.1, color=COLORS['text'])
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Spectral Radar
# ══════════════════════════════════════════════════════════════════════════════
def plot_spectral_radar(file_feats, dialect_avgs):
    """Radar chart of spectral features overlaid per dialect."""
    keys = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            'zcr', 'rms', 'spectral_flatness']
    labels = ['Centroid', 'Bandwidth', 'Rolloff', 'ZCR', 'RMS', 'Flatness']

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    angles = np.linspace(0, 2*np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]

    # Gather all values for normalization
    all_vals = []
    file_vals = [file_feats.get(k, 0) for k in keys]
    all_vals.append(file_vals)
    for d in DIALECT_COLORS:
        if d in dialect_avgs:
            all_vals.append([dialect_avgs[d].get(k, 0) for k in keys])

    mn = np.min(all_vals)
    mx = np.max(all_vals)
    rng = mx - mn if mx != mn else 1

    def norm(v):
        return [(x - mn)/rng for x in v]

    fv = norm(file_vals) + [norm(file_vals)[0]]
    ax.plot(angles, fv, 'o-', lw=2.5, label='Your File', color='#E2E8F0')
    ax.fill(angles, fv, alpha=0.12, color='#E2E8F0')

    for dialect, color in DIALECT_COLORS.items():
        if dialect in dialect_avgs:
            dv = norm([dialect_avgs[dialect].get(k, 0) for k in keys])
            dv += [dv[0]]
            ax.plot(angles, dv, 'o-', lw=1.5, label=DIALECT_LABELS[dialect], color=color, alpha=0.7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=COLORS['text'], fontsize=9)
    ax.tick_params(colors=COLORS['text'])
    ax.set_title('Spectral Feature Radar', color=COLORS['text'], fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), facecolor=COLORS['surface'],
              edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
    ax.grid(color=COLORS['border'], alpha=0.3)
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Pitch Contour
# ══════════════════════════════════════════════════════════════════════════════
def plot_pitch_contour(y, sr, dialect_avgs=None):
    """Pitch F0 contour with dialect average pitch ranges shaded."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    f0, _, _ = librosa.pyin(y, fmin=75, fmax=500, sr=sr)
    times = librosa.times_like(f0, sr=sr)
    ax.plot(times, f0, color='#E2E8F0', lw=1.5, label='Your File', alpha=0.9)

    if dialect_avgs:
        for dialect, color in DIALECT_COLORS.items():
            if dialect in dialect_avgs:
                avg = dialect_avgs[dialect]
                lo = avg.get('pitch_mean', 0) - avg.get('pitch_std', 0)
                hi = avg.get('pitch_mean', 0) + avg.get('pitch_std', 0)
                if lo > 0 and hi > 0:
                    ax.axhspan(lo, hi, alpha=0.12, color=color,
                               label=f'{DIALECT_LABELS[dialect]} μ±σ')

    ax.set_xlabel('Time (s)', color=COLORS['text'], fontsize=11)
    ax.set_ylabel('F0 (Hz)', color=COLORS['text'], fontsize=11)
    ax.set_title('Pitch (F0) Contour with Dialect Ranges', color=COLORS['text'], fontsize=13, fontweight='bold')
    ax.tick_params(colors=COLORS['text'])
    ax.legend(facecolor=COLORS['surface'], edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
    ax.grid(alpha=0.1, color=COLORS['text'])
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Energy & Rhythm
# ══════════════════════════════════════════════════════════════════════════════
def plot_energy_rhythm(file_feats, dialect_avgs):
    """Grouped bars comparing rhythm/energy features across dialects."""
    keys = ['speech_rate', 'pause_ratio', 'voiced_ratio', 'energy_mean_db']
    labels = ['Speech Rate', 'Pause Ratio', 'Voiced Ratio', 'Energy (dB)']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor(COLORS['bg'])

    for i, (key, label) in enumerate(zip(keys, labels)):
        ax = axes[i]
        ax.set_facecolor(COLORS['surface'])
        items = [('Your File', '#E2E8F0', file_feats.get(key, 0))]
        for dialect, color in DIALECT_COLORS.items():
            if dialect in dialect_avgs:
                items.append((DIALECT_LABELS[dialect].split(' (')[0], color, dialect_avgs[dialect].get(key, 0)))

        names = [it[0] for it in items]
        colors = [it[1] for it in items]
        vals = [it[2] for it in items]
        bars = ax.bar(range(len(items)), vals, color=colors, edgecolor=COLORS['border'], width=0.6)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels(names, color=COLORS['text'], fontsize=7, rotation=30, ha='right')
        ax.set_title(label, color=COLORS['text'], fontsize=10, fontweight='bold')
        ax.tick_params(colors=COLORS['text'], labelsize=8)
        ax.grid(axis='y', alpha=0.1, color=COLORS['text'])

    fig.suptitle('Energy & Rhythm Comparison', color=COLORS['text'], fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Formant Scatter (F1 vs F2)
# ══════════════════════════════════════════════════════════════════════════════
def plot_formant_scatter(y, sr, dialect_avgs):
    """F1 vs F2 vowel space scatter — file point vs dialect average ellipses."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])

    # Try to get formants for uploaded file
    try:
        import parselmouth
        import soundfile as sf, tempfile
        from pathlib import Path
        yn = y / (np.max(np.abs(y)) + 1e-12)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
            tmp = Path(tf.name)
        sf.write(tmp, yn.astype(np.float64), sr, subtype='PCM_16')
        snd = parselmouth.Sound(str(tmp))
        formant = snd.to_formant_burg()
        times = np.arange(0, snd.duration, 0.01)
        f1s, f2s = [], []
        for t in times:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            if np.isfinite(f1) and np.isfinite(f2) and f1 > 0 and f2 > 0:
                f1s.append(f1)
                f2s.append(f2)
        tmp.unlink(missing_ok=True)
        if f1s:
            ax.scatter(f2s, f1s, c='#E2E8F0', alpha=0.3, s=10, label='Your File')
            ax.scatter([np.mean(f2s)], [np.mean(f1s)], c='#E2E8F0', s=120, marker='*',
                       edgecolors='white', zorder=5)
    except Exception:
        pass

    # Dialect reference points (using pitch as proxy if formants unavailable)
    for dialect, color in DIALECT_COLORS.items():
        if dialect in dialect_avgs:
            avg = dialect_avgs[dialect]
            sc = avg.get('spectral_centroid', 1000)
            bw = avg.get('spectral_bandwidth', 500)
            ax.scatter([sc], [bw], c=color, s=150, marker='D', edgecolors='white',
                       label=DIALECT_LABELS[dialect], zorder=5)

    ax.set_xlabel('F2 / Spectral Centroid (Hz)', color=COLORS['text'], fontsize=11)
    ax.set_ylabel('F1 / Spectral Bandwidth (Hz)', color=COLORS['text'], fontsize=11)
    ax.set_title('Vowel Space / Spectral Position', color=COLORS['text'], fontsize=13, fontweight='bold')
    ax.tick_params(colors=COLORS['text'])
    ax.legend(facecolor=COLORS['surface'], edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
    ax.grid(alpha=0.1, color=COLORS['text'])
    ax.invert_xaxis()
    ax.invert_yaxis()
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7: Delta / Delta² MFCC Comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_delta_comparison(file_feats, dialect_avgs):
    """Grouped bars comparing Delta and Delta² MFCC standard deviations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['bg'])

    for ax, key, title in [(ax1, 'delta_std', 'Delta MFCC (σ) — Speech Dynamics'),
                            (ax2, 'delta2_std', 'Delta² MFCC (σ) — Articulation Acceleration')]:
        ax.set_facecolor(COLORS['surface'])
        n = 13
        x = np.arange(n)
        width = 0.15
        ax.bar(x - 2*width, file_feats[key], width, label='Your File', color='#E2E8F0', alpha=0.95)
        for idx, (dialect, color) in enumerate(DIALECT_COLORS.items()):
            if dialect in dialect_avgs:
                ax.bar(x + (idx-1)*width, dialect_avgs[dialect][key], width,
                       label=DIALECT_LABELS[dialect], color=color, alpha=0.8)
        ax.set_xlabel('Coefficient', color=COLORS['text'], fontsize=10)
        ax.set_ylabel('Std Dev', color=COLORS['text'], fontsize=10)
        ax.set_title(title, color=COLORS['text'], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i+1) for i in range(n)], color=COLORS['text'], fontsize=8)
        ax.tick_params(colors=COLORS['text'])
        ax.grid(axis='y', alpha=0.1, color=COLORS['text'])

    ax1.legend(facecolor=COLORS['surface'], edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8: Chromagram
# ══════════════════════════════════════════════════════════════════════════════
def plot_chromagram(y, sr):
    """Chromagram showing pitch class energy distribution over time."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['surface'])
    img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax)
    ax.set_title('Chromagram — Pitch Class Energy', color=COLORS['text'], fontsize=13, fontweight='bold')
    ax.tick_params(colors=COLORS['text'])
    ax.xaxis.label.set_color(COLORS['text'])
    ax.yaxis.label.set_color(COLORS['text'])
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT (legacy): Dialect Probabilities
# ══════════════════════════════════════════════════════════════════════════════
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
