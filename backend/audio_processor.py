"""
Audio processing module for Arabic Dialect Detection.
Handles loading, spectrogram generation, feature extraction, and audio mixing.
"""
import io, base64
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf


def load_audio(file_path, sr=22050, duration=None):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    y = librosa.util.normalize(y)
    return y, sr


def compute_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1A1410')
    ax.set_facecolor('#1A1410')
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogram', color='#F5E6D3', fontsize=14)
    ax.tick_params(colors='#F5E6D3')
    ax.xaxis.label.set_color('#F5E6D3')
    ax.yaxis.label.set_color('#F5E6D3')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1A1410', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_mel_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1A1410')
    ax.set_facecolor('#1A1410')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram', color='#F5E6D3', fontsize=14)
    ax.tick_params(colors='#F5E6D3')
    ax.xaxis.label.set_color('#F5E6D3')
    ax.yaxis.label.set_color('#F5E6D3')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1A1410', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#1A1410')
    ax.set_facecolor('#1A1410')
    times = np.arange(len(y)) / sr
    ax.plot(times, y, color='#7C3AED', linewidth=0.5, alpha=0.8)
    ax.set_xlabel('Time (s)', color='#F5E6D3')
    ax.set_ylabel('Amplitude', color='#F5E6D3')
    ax.set_title('Waveform', color='#F5E6D3', fontsize=14)
    ax.tick_params(colors='#F5E6D3')
    ax.set_xlim([0, times[-1]])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1A1410', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def extract_features(y, sr):
    features = {}
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
        features[f'mfcc_delta_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
        features[f'mfcc_delta2_{i+1}_mean'] = float(np.mean(mfcc_delta2[i]))
    # Spectral Centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(sc))
    features['spectral_centroid_std'] = float(np.std(sc))
    # Spectral Bandwidth
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = float(np.mean(sb))
    features['spectral_bandwidth_std'] = float(np.std(sb))
    # Spectral Rolloff
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(sr_))
    features['spectral_rolloff_std'] = float(np.std(sr_))
    # Spectral Contrast
    scon = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(scon.shape[0]):
        features[f'spectral_contrast_{i+1}_mean'] = float(np.mean(scon[i]))
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f'chroma_{i+1}_mean'] = float(np.mean(chroma[i]))
    # RMS
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    # Pitch
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0])
    if len(f0_clean) == 0:
        f0_clean = np.array([0])
    features['pitch_mean'] = float(np.mean(f0_clean))
    features['pitch_std'] = float(np.std(f0_clean))
    features['pitch_range'] = float(np.ptp(f0_clean))
    features['pitch_median'] = float(np.median(f0_clean))
    # Spectral Flatness
    sf_ = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = float(np.mean(sf_))
    features['spectral_flatness_std'] = float(np.std(sf_))
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(6):
        features[f'tonnetz_{i+1}_mean'] = float(np.mean(tonnetz[i]))
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    return features


def get_feature_vector(features_dict):
    return np.array(list(features_dict.values()))

def get_feature_names(features_dict):
    return list(features_dict.keys())


def mix_audio(y1, sr1, y2, sr2, weight, target_sr=22050):
    if sr1 != target_sr:
        y1 = librosa.resample(y1, orig_sr=sr1, target_sr=target_sr)
    if sr2 != target_sr:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=target_sr)
    max_len = max(len(y1), len(y2))
    if len(y1) < max_len:
        y1 = np.pad(y1, (0, max_len - len(y1)))
    if len(y2) < max_len:
        y2 = np.pad(y2, (0, max_len - len(y2)))
    mixed = (1 - weight) * y1 + weight * y2
    mixed = librosa.util.normalize(mixed)
    return mixed, target_sr


def audio_to_wav_bytes(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    return buf.read()

def audio_to_base64_wav(y, sr):
    return base64.b64encode(audio_to_wav_bytes(y, sr)).decode('utf-8')
