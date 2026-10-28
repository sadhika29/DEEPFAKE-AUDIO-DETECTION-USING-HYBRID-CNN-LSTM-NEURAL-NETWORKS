import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import (
    SAMPLE_RATE, NUM_SAMPLES, N_MELS, N_FFT, HOP_LENGTH,
    FMIN, FMAX, TIME_FRAMES, INPUT_SHAPE
)

def load_audio(file_path, target_sr=SAMPLE_RATE):
    """
    Load an audio file, resample to target_sr, and convert to mono.
    Supports WAV, FLAC, MP3, OGG.
    """
    try:
        y, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        try:
            data, sr = sf.read(str(file_path))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            return data.astype(np.float32), target_sr
        except Exception as inner_e:
            raise RuntimeError(f"Error loading audio file {file_path}: {e} | {inner_e}")

def pad_or_trim_audio(y, target_length=NUM_SAMPLES):
    """
    Ensures audio array is exactly target_length samples.
    - If shorter: repeats audio cyclically until target_length is reached.
    - If longer: trims center slice for balanced vocal content.
    """
    length = len(y)
    if length == target_length:
        return y
    if length < target_length:
        repeats = int(np.ceil(target_length / length))
        return np.tile(y, repeats)[:target_length]
    else:
        start = (length - target_length) // 2
        return y[start:start + target_length]

def compute_mel_spectrogram(
    y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    fmin=FMIN, fmax=FMAX, target_frames=TIME_FRAMES
):
    """
    Computes a normalized Log-Mel Spectrogram from audio array y.
    Returns:
        spectrogram: np.ndarray of shape (n_mels, target_frames, 1) normalized in [0, 1]
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )
    
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    current_frames = log_mel.shape[1]
    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='edge')
    elif current_frames > target_frames:
        log_mel = log_mel[:, :target_frames]

    mel_min = log_mel.min()
    mel_max = log_mel.max()
    if mel_max - mel_min > 1e-6:
        norm_mel = (log_mel - mel_min) / (mel_max - mel_min)
    else:
        norm_mel = np.zeros_like(log_mel)

    return np.expand_dims(norm_mel.astype(np.float32), axis=-1)

def extract_features_from_file(file_path):
    """
    Full pipeline: Loads audio, pads/trims, and computes normalized Mel-spectrogram.
    """
    y, sr = load_audio(file_path)
    y_padded = pad_or_trim_audio(y)
    return compute_mel_spectrogram(y_padded, sr)

def compute_acoustic_forensic_metrics(y, sr=SAMPLE_RATE):
    """
    Computes acoustic forensic cues that physically distinguish real speech from AI vocoders:
    1. High-Frequency Vocoder Energy Ratio: Neural vocoders often show unnatural roll-off or cutoffs >4kHz.
    2. Spectral Flatness: AI speech exhibits unnatural flatness/hiss in unvoiced sections.
    3. Spectral Centroid Variance: Human speech has dynamic pitch drift; AI speech has rigid pitch.
    4. Zero-Crossing Regularity: Measures phase predictability.
    """
    if len(y) < sr:
        y = pad_or_trim_audio(y)

    # 1. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_std = float(np.std(cent))
    cent_mean = float(np.mean(cent))

    # 2. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_mean = float(np.mean(flatness))

    # 3. High-Frequency Energy Ratio (>4000 Hz vs Total)
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hf_mask = freq_bins >= 4000
    hf_energy = np.sum(stft[hf_mask, :])
    total_energy = np.sum(stft) + 1e-8
    hf_ratio = float(hf_energy / total_energy)

    # 4. Zero-Crossing Rate variance
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_std = float(np.std(zcr))

    return {
        "centroid_mean": cent_mean,
        "centroid_variance": cent_std,
        "spectral_flatness": flatness_mean,
        "high_frequency_ratio": hf_ratio,
        "zcr_variance": zcr_std
    }

def plot_waveform_and_spectrogram(file_path, save_path=None):
    """
    Generates a dual-panel visualization of the raw audio waveform
    and its corresponding Log-Mel Spectrogram.
    """
    y, sr = load_audio(file_path)
    y_fixed = pad_or_trim_audio(y)
    mel_spec = compute_mel_spectrogram(y_fixed, sr)[:, :, 0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    time_axis = np.linspace(0, len(y_fixed) / sr, num=len(y_fixed))
    axes[0].plot(time_axis, y_fixed, color="#2563eb", linewidth=0.8)
    axes[0].set_title(f"Audio Waveform: {Path(file_path).name}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    img = axes[1].imshow(
        mel_spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, len(y_fixed) / sr, FMIN, FMAX]
    )
    axes[1].set_title("Log-Mel Spectrogram (Frequency vs Time)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
