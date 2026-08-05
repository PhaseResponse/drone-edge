"""
input audio samples (n, 4ch)
        |
        v
  sliding_windows()  --> list of (N_SEGMENT, 4) windows
        |
        v
   combine_channels()   --> combined mono signal
        |
        v
samples_to_spectrogram()  --> spectrogram, clipped and normalized [0,1]
        |
        v
   [ CNN model ]      --> score
        |
        v
    [threshold]
        |  
        v
  drone / no-drone
"""

import numpy as np
from scipy.signal import spectrogram as sg

#--- parameters ------------#
TARGET_RMS = 0.1          # mic fusion target RMS level (relative to full scale)
MAX_GAIN = 3.0            # mic fusion gain multiplier
COMP_THRESHOLD = 0.9      # compression threshold 
CEILING = 1.0             # signal max after fusion

N_FFT = 1024              # fft window [samples]
N_OVERLAP = 512           # fft overlap [samples]
FS = 16000                # sampling rate [Hz]
FMAX = 3000               # spectrogram max frequency [Hz]
VMIN = -150               # spectrogram min power [dB]
VMAX = -80                # spectrogram max power [dB]

N_SEGMENT = N_FFT * 20    # samples per segment. 1.28s @ 16kHz.
N_FFT_HOPS = 4            # hop size in units of N_FFT. Default 256ms.

DETECTION_THRESHOLD = 0.974  # drone/no-drone detection threshold, 6 on level_to_threshold scale


def combine_channels(buffer, channels=None):
    """buffer: (n_samples,) or (n_samples, n_channels) -> combined mono (n_samples,)"""
    if buffer.ndim == 1:
        x = buffer
    else:
        if channels is None:
            channels = list(range(buffer.shape[1]))
        x = np.sum(buffer[:, channels], axis=1)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-8
    gain = min(TARGET_RMS / rms, MAX_GAIN)
    x = x * gain

    peak = np.max(np.abs(x))
    if peak > COMP_THRESHOLD:
        ratio = (peak - COMP_THRESHOLD) / (CEILING - COMP_THRESHOLD)
        over = np.abs(x) > COMP_THRESHOLD
        x[over] = np.sign(x[over]) * (COMP_THRESHOLD + (np.abs(x[over]) - COMP_THRESHOLD) / ratio)
    return x


def samples_to_spectrogram(audio, fs=FS, n_fft=N_FFT, n_overlap=N_OVERLAP, fmax=FMAX):
    """audio: combined mono 1D array -> spectrogram, clipped and normalized [0,1]"""
    f, t, Sxx = sg(audio, fs=fs, nperseg=n_fft, noverlap=n_overlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    freq_mask = f <= fmax
    Sxx_db = Sxx_db[freq_mask, :]
    Sxx_db = np.clip(Sxx_db, VMIN, VMAX)
    Sxx_db = (Sxx_db - VMIN) / (VMAX - VMIN)
    return Sxx_db.astype(np.float32)


def sliding_windows(buffer, window_samples=N_SEGMENT, N_FFT_HOPS=4):
    """buffer: (n_samples, 4) arbitrarily long -> list of (N_SEGMENT, 4) windows."""
    hop_samples = N_FFT_HOPS * N_FFT
    windows = []
    for start in range(0, len(buffer) - window_samples + 1, hop_samples):
        windows.append(buffer[start:start + window_samples])
    return windows
    

def is_drone(logit, threshold=DETECTION_THRESHOLD):
    """logit: raw model output, True if sigmoid(logit) >= threshold"""
    score = 1 / (1 + np.exp(-logit))
    return score >= threshold


def level_to_threshold(level, breakpoint_level=5, t_low=0.5, t_mid=0.97, t_high=0.999, 
                       power_low=1.5, power_high=1.2
                       ):
    """level: 0-10 knob scale for threshold, finer resolution t_mid to t_high """
    if level <= breakpoint_level:
        frac = level / breakpoint_level
        return t_low + (t_mid - t_low) * (frac ** power_low)
    else:
        frac = (level - breakpoint_level) / (10 - breakpoint_level)
        return t_mid + (t_high - t_mid) * (frac ** power_high)