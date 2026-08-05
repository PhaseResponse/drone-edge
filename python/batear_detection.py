"""
Acoustic drone detector — Python reimplementation of Batear algorithm [2].
"""

import numpy as np
from scipy.io import wavfile

# --- constants  ---
N_FRAMES          = 16        # frames per segment
N_SEGMENT_OVERLAP = 8         # frames overlapping previous segment, for EMA initial conditions

# batear defaults
FFT_SIZE          = 1024      # fft window size [samples]     
SAMPLE_RATE       = 16000     # [Hz]

BIN_HZ            = SAMPLE_RATE / FFT_SIZE          # 15.625 Hz/bin
N_BINS_HALF       = FFT_SIZE // 2 + 1               # 513

F0_MIN_HZ         = 180.0
F0_MAX_HZ         = 2400.0

EMA_ALPHA         = 0.25
CONF_ON           = 0.30    # threshold to trigger alarm
CONF_OFF          = 0.18    # threshold to clear alarm 
CONFIRM_FRAMES    = 2       # consecutive positives to trigger alarm
CLEAR_FRAMES      = 8       # consecutive negatives to clear alarm
RMS_MIN           = 0.0004  # minimum RMS to process frame

HARM_PEAK_MIN_SNR = 4.0    # minimum SNR for fundamental
HARM_MIN_H2       = 0.07   # minimum h2/fundamental ratio
HARM_MIN_H3       = 0.035  # minimum h3/fundamental ratio
HARM_BIN_TOL      = 2      # ±bins tolerance for harmonic search

# for plotting results
ALARM_ALPHA       = 0.25

# --- helper functions ---

def clamp_bin(k, n):
    return max(0, min(k, n - 1))

def local_peak_max(psd, n_bins_half, center, tol):
    lo = clamp_bin(center - tol, n_bins_half)
    hi = clamp_bin(center + tol, n_bins_half)
    return float(np.max(psd[lo:hi+1]))

def noise_floor_estimate(psd):
    # return float(np.median(psd))
    return float(np.mean(psd[1:]))  # exclude DC bin 0

def analyze_harmonics(psd):
    """
    Detection function per frame from: batear/main/audio_processor.c/analyze_harmonics()
    """
    n = len(psd)
    nf = noise_floor_estimate(psd)

    """
    Frequency bins (PSD):
     
     bin: 0    k0                        k1        N/2
          |-----|--------------------------|---------|
          |     |   search range           |         |
          | DC  |<------------------------>| Nyquist |
          |     | F0_MIN_HZ .. F0_MAX_HZ   |         |
          |-----|--------------------------|---------|
    
    k0 = ceil(F0_MIN_HZ / BIN_HZ)   → first bin at or above F0_MIN_HZ
    k1 = floor(F0_MAX_HZ / BIN_HZ)  → last bin at or below F0_MAX_HZ
    
    clamp_bin ensures k0, k1 stay within [0, N/2]
    (guards against F0_MIN/MAX outside the actual spectrum range)
    """
    k0 = int(np.ceil(F0_MIN_HZ / BIN_HZ))
    k1 = int(np.floor(F0_MAX_HZ / BIN_HZ))
    k0 = clamp_bin(k0, n)
    k1 = clamp_bin(k1, n)

    if k1 <= k0:
        return None
    """
    PSD bins k0 to k1 (search range):
    
    For example:
    bin:  k0  k0+1  k0+2  k0+3  ...  k0+j  ...  k1
    psd: [0.2, 0.1,  0.8,  0.3, ..., 2.1,  ..., 0.4]
                                       ↑
                                  argmax = j
                                  peak_bin = k0 + j
                                  peak = psd[peak_bin] = 2.1
    
    SNR:
                    peak          2.1
          snr = ————————————— = ———————— , where nf is the noise floor
                  nf + 1e-18      nf
                                  
          (1e-18 avoids division by zero if noise floor = 0)
    """    
    peak_bin = k0 + int(np.argmax(psd[k0:k1+1]))
    peak     = psd[peak_bin]
    snr      = peak / (nf + 1e-18)

    result = dict(
        noise_floor     = nf,
        fundamental_bin = peak_bin,
        fundamental_hz  = peak_bin * BIN_HZ,
        fundamental_pwr = peak,
        snr             = snr,
        h2_ratio        = 0.0,
        h3_ratio        = 0.0,
        confidence      = 0.0,
        detected        = False,
    )

    """
    SNR check — does the peak stand out enough above the noise floor?
    
              peak
      snr = ————————
               nf
    
      snr:  0    1    2    3    4    5    ...
            |————|————|————|————|————|————>
                                ↑
                       HARM_PEAK_MIN_SNR = 4.0
                       
      snr < 4.0 → peak too weak, not drone-like → return early, confidence = 0
      snr ≥ 4.0 → peak strong enough → continue to harmonic check
    """  

    if snr < HARM_PEAK_MIN_SNR:
        return result

    """
    Check that 2nd and 3rd harmonics fall within the spectrum:
    
      fundamental at peak_bin:    peak_bin
      2nd harmonic center:        h2c = peak_bin * 2
      3rd harmonic center:        h3c = peak_bin * 3
    
      spectrum bins: 0 ————————————————————— n-1 (= N/2 = 512)
    
      Case 1: peak_bin small → harmonics fit
      
      bin:    0    peak_bin    h2c         h3c      n-1
              |————|————————————|———————————|————————|
                                ✓           ✓    → CONTINUE
    
      Case 2: peak_bin large → h3c or h2c out of range
      
      bin:    0              peak_bin    h2c    n-1      h3c
              |——————————————|————————————|——————|        |
                                                ✗    → RETURN EARLY
                                          (h3c >= n, above Nyquist)
    
    n = N/2 + 1 = 513  (= Nyquist bin + 1)
    so bin >= n means above Nyquist by definition
    """
    h2c = peak_bin * 2
    h3c = peak_bin * 3
    if h2c >= n or h3c >= n:
        return result
    """
    Harmonic power check with bin tolerance (HARM_BIN_TOL = 2):
    
      local_peak_max searches ±2 bins around expected harmonic center:
    
      bin:  h2c-2  h2c-1  h2c  h2c+1  h2c+2
      psd: [ 0.1,   0.3,  1.2,  0.8,   0.2 ]
                           ↑ h2_pwr = max in window = 1.2
    
      (accounts for slight RPM variation shifting harmonics off exact integer multiples)
    
    Harmonic ratios:
      h2_ratio = h2_pwr / peak      h3_ratio = h3_pwr / peak
      (how strong is harmonic relative to fundamental?)
    
    Threshold check:
      h2_ok = h2_ratio >= HARM_MIN_H2 (0.07)
      h3_ok = h3_ratio >= HARM_MIN_H3 (0.035)
    """
    h2_pwr = local_peak_max(psd, n, h2c, HARM_BIN_TOL)
    h3_pwr = local_peak_max(psd, n, h3c, HARM_BIN_TOL)

    denom        = peak + 1e-18
    result['h2_ratio'] = h2_pwr / denom
    result['h3_ratio'] = h3_pwr / denom

    h2_ok = result['h2_ratio'] >= HARM_MIN_H2
    h3_ok = result['h3_ratio'] >= HARM_MIN_H3

    """
      ┌─────────────┬─────────────┬──────────────────────────────────────┐
      │    h2_ok    │    h3_ok    │ result                               │
      ├─────────────┼─────────────┼──────────────────────────────────────┤
      │     ✗       │  ✗ or ✓     │ partial confidence (max of ratios)   │
      │     ✓       │     ✗       │ partial confidence (max of ratios)   │
      │     ✓       │     ✓       │ full confidence (geometric mean)     │
      └─────────────┴─────────────┴──────────────────────────────────────┘
    
    Confidence formulas:
    
      FAIL (h2 or h3 below threshold):
        conf = min(1.0, snr/40 * max(h2_ratio/HARM_MIN_H2, h3_ratio/HARM_MIN_H3))
        conf = min(conf, 0.99)   ← never reaches 1.0 on partial detection
        detected = False
    
      PASS (both above threshold):
        conf = min(1.0, snr/25 * sqrt(h2_ratio * h3_ratio))
        detected = True
    
      snr/40 vs snr/25: stricter scaling for partial detection
      sqrt(h2*h3): geometric mean — both harmonics must be strong
    """
    if not h2_ok or not h3_ok:
        result['confidence'] = min(1.0,
            (snr / 40.0) * max(result['h2_ratio'] / HARM_MIN_H2,
                               result['h3_ratio'] / HARM_MIN_H3))
        result['confidence'] = min(result['confidence'], 0.99)
        return result

    result['confidence'] = min(1.0,
        (snr / 25.0) * np.sqrt(result['h2_ratio'] * result['h3_ratio']))
    result['detected'] = True
    return result


def process_file(wav_path, hop_size=None):
    """
    process full audio file
    EMA, hysteresis and alarm logic from: batear/main/audio_task.c
    """
    fs, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
           
    if fs != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, fs)
        audio = resample_poly(audio, SAMPLE_RATE // g, fs // g)
        fs = SAMPLE_RATE

    if hop_size is None:
        hop_size = FFT_SIZE // 2

    window   = np.hanning(FFT_SIZE)
    n_frames = (len(audio) - FFT_SIZE) // hop_size + 1

    times       = []
    confidences = []
    detections  = []
    fundamentals= []
    alarms      = []
    h2_ratios = []
    h3_ratios = []
    
    ema_conf       = 0.0
    confirm_count  = 0
    clear_count    = 0
    alarm_active   = False

    for i in range(n_frames):
        start  = i * hop_size
        frame  = audio[start:start + FFT_SIZE]
        if len(frame) < FFT_SIZE:
            break

        windowed = frame * window
        rms      = np.sqrt(np.mean(windowed ** 2))
        spectrum = np.fft.rfft(windowed)
        psd      = (np.abs(spectrum) ** 2) / FFT_SIZE
        t        = (start + FFT_SIZE // 2) / fs

        if rms < RMS_MIN:
            ema_conf *= (1 - EMA_ALPHA)
            harm_ok = False
            conf = 0.0
            f0 = 0.0
        else:
            r = analyze_harmonics(psd)
            h2_ratios.append(r['h2_ratio'] if r is not None else 0.0)
            h3_ratios.append(r['h3_ratio'] if r is not None else 0.0)
            if r is None:
                harm_ok = False
                conf    = 0.0
                f0      = 0.0
            else:
                harm_ok = r['detected']
                conf    = r['confidence']
                f0      = r['fundamental_hz']

            if harm_ok:
                ema_conf = EMA_ALPHA * conf + (1 - EMA_ALPHA) * ema_conf
            else:
                ema_conf = EMA_ALPHA * min(conf, 0.15) + (1 - EMA_ALPHA) * ema_conf

        if not alarm_active:
            if harm_ok and ema_conf >= CONF_ON:
                confirm_count += 1
                clear_count    = 0
                if confirm_count >= CONFIRM_FRAMES:
                    alarm_active = True
            else:
                confirm_count = 0
        else:
            if not harm_ok or ema_conf < CONF_OFF:
                clear_count   += 1
                confirm_count  = 0
                if clear_count >= CLEAR_FRAMES:
                    alarm_active = False
            else:
                clear_count = 0

        times.append(t)
        confidences.append(ema_conf)
        detections.append(harm_ok)
        fundamentals.append(f0)
        alarms.append(alarm_active)

    return dict(
        times        = np.array(times),
        confidences  = np.array(confidences),
        detections   = np.array(detections),
        fundamentals = np.array(fundamentals),
        h2_ratios = np.array(h2_ratios),
        h3_ratios = np.array(h3_ratios),
        alarms       = np.array(alarms),
        audio        = audio,
        fs           = fs,
    )    


def score_segment(audio_segment, fs=None, n_overlap=N_SEGMENT_OVERLAP):
    """Returns max confidence score over scored frames, after EMA warmup."""
    if fs is None:
        fs = SAMPLE_RATE
    hop = FFT_SIZE // 2
    window = np.hanning(FFT_SIZE)
    ema_conf = 0.0
    max_conf = 0.0
    total_frames = N_FRAMES + n_overlap
    for i in range(total_frames):
        start = i * hop
        frame = audio_segment[start:start + FFT_SIZE]
        if len(frame) < FFT_SIZE:
            break
        windowed = frame * window
        rms = np.sqrt(np.mean(windowed ** 2))
        spectrum = np.fft.rfft(windowed)
        psd = (np.abs(spectrum) ** 2) / FFT_SIZE
        if rms < RMS_MIN:
            ema_conf *= (1 - EMA_ALPHA)
        else:
            r = analyze_harmonics(psd)
            conf = r['confidence'] if r else 0.0
            harm_ok = r['detected'] if r else False
            if harm_ok:
                ema_conf = EMA_ALPHA * conf + (1 - EMA_ALPHA) * ema_conf
            else:
                ema_conf = EMA_ALPHA * min(conf, 0.15) + (1 - EMA_ALPHA) * ema_conf
        if i >= n_overlap:  # only score after warmup frames
            max_conf = max(max_conf, ema_conf)
    return max_conf
    
