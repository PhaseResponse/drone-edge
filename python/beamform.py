"""
raw 4ch buffer
        |
        v
combine_channels_beamform()
        |
        +-- calls align_channels()
        |      (which uses estimate_delay_subsample() and fractional_shift())
        |
        +-- calls combine_channels()
        |
        v
   combined mono signal
"""

import numpy as np
from scipy.signal import spectrogram as sg
from scipy.interpolate import interp1d
from realtime_pipeline import combine_channels
from scipy.signal import butter, filtfilt

#--- user configurable parameters ------------#
MAX_LAG = 5                       # max delay between channels [samples]
HIGHPASS_CUTOFF = 100             # noise floor [Hz]
LOWPASS_CUTOFF = 3430             # [Hz] spatial nyquist for mic spacing 0.05 m


def combine_channels_beamform(buffer, max_lag=MAX_LAG, verbose=False):
    """buffer: (n_samples, n_channels) -> combined mono, phase-aligned first"""
    aligned = align_channels(buffer, max_lag=max_lag, verbose=verbose)
    return combine_channels(aligned)
    

def estimate_delay_subsample(ref, sig, max_lag=MAX_LAG):
    """Cross-correlation delay estimate with sub-sample precision via
    parabolic interpolation around the correlation peak."""
    lags = np.arange(-max_lag, max_lag + 1)
    sig_padded = np.pad(sig, max_lag)
    corr = np.array([np.dot(ref, sig_padded[max_lag - lag : max_lag - lag + len(ref)]) for lag in lags])
    peak_idx = np.argmax(corr)
    if peak_idx == 0 or peak_idx == len(corr) - 1:
        return float(lags[peak_idx])  # peak at edge, can't interpolate

    y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
    denom = (y0 - 2 * y1 + y2)
    offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return float(lags[peak_idx] + offset)

    
def align_channels(buffer, channels=None, max_lag=MAX_LAG, segment_index=None, 
                   total_segments=None, verbose=False):
    """buffer: (n_samples, n_channels) -> aligned channels (n_samples, n_channels)."""
    if buffer.ndim == 1: buffer = buffer.reshape(-1, 1)

    b_hp, a_hp = butter(4, HIGHPASS_CUTOFF/(16000/2), btype='high')
    b_lp, a_lp = butter(4, LOWPASS_CUTOFF/(16000/2), btype='low')
    buffer_for_delay = filtfilt(b_lp, a_lp, filtfilt(b_hp, a_hp, buffer, axis=0), axis=0)

    if channels is None:
        channels = list(range(buffer.shape[1]))
    ref = buffer_for_delay[:, channels[0]]      # delay estimate: band-limited
    ref_full = buffer[:, channels[0]]           # output: full-band
    aligned = [ref_full]
    for ch in channels[1:]:
        sig = buffer_for_delay[:, ch]           # delay estimate: band-limited
        sig_full = buffer[:, ch]                # output: full-band
        delay = estimate_delay_subsample(ref, sig, max_lag)
        if abs(delay) >= max_lag:
            delay = 0
        if verbose:
            if segment_index is not None and total_segments is not None:
                print(f"segment {segment_index+1}/{total_segments} channel {ch} delay: {delay:.2f}")
            else:
                print(f"channel {ch} delay: {delay:.2f}")
        aligned.append(fractional_shift(sig_full, -delay))
    return np.stack(aligned, axis=1)
    

def fractional_shift(sig, delay):
    """Shift sig by a (possibly fractional) number of samples via cubic interpolation."""
    n = len(sig)
    idx = np.arange(n)
    interp = interp1d(idx, sig, kind='cubic', fill_value=0.0, bounds_error=False)
    return interp(idx + delay)



    
  
