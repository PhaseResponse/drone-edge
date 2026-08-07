"""
raw 4ch buffer
        |
        v
  align_channels()   --> phase-aligned channels
  uses estimate_delay_subsample() per channel and fractional_shift()
        |
        v
  combine_channels_beamform()  --> combined mono signal
"""

import numpy as np
from scipy.signal import spectrogram as sg
from scipy.interpolate import interp1d
from realtime_pipeline import combine_channels
from scipy.signal import butter, filtfilt

#--- user configurable parameters ------------#
MAX_LAG = 5               # max delay between channels [samples]
HIGHPASS_CUTOFF = 30      # noise floor [Hz]


def combine_channels_beamform(buffer, max_lag=MAX_LAG, verbose=False):
    """buffer: (n_samples, n_channels) -> combined mono, phase-aligned first"""
    aligned = align_channels(buffer, max_lag=max_lag, verbose=verbose)
    return combine_channels(aligned)
    

def estimate_delay_subsample(ref, sig, max_lag=MAX_LAG):
    """Cross-correlation delay estimate with sub-sample precision via
    parabolic interpolation around the correlation peak."""
    corr = np.correlate(sig, ref, mode='full')
    lags = np.arange(-len(ref) + 1, len(sig))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr[mask]
    lags = lags[mask]

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
    buffer = filtfilt(*butter(4, HIGHPASS_CUTOFF/(16000/2), btype='high'), buffer, axis=0)
    if channels is None:
        channels = list(range(buffer.shape[1]))
    ref = buffer[:, channels[0]]
    aligned = [ref]
    for ch in channels[1:]:
        sig = buffer[:, ch]
        delay = estimate_delay_subsample(ref, sig, max_lag)
        if abs(delay) >= max_lag:
            delay = 0
        if verbose:
            if segment_index is not None and total_segments is not None:
                print(f"segment {segment_index+1}/{total_segments} channel {ch} delay: {delay:.2f}")
            else:
                print(f"channel {ch} delay: {delay:.2f}")
        aligned.append(fractional_shift(sig, delay))
    return np.stack(aligned, axis=1)


def fractional_shift(sig, delay):
    """Shift sig by a (possibly fractional) number of samples via cubic interpolation."""
    n = len(sig)
    idx = np.arange(n)
    interp = interp1d(idx, sig, kind='cubic', fill_value=0.0, bounds_error=False)
    return interp(idx + delay)



    
  
