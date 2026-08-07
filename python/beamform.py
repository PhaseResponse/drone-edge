"""
raw 4ch buffer
        |
        v
  align_channels()   --> phase-aligned channels
  (uses estimate_delay_subsample() per channel)
        |
        v
  combine_channels_beamform()  --> combined mono signal

  estimate_doa_3d()  --> 3D direction of arrival (unit vector), given known mic positions
"""

import numpy as np
from scipy.signal import spectrogram as sg
from scipy.interpolate import interp1d
from realtime_pipeline import combine_channels


#--- user configurable parameters ------------#
MAX_LAG = 5               # beamforming max estimated delay [samples]



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


def estimate_doa_3d(delays_samples, mic_positions_m, fs, reference_channel=0, speed_of_sound=343.0):
    """Estimate 3D direction of arrival (unit vector) from delays measured
    relative to reference_channel.
    delays_samples: dict {channel_index: delay_samples}, one entry per non-reference mic.
    mic_positions_m: dict {channel_index: [x, y, z]} for all mics, in meters.
    reference_channel: which channel index is the reference (delay = 0).
    Returns unit vector (x, y, z) pointing toward the source (far-field)."""
    ref_pos = np.array(mic_positions_m[reference_channel])
    other_channels = [ch for ch in mic_positions_m if ch != reference_channel]
    rel_positions = np.array([mic_positions_m[ch] for ch in other_channels]) - ref_pos
    tdoa_s = np.array([delays_samples[ch] for ch in other_channels]) / fs
    rhs = speed_of_sound * tdoa_s
    u, *_ = np.linalg.lstsq(rel_positions, rhs, rcond=None)
    norm = np.linalg.norm(u)
    if norm > 0:
        u = u / norm
    return u

    
  
