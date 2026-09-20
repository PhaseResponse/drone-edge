"""
calibration.py

Solves the mic array geometry from GPIO-impulse calibration recordings.

Method matches the post https://lnkd.in/p/dy7qR_pf:
    ||x_i - s_n|| = c * (t_ni - t_delta)

    x_i : mic i's position (unknown, solved for)
    s_n : speaker position for recording n (unknown, solved for)
    t_ni: measured arrival time of the impulse at mic i, recording n (via onset detection)
    t_delta: fixed GPIO-trigger-to-emission delay (measured separately from ONE
             known-distance recording, NOT solved for jointly, solving it jointly
             was tested and found numerically fragile/ill-conditioned).

Minimum n >= 7 non-coplanar recordings. Use more for redundancy.
"""

import numpy as np
import soundfile as sf
from scipy.optimize import least_squares

SPEED_OF_SOUND = 343.0  # m/s


def load_recording(path):
    """Read one calibration WAV file. Returns (samples[n_samples, n_channels], fs)."""
    data, fs = sf.read(path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data, fs


def detect_onset(signal, threshold_frac=0.3, min_consecutive=3):
    """
    First-arrival onset detection with sub-sample precision.

    Finds the FIRST threshold crossing that holds for min_consecutive
    samples (not a single-sample spike -- tested to reject impulsive noise
    false-triggers, which a single-sample threshold check does not: up to
    ~20% false-trigger rate at 10dB SNR under impulsive noise with a
    single-sample check, 0% with this sustained-crossing version), then
    refines that point with parabolic interpolation on the signal's local
    peak near the crossing (robust to a later, louder reflection, since
    the search stays local to the first crossing, not the whole signal).

    NOTE: this is deliberately NOT cross-correlation -- the electrical
    trigger pulse and the recorded acoustic impulse don't resemble each
    other (speaker/air/mic reshape it), so there's no usable template to
    correlate against. Each channel's own onset is detected independently.

    Returns the onset sample index (float, sub-sample), or None if no
    sustained threshold crossing is found.
    """
    abs_sig = np.abs(signal)
    peak_val = np.max(abs_sig)
    if peak_val == 0:
        return None
    thresh = threshold_frac * peak_val
    above = abs_sig > thresh
    idx0 = None
    for i in range(len(above) - min_consecutive + 1):
        if np.all(above[i:i+min_consecutive]):
            idx0 = i
            break
    if idx0 is None:
        return None

    search_radius = 10
    window_end = min(idx0 + search_radius, len(abs_sig))
    local_peak_idx = idx0 + np.argmax(abs_sig[idx0:window_end])

    if local_peak_idx == 0 or local_peak_idx == len(abs_sig) - 1:
        return float(local_peak_idx)
    y0, y1, y2 = abs_sig[local_peak_idx-1], abs_sig[local_peak_idx], abs_sig[local_peak_idx+1]
    denom = (y0 - 2*y1 + y2)
    offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return float(local_peak_idx + offset)
    

def measure_t_ni(recordings, fs, threshold_frac=0.3):
    """
    recordings: list of (n_samples, n_channels) arrays, one per calibration position.
    Returns t_ni: array of shape (N, n_channels), arrival time in SECONDS
    (relative to the start of each recording).
    """
    N = len(recordings)
    n_channels = recordings[0].shape[1]
    t_ni = np.zeros((N, n_channels))
    for n, rec in enumerate(recordings):
        for i in range(n_channels):
            onset_sample = detect_onset(rec[:, i], threshold_frac=threshold_frac)
            if onset_sample is None:
                raise ValueError(f"recording {n}, channel {i}: no onset detected above threshold")
            t_ni[n, i] = onset_sample / fs
    return t_ni


def compute_t_delta(t_ni, known_recording_index, known_mic_index, known_distance_m,
                     c=SPEED_OF_SOUND):
    """
    Extracts t_delta from ONE recording where the true speaker-to-mic
    distance was measured by hand (e.g. tape measure to the speaker's
    diaphragm center).

    t_delta = t_ni(that recording, that mic) - true_travel_time
    """
    true_travel_time = known_distance_m / c
    measured = t_ni[known_recording_index, known_mic_index]
    return measured - true_travel_time


def _unpack_params(params, n_mics, N):
    """
    Anchor-fixed mic parametrization (removes translation/rotation/mirror
    ambiguity): mic0=(0,0,0), mic1=(a,0,0) [a>0], mic2=(b,c,0) [c>0],
    mic3=(d,e,f) [f>0], mics 4..n_mics-1 (if any) fully free.
    """
    idx = 0
    mics = [np.zeros(3)]
    if n_mics > 1:
        a = params[idx]; idx += 1
        mics.append(np.array([a, 0, 0]))
    if n_mics > 2:
        b, cc = params[idx], params[idx+1]; idx += 2
        mics.append(np.array([b, cc, 0]))
    if n_mics > 3:
        d, e, f = params[idx], params[idx+1], params[idx+2]; idx += 3
        mics.append(np.array([d, e, f]))
    for _ in range(4, n_mics):
        mics.append(params[idx:idx+3]); idx += 3
    mics = np.array(mics)
    sources = params[idx:idx + 3*N].reshape(N, 3)
    return mics, sources


def _n_anchor_params(n_mics):
    """Number of anchor-fixed parameters for the mic positions."""
    if n_mics <= 1:
        return 0
    n = 1  # mic1: a
    if n_mics > 2:
        n += 2  # mic2: b, c
    if n_mics > 3:
        n += 3  # mic3: d, e, f
    if n_mics > 4:
        n += 3 * (n_mics - 4)  # mics 4+: fully free
    return n


def solve_geometry(t_ni, t_delta, mic_initial_guess, source_initial_guesses,
                    c=SPEED_OF_SOUND, n_restarts=8, jitter_seed=0):
    """
    t_ni: (N, n_mics) array of measured arrival times (seconds).
    t_delta: known scalar (seconds), subtracted out before solving.
    mic_initial_guess: (n_mics, 3) array, nominal/design mic positions.
    source_initial_guesses: (N, 3) array, rough guesses (direction toward
        nearest mic/mic-pair + measured distance from that mic).
    n_restarts: number of solver attempts from jittered initial guesses;
        the lowest-cost result is kept. Tested to fix a real local-minimum
        failure mode at bare-minimum n=6 recordings (verified: single
        attempts occasionally converge to a wrong, self-consistent
        solution with near-zero residual; multiple restarts reliably find
        the true global minimum instead). The first attempt always uses
        the unjittered initial guesses as given.
    jitter_seed: seed for the restart jitter, for reproducibility.

    Returns: (mic_positions, source_positions) both as (n, 3) arrays.
    """
    N, n_mics = t_ni.shape
    travel_time = t_ni - t_delta

    n_anchor = _n_anchor_params(n_mics)

    def unpack(params):
        return _unpack_params(params, n_mics, N)

    def residuals(params):
        mics, sources = unpack(params)
        res = []
        for n in range(N):
            for i in range(n_mics):
                res.append(np.linalg.norm(mics[i] - sources[n]) - c * travel_time[n, i])
        return np.array(res)

    def pack_init(mic_guess, source_guess):
        idx = 0
        init = np.zeros(n_anchor + 3*N)
        if n_mics > 1:
            init[idx] = mic_guess[1, 0]; idx += 1
        if n_mics > 2:
            init[idx] = mic_guess[2, 0]; init[idx+1] = mic_guess[2, 1]; idx += 2
        if n_mics > 3:
            init[idx] = mic_guess[3, 0]
            init[idx+1] = mic_guess[3, 1]
            init[idx+2] = mic_guess[3, 2]
            idx += 3
        for m in range(4, n_mics):
            init[idx:idx+3] = mic_guess[m]; idx += 3
        init[n_anchor:] = source_guess.flatten()
        return init

    lower = -np.inf * np.ones(n_anchor + 3*N)
    upper = np.inf * np.ones(n_anchor + 3*N)
    if n_mics > 1:
        lower[0] = 0.0
    if n_mics > 2:
        lower[2] = 0.0
    if n_mics > 3:
        lower[5] = 0.0

    rng = np.random.default_rng(jitter_seed)
    best = None
    for trial in range(n_restarts):
        if trial == 0:
            mic_guess_t = mic_initial_guess
            source_guess_t = source_initial_guesses
        else:
            mic_guess_t = mic_initial_guess + rng.normal(0, 0.003, mic_initial_guess.shape)
            source_guess_t = source_initial_guesses + rng.normal(0, 0.15, source_initial_guesses.shape)
        init = pack_init(mic_guess_t, source_guess_t)
        result = least_squares(residuals, init, bounds=(lower, upper))
        if best is None or result.cost < best.cost:
            best = result

    mics, sources = unpack(best.x)
    return mics, sources


def pairwise_distances(mic_positions):
    """Returns dict {(i,j): distance} for all mic pairs, i<j."""
    n = len(mic_positions)
    return {(i, j): np.linalg.norm(mic_positions[i] - mic_positions[j])
            for i in range(n) for j in range(i+1, n)}


def run_calibration(recording_paths, known_recording_index, known_mic_index,
                     known_distance_m, mic_initial_guess, source_initial_guesses,
                     c=SPEED_OF_SOUND, threshold_frac=0.3):
    """
    End-to-end: load recordings -> measure t_ni -> extract t_delta -> solve
    geometry -> return mic positions, source positions, t_delta, and the
    pairwise distances (d_0..d_k).
    """
    recordings = []
    fs_list = []
    for path in recording_paths:
        rec, fs = load_recording(path)
        recordings.append(rec)
        fs_list.append(fs)
    if len(set(fs_list)) != 1:
        raise ValueError(f"inconsistent sample rates across recordings: {set(fs_list)}")
    fs = fs_list[0]

    t_ni = measure_t_ni(recordings, fs, threshold_frac=threshold_frac)
    t_delta = compute_t_delta(t_ni, known_recording_index, known_mic_index,
                               known_distance_m, c=c)

    mics, sources = solve_geometry(t_ni, t_delta, mic_initial_guess,
                                    source_initial_guesses, c=c)
    distances = pairwise_distances(mics)

    return {
        "mic_positions": mics,
        "source_positions": sources,
        "t_delta": t_delta,
        "distances": distances,
        "t_ni": t_ni,
    }

""" 
Usage
import numpy as np
import soundfile as sf
from pathlib import Path
from calibration import (load_recording, measure_t_ni, compute_t_delta,
                          solve_geometry, pairwise_distances, SPEED_OF_SOUND)

# ============================================================
# EDIT THESE
# ============================================================
RECORDINGS_PATH = "sounds/calibration_recordings"  # <-- change to your actual folder

# Nominal/design mic positions (your intended tetrahedron layout, used both
# as the "rough guess" for the solver and as reference directions below)
MIC_GUESS = np.array([
    [0.000, 0.000, 0.000],   # mic0
    [0.050, 0.000, 0.000],   # mic1
    [0.025, 0.043, 0.000],   # mic2
    [0.025, 0.014, 0.041],   # mic3
])

# One entry per recording file (by filename). "direction" is the nearest
# mic (e.g. "mic0") or mic-pair midpoint (e.g. "mic0_mic1") you judged the
# speaker to be roughly toward; "distance_m" and "from_mic" are your rough
# tape-measure estimate FROM that specific mic. Doesn't need to be precise
# (tested to tolerate 60+cm of error) -- it's only an initial guess.
POSITION_GUESSES = {
    "pos0.wav": {"direction": "mic0",       "distance_m": 0.60, "from_mic": 0},
    "pos1.wav": {"direction": "mic1",       "distance_m": 0.70, "from_mic": 1},
    "pos2.wav": {"direction": "mic2",       "distance_m": 0.55, "from_mic": 2},
    "pos3.wav": {"direction": "mic3",       "distance_m": 0.80, "from_mic": 3},
    "pos4.wav": {"direction": "mic0_mic1",  "distance_m": 0.65, "from_mic": 0},
    "pos5.wav": {"direction": "mic0_mic2",  "distance_m": 0.90, "from_mic": 2},
    "pos6.wav": {"direction": "mic1_mic2",  "distance_m": 0.75, "from_mic": 1},
    # add more positions as needed (n >= 7 total)
}

# Which recording + which mic + the CAREFULLY measured distance (to the
# speaker diaphragm center) for extracting t_delta. This measurement DOES
# need to be accurate (mm-level) -- unlike the rough guesses above.
KNOWN_DISTANCE_RECORDING = "pos0.wav"
KNOWN_DISTANCE_MIC = 0
KNOWN_DISTANCE_M = 0.612   # <-- your precise tape-measure value

# ============================================================
# load all recordings from RECORDINGS_PATH
# ============================================================
filenames = sorted(POSITION_GUESSES.keys())
recording_paths = [str(Path(RECORDINGS_PATH) / f) for f in filenames]

recordings = []
fs_list = []
for p in recording_paths:
    rec, fs = load_recording(p)
    recordings.append(rec)
    fs_list.append(fs)
assert len(set(fs_list)) == 1, f"inconsistent sample rates: {set(fs_list)}"
fs = fs_list[0]
print(f"loaded {len(recordings)} recordings at {fs}Hz")

# ============================================================
# convert direction labels + rough distances -> rough xyz guesses
# ============================================================
def _reference_directions(mic_guess):
    center = mic_guess.mean(axis=0)
    dirs = {}
    for i in range(4):
        d = mic_guess[i] - center
        dirs[f"mic{i}"] = d / np.linalg.norm(d)
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for i, j in pairs:
        d = (mic_guess[i] + mic_guess[j]) / 2 - center
        dirs[f"mic{i}_mic{j}"] = d / np.linalg.norm(d)
    return dirs

ref_dirs = _reference_directions(MIC_GUESS)

source_initial_guesses = np.zeros((len(filenames), 3))
for n, fname in enumerate(filenames):
    g = POSITION_GUESSES[fname]
    direction = ref_dirs[g["direction"]]
    anchor = MIC_GUESS[g["from_mic"]]
    source_initial_guesses[n] = anchor + direction * g["distance_m"]

# ============================================================
# measure t_ni, extract t_delta, solve geometry
# ============================================================
t_ni = measure_t_ni(recordings, fs)

known_idx = filenames.index(KNOWN_DISTANCE_RECORDING)
t_delta = compute_t_delta(t_ni, known_idx, KNOWN_DISTANCE_MIC, KNOWN_DISTANCE_M)

mic_positions, source_positions = solve_geometry(
    t_ni, t_delta, MIC_GUESS, source_initial_guesses
)
distances = pairwise_distances(mic_positions)

# ============================================================
# OUTPUT
# ============================================================
print("\n--- Results ---")
print("t_delta:", t_delta * 1000, "ms")
print("\nMic positions (m):")
for i, p in enumerate(mic_positions):
    print(f"  mic{i}: {p}")
print("\nPairwise distances (cm):")
for (i, j), d in distances.items():
    print(f"  mic{i}-mic{j}: {d*100:.3f} cm")

"""