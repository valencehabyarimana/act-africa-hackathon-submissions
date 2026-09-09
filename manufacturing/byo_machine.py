"""
Track B: preprocess a vibration recording from YOUR machine.

    from byo_machine import prepare
    windows = prepare(signal, fs=400)     # -> (n_windows, 512)

The window covers 256 ms of signal regardless of sampling rate, so periodicity in the
band you CAN measure is preserved.

IMPORTANT - what this is and is not. The challenge data was sampled at 64 kHz and
band-passed 1-10 kHz to capture the resonances that bearing impacts excite. A phone
accelerometer at 200-400 Hz has a Nyquist limit of 100-200 Hz, so that band is not
attenuated, it is absent. `prepare` defaults to roughly 15-95% of YOUR Nyquist
frequency, which at 400 Hz is about 30-190 Hz. What you get is low-frequency
exploratory anomaly detection - shaft harmonics, imbalance, looseness, structural
response - not a reproduction of the Paderborn bearing pipeline.

SAFETY: never modify, loosen or unbalance a machine to create a fault, and never place
a sensor or yourself near a rotating part. Use operating variations that are already
safe and permitted.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample
from scipy.fft import next_fast_len

WINDOW_S = 0.256          # seconds of signal per window
OUT_LEN  = 512            # MOMENT-1-small native context
OVERLAP  = 0.5


def prepare(x, fs, band=None):
    """Envelope-demodulate, resample to 512 samples per 256 ms window."""
    x = np.asarray(x, dtype=float).squeeze()
    if band is None:                       # default: upper half of the usable band
        band = (0.15 * fs / 2, 0.95 * fs / 2)
    lo, hi = band
    if hi >= fs / 2:
        hi = 0.98 * fs / 2
    if lo < hi:
        b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        x = filtfilt(b, a, x)
    env = np.abs(hilbert(x, N=next_fast_len(len(x))))[:len(x)]
    env = env - env.mean()

    raw_win = int(WINDOW_S * fs)
    if raw_win < 16:
        raise ValueError("sampling rate too low for a 256 ms window")
    hop = max(1, int(raw_win * (1 - OVERLAP)))
    segs = [env[i:i + raw_win] for i in range(0, len(env) - raw_win + 1, hop)]
    if not segs:
        raise ValueError(f"signal shorter than one {WINDOW_S*1000:.0f} ms window")
    return np.stack([resample(s, OUT_LEN) for s in segs]).astype(np.float32)


def anomaly_scores(ref, new, n_components=None):
    """Mahalanobis distance of `new` w.r.t. a reference fitted on `ref`.

    Returns (distances, threshold) where threshold is the 95th percentile of the
    distances of `ref` itself.

    Two traps, both of which change the answer completely:

    1. CIRCULARITY. If you pass the same array as both arguments, about 5% of it will
       exceed the threshold by construction - the threshold IS its own 95th percentile.
       That is arithmetic, not a false-alarm rate. Fit on one part of your normal data
       and score a held-out part.

    2. DIMENSIONALITY. A window is 512 numbers. A minute of recording is a few hundred
       windows, so the covariance is estimated from fewer samples than dimensions and
       does not generalise: on the demo signal the held-out false-alarm rate is 84%
       with no reduction, and 11% with n_components=64. Set `n_components` to reduce
       first, and report which value you used - it trades false alarms against
       detection and there is no free lunch.
    """
    import numpy as np
    from sklearn.covariance import LedoitWolf
    ref, new = np.asarray(ref), np.asarray(new)

    if n_components:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, *ref.shape), random_state=0).fit(ref)
        ref, new = pca.transform(ref), pca.transform(new)
    elif ref.shape[0] < ref.shape[1]:
        import warnings
        warnings.warn(f"fitting a {ref.shape[1]}-d covariance from {ref.shape[0]} windows; "
                      "distances on unseen normal data will be inflated. "
                      "Pass n_components to reduce first.", stacklevel=2)

    lw = LedoitWolf().fit(ref)
    return lw.mahalanobis(new), float(np.percentile(lw.mahalanobis(ref), 95))
