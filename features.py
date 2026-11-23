"""
EMG features per window, shared by the LDA baseline and (later) fatigue analysis.

Input windows: (W, 8, L) or (W, 8, L, 1), i.e. channels x samples per window.
"""
import numpy as np


def _squeeze(x):
    x = np.asarray(x, dtype=np.float32)
    return x[..., 0] if x.ndim == 4 else x


def mav(x):
    return np.mean(np.abs(x), axis=-1)


def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=-1))


def wl(x):
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)


def zc(x, thresh=0.0):
    a, b = x[..., :-1], x[..., 1:]
    return np.sum((a * b < 0) & (np.abs(a - b) >= thresh), axis=-1)


def ssc(x, thresh=0.0):
    d1 = x[..., 1:-1] - x[..., :-2]
    d2 = x[..., 1:-1] - x[..., 2:]
    return np.sum(d1 * d2 > thresh, axis=-1)


def td_features(windows, thresh=0.0):
    """Classic time-domain set (MAV, RMS, WL, ZC, SSC) per channel -> (W, 40)."""
    x = _squeeze(windows)
    return np.concatenate([mav(x), rms(x), wl(x), zc(x, thresh), ssc(x, thresh)],
                          axis=-1).astype(np.float32)


def spectral(windows, fs=200):
    """Mean (MNF) and median (MDF) frequency per channel, each (W, 8).
    Meant for fatigue analysis on longer windows (0.5-1 s), not for 160 ms
    classification windows. Note: the Myo only captures content up to 100 Hz."""
    x = _squeeze(windows)
    x = x - x.mean(axis=-1, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / fs)
    total = np.maximum(p.sum(axis=-1), 1e-12)
    mnf = (p * f).sum(axis=-1) / total
    cum = np.cumsum(p, axis=-1)
    mdf = f[np.argmax(cum >= 0.5 * cum[..., -1:], axis=-1)]
    return mnf, mdf