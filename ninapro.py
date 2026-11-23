"""
Load one NinaPro DB5 subject as a Session, so the whole pipeline can be developed
without the armband. Repetitions 1-6 of each movement become rounds 0-5, so the
last repetition is used for validation, just like the last round of our own recordings.
"""
import os

import numpy as np
import scipy.io

from myo_io import FS
from recording import Session

# Exercise B (E2) movement numbers as stored in the .mat files
# (config.py's targets add +12 to these: 13, 15, 17, 18, 25, 26)
DB5_E2 = {
    "Thumbs Up": 1,
    "Ring/Little Flexion": 3,
    "Open Hand": 5,
    "Fist": 6,
    "Wrist Flexion": 13,
    "Wrist Extension": 14,
}


def load_subject(db5_root, subject, gestures, myo="elbow", min_len=64):
    missing = [g for g in gestures if g != "Rest" and g not in DB5_E2]
    if missing:
        raise ValueError(f"Not in DB5 exercise B: {missing}. Available: Rest, {', '.join(DB5_E2)}")

    sdir = os.path.join(db5_root, f"s{subject}")
    fname = next(f for f in os.listdir(sdir) if "_E2_" in f)
    mat = scipy.io.loadmat(os.path.join(sdir, fname))

    emg = mat["emg"][:, :8] if myo == "elbow" else mat["emg"][:, 8:16]
    # 'restimulus'/'rerepetition' are relabelled to match the actual movement; fall back if absent
    lab = (mat["restimulus"] if "restimulus" in mat else mat["stimulus"]).ravel()
    rep = (mat["rerepetition"] if "rerepetition" in mat else mat["repetition"]).ravel()

    wanted = {DB5_E2[g]: g for g in gestures if g != "Rest"}
    change = np.flatnonzero(np.diff(lab)) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(lab)]

    events, prev_round = [], None
    for s, e in zip(starts, ends):
        if e - s < min_len:
            continue
        label = int(lab[s])
        if label in wanted:
            r = int(rep[s]) - 1
            events.append(dict(gesture=wanted[label], round=r, start=int(s), end=int(e),
                               t_start=s / FS, t_end=(e - 1) / FS))
            prev_round = r
        elif label == 0 and "Rest" in gestures and prev_round is not None:
            # rest right after a selected movement gets that movement's round
            events.append(dict(gesture="Rest", round=prev_round, start=int(s), end=int(e),
                               t_start=s / FS, t_end=(e - 1) / FS))
            prev_round = None

    t = np.arange(len(emg)) / FS
    meta = dict(participant=f"DB5_s{subject}", source=f"NinaPro DB5 {fname}",
                myo=myo, gestures=list(gestures))
    return Session(emg.astype(np.float32), t, events, meta)