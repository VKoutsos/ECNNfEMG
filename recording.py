"""
Sessions: everything recorded in one sitting, saved to its own folder.

    sessions/<date>_<time>_<participant>/
        emg.npy       (N, 8) float32   raw EMG, recorded continuously
        t.npy         (N,)   float64   timestamp per sample (myo_io.clock)
        events.json   labelled segments: gesture, round, start/end sample index (end exclusive),
                      t_start/t_end
        meta.json     participant, posture, gestures, settings, ...
        model/        fine-tuned weights + gesture list   (added by run_live.py)
        results.json  validation confusion matrix          (added by run_live.py)
"""
import asyncio
import datetime
import json
import os

import numpy as np

from myo_io import FS


class Session:
    def __init__(self, emg, t, events, meta):
        self.emg = np.asarray(emg, dtype=np.float32)
        self.t = np.asarray(t, dtype=np.float64)
        self.events = events
        self.meta = meta

    @staticmethod
    def new_folder(root, participant):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in participant)
        folder = os.path.join(root, f"{stamp}_{safe}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "emg.npy"), self.emg)
        np.save(os.path.join(folder, "t.npy"), self.t)
        with open(os.path.join(folder, "events.json"), "w") as f:
            json.dump(self.events, f, indent=1)
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    @classmethod
    def load(cls, folder):
        emg = np.load(os.path.join(folder, "emg.npy"))
        t = np.load(os.path.join(folder, "t.npy"))
        with open(os.path.join(folder, "events.json")) as f:
            events = json.load(f)
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        return cls(emg, t, events, meta)


class _Collector:
    def __init__(self):
        self.emg, self.t = [], []

    def __call__(self, t, x):
        self.t.append(t)
        self.emg.append(x)

    def __len__(self):
        return len(self.t)


async def record_protocol(source, gestures, rounds=3, hold_s=4.0, prepare_s=2.0, relax_s=1.0):
    """Cycle through all gestures `rounds` times. The stream runs continuously (no gaps,
    good for later synchronisation); only the held part of each gesture is labelled."""
    rec = _Collector()
    events = []
    await source.start(rec)
    try:
        for r in range(rounds):
            print(f"\n========== Round {r + 1}/{rounds} ==========")
            for g in gestures:
                print(f"\nGet ready: {g}  (go into the pose NOW and hold it)")
                await asyncio.sleep(prepare_s)
                print(f"Recording {g}...")
                start = len(rec)
                end = start + int(hold_s * FS)
                while len(rec) < end:
                    await asyncio.sleep(0.02)
                events.append(dict(gesture=g, round=r, start=start, end=end,
                                   t_start=rec.t[start], t_end=rec.t[end - 1]))
                print("Done, relax.")
                await asyncio.sleep(relax_s)
    finally:
        await source.stop()
    return np.array(rec.emg, dtype=np.float32), np.array(rec.t), events


def import_legacy(npy_path, gestures):
    """Convert an old finetune_recording.npy (8 x N, gestures back-to-back, equal length)."""
    emg = np.load(npy_path).T.astype(np.float32)
    n = len(emg) // len(gestures)
    events = [dict(gesture=g, round=0, start=i * n, end=(i + 1) * n,
                   t_start=i * n / FS, t_end=((i + 1) * n - 1) / FS)
              for i, g in enumerate(gestures)]
    t = np.arange(len(emg)) / FS
    meta = dict(participant="unknown", source=f"legacy import of {npy_path}",
                gestures=list(gestures), note="no real timestamps in the old format")
    return Session(emg, t, events, meta)