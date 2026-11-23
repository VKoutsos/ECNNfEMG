"""
Preprocessing, fine-tuning and live prediction on top of the pretrained EffiE base model.
"""
import json
import os
from collections import Counter, deque

import numpy as np
import tensorflow as tf

import config
from model import get_finetune

WINDOW = config.window  # 32 samples = 160 ms


def _load_scaling(path=config.std_mean_path):
    with open(path) as f:
        p = json.load(f)
    mean = np.array([p[str(c)][0] for c in range(8)], dtype=np.float32)
    std = np.array([p[str(c)][1] for c in range(8)], dtype=np.float32)
    return mean, std


MEAN, STD = _load_scaling()


def standardize(emg):
    """(N, 8) raw -> standardized with the DB5 statistics used in pretraining."""
    return (np.asarray(emg, dtype=np.float32) - MEAN) / STD


def windows(emg, step):
    """(N, 8) -> (W, 8, WINDOW, 1), channels-first like the pretrained model expects."""
    idx = range(0, len(emg) - WINDOW + 1, step)
    if len(idx) == 0:
        return np.empty((0, 8, WINDOW, 1), dtype=np.float32)
    return np.stack([emg[i:i + WINDOW].T for i in idx])[..., None].astype(np.float32)


def split_spans(session, gestures, val_frac=0.2):
    """Return (train_spans, val_spans) as lists of (start, end, class_index).
    Several rounds -> the whole last round is validation.
    One round      -> last 20% of each segment, with a one-window gap."""
    rounds = sorted({e["round"] for e in session.events})
    train, val = [], []
    for e in session.events:
        if e["gesture"] not in gestures:
            continue
        c, s, en = gestures.index(e["gesture"]), e["start"], e["end"]
        if len(rounds) > 1:
            (val if e["round"] == rounds[-1] else train).append((s, en, c))
        else:
            cut = s + int((en - s) * (1 - val_frac))
            train.append((s, cut - WINDOW, c))
            val.append((cut, en, c))
    return train, val


def spans_to_windows(emg_std, spans, step):
    X, y = [], []
    for s, e, c in spans:
        w = windows(emg_std[s:e], step)
        if len(w):
            X.append(w)
            y.append(np.full(len(w), c))
    return np.concatenate(X), np.concatenate(y)


def replay_arrays(session, spans):
    """Concatenate the raw EMG of the given spans, plus the true class per sample."""
    emg = np.concatenate([session.emg[s:e] for s, e, _ in spans])
    truth = np.concatenate([np.full(e - s, c) for s, e, c in spans])
    return emg, truth


def finetune(Xtr, ytr, Xva, yva, n_classes, lr=0.001, epochs=100, batch=32, patience=10,
             balance=True):
    model = get_finetune(config.save_path, config.prev_params, lr=lr, num_classes=n_classes)
    # recompile so weighted validation data also gets a weighted accuracy (no warnings)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"], weighted_metrics=["accuracy"])
    cw = None
    if balance:
        # weight each gesture equally, regardless of how many windows it has
        counts = np.bincount(ytr, minlength=n_classes)
        cw = {c: len(ytr) / (n_classes * max(n, 1)) for c, n in enumerate(counts)}
    # validation gets the same weights, so early stopping also treats gestures equally
    val = (Xva, yva, np.array([cw[c] for c in yva])) if cw else (Xva, yva)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                             restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=val, batch_size=batch, epochs=epochs,
              shuffle=True, callbacks=[early], class_weight=cw, verbose=2)
    return model


def confusion(model, X, y, gestures):
    pred = np.argmax(model.predict(X, verbose=0), axis=1)
    n = len(gestures)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    print("\nValidation (rows = true gesture, columns = predicted):")
    print(" " * 18 + "".join(f"{g[:8]:>9s}" for g in gestures))
    for i, g in enumerate(gestures):
        acc = cm[i, i] / max(cm[i].sum(), 1)
        print(f"  {g:16s}" + "".join(f"{v:9d}" for v in cm[i]) + f"   {acc:.0%}")
    overall = float(np.trace(cm) / max(cm.sum(), 1))
    balanced = float(np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1)))
    print(f"  Overall: {overall:.1%}   Balanced (mean per gesture): {balanced:.1%}")
    return cm, overall, balanced


def save_model(model, folder, gestures):
    os.makedirs(folder, exist_ok=True)
    model.save_weights(os.path.join(folder, "weights.ckpt"))
    with open(os.path.join(folder, "gestures.json"), "w") as f:
        json.dump(gestures, f)


def load_model(folder):
    with open(os.path.join(folder, "gestures.json")) as f:
        gestures = json.load(f)
    model = get_finetune(config.save_path, config.prev_params, num_classes=len(gestures))
    model.load_weights(os.path.join(folder, "weights.ckpt")).expect_partial()
    return model, gestures


class Predictor:
    """Feed raw samples one by one; every `step` samples it classifies the last window.
    Predictions below `threshold` confidence become `unknown` (e.g. "Rest"),
    then a majority vote over the last `smoothing` predictions gives the output."""

    def __init__(self, model, gestures, step=10, smoothing=5, threshold=0.6, unknown="Rest"):
        self.model, self.gestures = model, gestures
        self.step, self.threshold, self.unknown = step, threshold, unknown
        self.buf = deque(maxlen=WINDOW)
        self.since = 0
        self.recent = deque(maxlen=smoothing)

    def __call__(self, t, x):
        """Returns (label, confidence) when a new prediction is made, otherwise None."""
        self.buf.append(x)
        self.since += 1
        if len(self.buf) < WINDOW or self.since < self.step:
            return None
        self.since = 0
        w = standardize(np.array(self.buf)).T[None, :, :, None]
        probs = self.model(w, training=False).numpy()[0]
        k = int(np.argmax(probs))
        conf = float(probs[k])
        self.recent.append(self.gestures[k] if conf >= self.threshold else self.unknown)
        return Counter(self.recent).most_common(1)[0][0], conf