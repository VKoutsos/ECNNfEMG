"""
Benchmark on NinaPro DB5, same split as run_live.py --ninapro:
fine-tune on repetitions 1-5, evaluate on repetition 6, for every subject.

Compares the CNN with an LDA baseline on time-domain features, and simulates
live use offline (confidence threshold + majority-vote smoothing, exactly like
classifier.Predictor) to compare thresholds and false-trigger rates.

Usage:
  python benchmark.py                                   all 10 subjects, CNN + LDA (~15-20 min)
  python benchmark.py --subjects 1 2 3
  python benchmark.py --thresholds 0.4 0.6 0.8
  python benchmark.py --no-cnn                          LDA only (fast)

Results are saved to benchmarks/benchmark_<date>_<time>.csv
"""
import argparse
import csv
import datetime
import os
from collections import Counter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import classifier as clf
from features import td_features
from ninapro import load_subject

DB5_GESTURES = ["Rest", "Fist", "Open Hand", "Thumbs Up", "Wrist Flexion", "Wrist Extension"]


def parse_args():
    p = argparse.ArgumentParser(description="DB5 benchmark: CNN vs LDA")
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--gestures", nargs="+", default=DB5_GESTURES)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.4, 0.6, 0.8])
    p.add_argument("--smoothing", type=int, default=5)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--no-cnn", action="store_true", help="LDA only")
    p.add_argument("--db5", default="Ninapro_DB5")
    p.add_argument("--out", default="benchmarks")
    return p.parse_args()


def confusion_matrix(y, pred, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def balanced_acc(cm):
    return float(np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1)))


def live_windows(session, val_spans, step):
    """Windows at exactly the moments Predictor would classify during a replay."""
    emg, truth = clf.replay_arrays(session, val_spans)
    emg = clf.standardize(emg)
    idx = np.arange(clf.WINDOW - 1, len(emg), step)
    X = np.stack([emg[i - clf.WINDOW + 1:i + 1].T for i in idx])[..., None].astype(np.float32)
    return X, truth[idx]


def simulate_live(probs, true, gestures, thresholds, smoothing):
    """Threshold -> 'Rest', then majority vote over the last `smoothing` predictions."""
    rest = gestures.index("Rest") if "Rest" in gestures else -1
    k, conf = probs.argmax(axis=1), probs.max(axis=1)
    results = {}
    for th in thresholds:
        raw = np.where(conf >= th, k, rest)
        smooth = np.array([Counter(raw[max(0, j - smoothing + 1):j + 1]).most_common(1)[0][0]
                           for j in range(len(raw))])
        acc = float(np.mean(smooth == true))
        is_rest = true == rest
        false_trig = float(np.mean(smooth[is_rest] != rest)) if is_rest.any() else float("nan")
        results[th] = (acc, false_trig)
    return results


def main():
    a = parse_args()
    gestures, n = a.gestures, len(a.gestures)
    models = ["lda"] if a.no_cnn else ["lda", "cnn"]
    cm_sum = {m: np.zeros((n, n), dtype=int) for m in models}
    rows = []

    for s in a.subjects:
        print(f"\n=== Subject {s} ===")
        session = load_subject(a.db5, s, gestures)
        emg = clf.standardize(session.emg)
        train_spans, val_spans = clf.split_spans(session, gestures)
        Xtr, ytr = clf.spans_to_windows(emg, train_spans, a.step)
        Xva, yva = clf.spans_to_windows(emg, val_spans, a.step)
        row = {"subject": s}

        # LDA baseline (equal priors, so Rest isn't favoured)
        lda = LinearDiscriminantAnalysis(priors=np.full(n, 1.0 / n))
        lda.fit(td_features(Xtr), ytr)
        cm = confusion_matrix(yva, lda.predict(td_features(Xva)), n)
        cm_sum["lda"] += cm
        row["lda_balanced"] = balanced_acc(cm)

        # CNN fine-tuning
        if "cnn" in models:
            model = clf.finetune(Xtr, ytr, Xva, yva, n, verbose=0)
            cm = confusion_matrix(yva, np.argmax(model.predict(Xva, verbose=0), axis=1), n)
            cm_sum["cnn"] += cm
            row["cnn_balanced"] = balanced_acc(cm)

            Xl, tl = live_windows(session, val_spans, a.step)
            probs = model.predict(Xl, verbose=0, batch_size=256)
            for th, (acc, ft) in simulate_live(probs, tl, gestures, a.thresholds, a.smoothing).items():
                row[f"live@{th}"] = acc
                row[f"falsetrig@{th}"] = ft

        print("  " + "   ".join(f"{k}: {v:.1%}" for k, v in row.items() if k != "subject"))
        rows.append(row)

    # ---------- summary ----------
    keys = [k for k in rows[0] if k != "subject"]
    width = max(14, max(len(k) for k in keys) + 2)
    print("\n" + "=" * (9 + width * len(keys)))
    print(f"{'subject':>9s}" + "".join(f"{k:>{width}s}" for k in keys))
    for r in rows:
        print(f"{r['subject']:>9d}" + "".join(f"{r[k]:>{width}.1%}" for k in keys))
    vals = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    print(f"{'mean':>9s}" + "".join(f"{v:>{width}.1%}" for v in np.nanmean(vals, axis=0)))
    print(f"{'std':>9s}" + "".join(f"{v:>{width}.1%}" for v in np.nanstd(vals, axis=0)))

    print("\nPer-gesture recall over all subjects:")
    print(f"  {'':16s}" + "".join(f"{m.upper():>8s}" for m in models))
    for i, g in enumerate(gestures):
        print(f"  {g:16s}" + "".join(
            f"{cm_sum[m][i, i] / max(cm_sum[m][i].sum(), 1):>8.0%}" for m in models))

    cm = cm_sum[models[-1]]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.fill_diagonal(norm, 0)
    print(f"\nMost common confusions ({models[-1].upper()}, share of the true gesture's windows):")
    for flat in np.argsort(norm, axis=None)[::-1][:5]:
        i, j = divmod(int(flat), n)
        print(f"  {gestures[i]:16s} -> {gestures[j]:16s} {norm[i, j]:.1%}")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, datetime.datetime.now().strftime("benchmark_%Y-%m-%d_%H%M%S.csv"))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject"] + keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()