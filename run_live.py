"""
Record -> fine-tune -> live recognition, with every session saved to sessions/.

With the armband:
  python run_live.py --participant kouts            record 3 rounds, fine-tune, go live
  python run_live.py --load sessions/<folder>       reuse a saved model, straight to live

Without the armband:
  python run_live.py --ninapro 1                    fine-tune on DB5 subject 1 (reps 1-5),
                                                    replay rep 6 as if it were live
  python run_live.py --replay sessions/<folder>     fine-tune on a saved session, replay its last round
  python run_live.py --legacy finetune_recording.npy --gestures Rest Fist "Thumbs Up" "Ok Sign"
"""
import argparse
import asyncio
import datetime
import json
import os

import classifier as clf
from myo_io import FS, MyoBLE, ReplayStream
from ninapro import load_subject
from recording import Session, import_legacy, record_protocol

DEFAULT_GESTURES = ["Rest", "Fist", "Open Hand", "Ok Sign", "Wrist Flexion", "Wrist Extension"]
DB5_GESTURES = ["Rest", "Fist", "Open Hand", "Thumbs Up", "Wrist Flexion", "Wrist Extension"]


def parse_args():
    p = argparse.ArgumentParser(description="Myo gesture recognition")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--load", help="session folder with a saved model: skip recording (Myo)")
    src.add_argument("--replay", help="session folder: fine-tune and replay its last round (no band)")
    src.add_argument("--ninapro", type=int, help="DB5 subject 1-10: fine-tune and replay (no band)")
    src.add_argument("--legacy", help="old finetune_recording.npy: import, fine-tune, replay (no band)")
    p.add_argument("--gestures", nargs="+", help='e.g. --gestures Rest Fist "Open Hand"')
    p.add_argument("--participant", default="unknown")
    p.add_argument("--posture", default="forearm and elbow on table, palm sideways")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--hold", type=float, default=4.0, help="seconds per gesture per round")
    p.add_argument("--threshold", type=float, default=0.6, help="min confidence, else 'Rest'")
    p.add_argument("--smoothing", type=int, default=5)
    p.add_argument("--step", type=int, default=10, help="samples between predictions")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed factor")
    p.add_argument("--myo-name", default="Myo")
    p.add_argument("--sessions", default="sessions")
    p.add_argument("--db5", default="Ninapro_DB5")
    return p.parse_args()


def train_on(session, gestures, args, folder):
    emg = clf.standardize(session.emg)
    train_spans, val_spans = clf.split_spans(session, gestures)
    Xtr, ytr = clf.spans_to_windows(emg, train_spans, args.step)
    Xva, yva = clf.spans_to_windows(emg, val_spans, args.step)
    print(f"\nTraining windows: {len(Xtr)}   validation windows: {len(Xva)}")
    model = clf.finetune(Xtr, ytr, Xva, yva, len(gestures))
    cm, acc, bal = clf.confusion(model, Xva, yva, gestures)
    clf.save_model(model, os.path.join(folder, "model"), gestures)
    with open(os.path.join(folder, "results.json"), "w") as f:
        json.dump(dict(gestures=gestures, confusion=cm.tolist(), val_accuracy=acc,
                       balanced_accuracy=bal, train_windows=len(Xtr), val_windows=len(Xva)),
                  f, indent=2)
    print(f"Model and results saved to {folder}")
    return model, val_spans


async def live(source, predictor, gestures, truth=None):
    state = dict(n=0, hits=0, total=0, last=None)

    def on_sample(t, x):
        i = state["n"]
        state["n"] += 1
        out = predictor(t, x)
        if out is None:
            return
        label, conf = out
        true = gestures[truth[i]] if truth is not None else None
        if true is not None:
            state["total"] += 1
            state["hits"] += int(label == true)
        if label != state["last"]:
            msg = f"-> {label:16s} (conf {conf:.2f})"
            if true is not None:
                msg += f"   [true: {true}]"
            print(msg)
            state["last"] = label

    await source.start(on_sample)
    print("\nLive recognition running. Ctrl+C to stop.")
    while source.connected:
        await asyncio.sleep(0.1)
    if truth is not None and state["total"]:
        print(f"\nReplay finished. Live accuracy (after smoothing): {state['hits'] / state['total']:.1%}")


async def main():
    a = parse_args()

    # ---------- without the armband ----------
    if a.ninapro or a.legacy or a.replay:
        if a.ninapro:
            gestures = a.gestures or DB5_GESTURES
            session = load_subject(a.db5, a.ninapro, gestures)
            folder = Session.new_folder(a.sessions, f"DB5_s{a.ninapro}")
            session.save(folder)
        elif a.legacy:
            gestures = a.gestures or ["Rest", "Fist", "Thumbs Up", "Ok Sign"]
            session = import_legacy(a.legacy, gestures)
            folder = Session.new_folder(a.sessions, "legacy")
            session.save(folder)
        else:
            session = Session.load(a.replay)
            folder = a.replay
            gestures = a.gestures or session.meta["gestures"]

        model, val_spans = train_on(session, gestures, a, folder)
        emg, truth = clf.replay_arrays(session, val_spans)
        source = ReplayStream(emg, speed=a.speed)
        await source.connect()
        unknown = "Rest" if "Rest" in gestures else "Unknown"
        predictor = clf.Predictor(model, gestures, a.step, a.smoothing, a.threshold, unknown)
        await live(source, predictor, gestures, truth)
        return

    # ---------- with the armband ----------
    myo = MyoBLE(a.myo_name)
    await myo.connect()
    try:
        if a.load:
            model, gestures = clf.load_model(os.path.join(a.load, "model"))
            print("Loaded model for:", ", ".join(gestures))
        else:
            gestures = a.gestures or DEFAULT_GESTURES
            emg, t, events = await record_protocol(myo, gestures, rounds=a.rounds, hold_s=a.hold)
            meta = dict(participant=a.participant, posture=a.posture, gestures=gestures,
                        rounds=a.rounds, hold_s=a.hold, fs=FS, window=clf.WINDOW, step=a.step,
                        myo_name=myo.device.name,
                        date=datetime.datetime.now().isoformat(timespec="seconds"))
            folder = Session.new_folder(a.sessions, a.participant)
            session = Session(emg, t, events, meta)
            session.save(folder)
            print(f"\nSession saved to {folder}")
            model, _ = train_on(session, gestures, a, folder)

        unknown = "Rest" if "Rest" in gestures else "Unknown"
        predictor = clf.Predictor(model, gestures, a.step, a.smoothing, a.threshold, unknown)
        await live(myo, predictor, gestures)
    finally:
        await myo.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")