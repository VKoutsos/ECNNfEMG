"""
Real-time Myo gesture recognition (patched EffiE realtime.py)
Original: Amir Modan, Jimmy L. @ SF State MIC Lab
Patched for current Bleak / Python 3.9+, with preprocessing and fine-tuning fixes.
"""
import asyncio
import json
import os
from collections import Counter, deque

import numpy as np
import tensorflow as tf
from bleak import BleakClient, BleakScanner

import config
from dataset import realtime_preprocessing
from model import get_finetune

tf.get_logger().setLevel("ERROR")

# ---------------- Settings ----------------
MYO_NAME = "Myo"                     # part of your armband's BLE name (see scan output)
GESTURES = ["Rest", "Fist", "Open Hand", "Wrist Flexion", "Wrist Extension"]

window = 32                          # must match config.window (pretrained input size)
step_size = 10                       # live prediction every 10 samples = 50 ms
SAMPLES_PER_GESTURE = 40 * window    # 1280 samples ~ 6.4 s per gesture
delay = 2.0                          # seconds to get into the pose before recording
val_fraction = 0.2

realtime_batch_size = 32
realtime_epochs = 120
finetune_lr = 0.0001
SMOOTHING = 5                        # majority vote over last N predictions

finetuned_path = "finetuned/checkpoint.ckpt"   # set to None to skip saving
recording_path = "finetune_recording.npy"      # raw EMG, channels x samples

# Myo BLE UUIDs
CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"
EMG_CHARS = [
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060205-a904-deb9-4748-2c7f4a124842",
    "d5060305-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842",
]

with open(config.std_mean_path) as f:
    params = json.load(f)


def decode(payload):
    """One notification = 2 consecutive samples x 8 channels, signed int8."""
    vals = np.frombuffer(bytes(payload[:16]), dtype=np.int8).astype(np.float32)
    return vals[:8], vals[8:]


async def find_myo():
    while True:
        print("Scanning for Myo...")
        devices = await BleakScanner.discover(timeout=5.0)
        for d in devices:
            if d.name:
                print("  found:", d.name, d.address)
        for d in devices:
            if d.name and MYO_NAME.lower() in d.name.lower():
                return d
        print("Myo not found. Is it charged, awake, and not connected elsewhere? Retrying...\n")


async def record(client):
    data = [[] for _ in range(8)]  # channels-first

    def handler(_, payload):
        s1, s2 = decode(payload)
        for ch in range(8):
            data[ch].append(s1[ch])
            data[ch].append(s2[ch])

    for gesture in GESTURES:
        start = len(data[0])
        print(f"\nGet ready: {gesture}  (go into the pose NOW and hold it)")
        await asyncio.sleep(delay)
        print(f"Recording {gesture}...")
        for c in EMG_CHARS:
            await client.start_notify(c, handler)
        while len(data[0]) - start < SAMPLES_PER_GESTURE:
            await asyncio.sleep(0.05)
        for c in EMG_CHARS:
            await client.stop_notify(c)
        await asyncio.sleep(0.2)  # let in-flight notifications arrive before trimming
        for ch in range(8):
            data[ch] = data[ch][:start + SAMPLES_PER_GESTURE]
        print("Done, relax.")
    return data


def finetune(data):
    inputs, y = realtime_preprocessing(
        data, params_path=config.std_mean_path,
        num_classes=len(GESTURES), window=window, step=step_size,
    )
    X = inputs.reshape(-1, 8, window, 1).astype(np.float32)

    # Validation = last 20% of each gesture's recording, with a gap so
    # overlapping windows don't leak between train and validation.
    gap = window // step_size + 1
    tr, va = [], []
    for c in range(len(GESTURES)):
        idx = np.where(y == c)[0]
        cut = int(len(idx) * (1 - val_fraction))
        tr.extend(idx[:cut - gap])
        va.extend(idx[cut:])
    tr, va = np.array(tr), np.array(va)

    model = get_finetune(config.save_path, config.prev_params,
                         lr=finetune_lr, num_classes=len(GESTURES))

     # Stop when val_loss hasn't improved for 10 epochs, keep the best weights
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model.fit(X[tr], y[tr], validation_data=(X[va], y[va]),
              batch_size=realtime_batch_size, epochs=realtime_epochs,
              shuffle=True, callbacks=[early])

    # Confusion matrix: rows = true gesture, columns = predicted
    pred = np.argmax(model.predict(X[va], verbose=0), axis=1)
    print("\nValidation (rows = true gesture, columns = predicted):")
    print(" " * 16 + "".join(f"{g[:8]:>9s}" for g in GESTURES))
    for c, g in enumerate(GESTURES):
        m = y[va] == c
        counts = [np.sum(pred[m] == k) for k in range(len(GESTURES))]
        print(f"  {g:14s}" + "".join(f"{n:9d}" for n in counts) + f"   {np.mean(pred[m] == c):.0%}")

    if finetuned_path:
        os.makedirs(os.path.dirname(finetuned_path), exist_ok=True)
        model.save_weights(finetuned_path)
    return model


async def live(client, model):
    buf = [[] for _ in range(8)]
    recent = deque(maxlen=SMOOTHING)
    last = None

    def handler(_, payload):
        nonlocal buf, last
        s1, s2 = decode(payload)
        for ch in range(8):
            buf[ch].append(s1[ch])
            buf[ch].append(s2[ch])
        if len(buf[0]) >= window:
            x = np.array([b[-window:] for b in buf], dtype=np.float32)  # float32 fix
            for ch in range(8):
                mean, std = params[str(ch)]
                x[ch] = (x[ch] - mean) / std
            probs = model(x.reshape(1, 8, window, 1), training=False).numpy()[0]
            recent.append(int(np.argmax(probs)))
            pred = Counter(recent).most_common(1)[0][0]
            if pred != last:
                print("->", GESTURES[pred])
                last = pred
            buf = [b[-(window - step_size):] for b in buf]

    for c in EMG_CHARS:
        await client.start_notify(c, handler)
    print("\nLive recognition running. Ctrl+C to stop.")
    while client.is_connected:
        await asyncio.sleep(0.1)


async def main():
    device = await find_myo()
    async with BleakClient(device, disconnected_callback=lambda _: print("Disconnected!")) as client:
        print("Connected to", device.name)
        await client.write_gatt_char(CONTROL, bytearray([0x09, 0x01, 0x01]))              # never sleep
        await client.write_gatt_char(CONTROL, bytearray([0x01, 0x03, 0x02, 0x00, 0x00]))  # start EMG
        data = await record(client)
        np.save(recording_path, np.array(data, dtype=np.float32))
        model = finetune(data)
        await live(client, model)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")