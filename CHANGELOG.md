# Changelog

All changes compared to the original EffiE code
(https://github.com/MIC-Laboratory/IEEE-NER-2023-EffiE).

## Modular pipeline (branch `vasilis/modular-pipeline`)

### New modules
- `myo_io.py`: EMG sources with one common interface. `MyoBLE` (real armband over
  Bluetooth LE) and `ReplayStream` (plays back recorded data at 200 Hz). Shared
  high-resolution clock (`clock()`) for later synchronisation with other devices
  (e.g. ultrasound).
- `recording.py`: `Session` (save/load), repeated-rounds recording protocol, import of
  old `finetune_recording.npy` files.
- `ninapro.py`: loads one NinaPro DB5 subject (exercise B) as a session, so the full
  pipeline can be developed and tested without the armband. Uses `restimulus` /
  `rerepetition` labels (falls back to `stimulus` / `repetition`).
- `classifier.py`: standardisation, windowing, train/validation split, fine-tuning,
  confusion matrix, save/load of fine-tuned models, live `Predictor`.
- `run_live.py`: main entry point. Record → fine-tune → live with the armband;
  `--load` to reuse a saved model; `--ninapro`, `--replay`, `--legacy` to work without it.
- `features.py`: time-domain features (MAV, RMS, WL, ZC, SSC) for the LDA baseline;
  mean / median frequency (MNF / MDF) for later fatigue analysis.
- `benchmark.py`: CNN vs. LDA baseline on all DB5 subjects with the same split as
  `run_live.py --ninapro`; simulates live use offline to compare confidence thresholds
  and false-trigger rates; saves results to `benchmarks/`.

### Recording and data
- Sessions saved to `sessions/<date>_<time>_<participant>/`: raw EMG, a timestamp per
  sample, labelled events, metadata, fine-tuned model and validation results.
- EMG is streamed continuously during recording (no gaps); only the held part of each
  gesture is labelled.
- Repeated rounds: each gesture is recorded once per round (default 3 rounds × 4 s).

### Training and evaluation
- Validation on the whole last round (a fresh repetition) instead of the last 20% of
  each recording; single-round data falls back to 20% with a one-window gap.
- Windows are cut within each labelled segment only, never across gesture boundaries.
- Class balancing (class weights for training, sample weights for validation), so Rest
  no longer dominates training or early stopping.
- Balanced accuracy (mean per-gesture accuracy) reported alongside overall accuracy;
  weighted accuracy shown during training.
- Up to 100 epochs, so early stopping (patience 10) actually triggers and restores the
  best weights.

### Live recognition
- Confidence threshold (default 0.6): low-confidence predictions are output as Rest,
  to avoid false triggers.
- Predictor shared between the real armband and replay mode.

### Repository
- `.gitignore`: `sessions/` added.
- `__pycache__/` removed from tracking (it had been committed in the original repo).
- `requirements-myo.txt`: `scikit-learn` added (for the LDA baseline).
- README restructured: new usage documentation on top, original EffiE README kept
  below; detailed change list moved to this file.

## Patched `realtime.py` (single-script version, on `main`)

### Compatibility (required to run on current Python / Bleak)
- Updated to the current Bleak API (`BleakScanner.discover`, `disconnected_callback`,
  `is_connected` property); removed calls dropped in newer Bleak versions
- Replaced `asyncio.sleep(..., loop=loop)` and the manual event loop with `asyncio.run()`;
  `nest_asyncio` no longer needed
- Device search matches `MYO_NAME` instead of the hardcoded name `"Cyclops"`,
  prints all found devices and retries instead of crashing

### Bug fixes
- Live prediction stored standardized EMG in an integer array, truncating values to
  whole numbers; now float32, matching the fine-tuning preprocessing
- Live loop re-subscribed to EMG notifications endlessly; now subscribes once

### Fine-tuning
- Recording per gesture: 1.6 s → 6.4 s (`SAMPLES_PER_GESTURE = 40 * window`)
- Batch size 2 → 32 (BatchNorm needs reasonable batch statistics)
- Learning rate 0.0002 → 0.001, up to 60 epochs with early stopping (patience 10;
  in TF 2.9 the best weights are only restored if early stopping actually triggers)
- Proper validation: last 20% of each gesture held out, with a gap so overlapping
  windows don't leak between train and validation
- Prints a confusion matrix (true vs. predicted gesture) after training
- Default gestures: Rest, Fist, Open Hand, Ok Sign, Wrist Flexion, Wrist Extension
  (Thumbs Up removed: hard to distinguish from Fist for a forearm armband)

### Live recognition
- Majority-vote smoothing over the last 5 predictions
- Prints only when the recognized gesture changes
- Calls the model directly instead of `model.predict()` (lower latency)

### Other
- Sends the Myo "never sleep" command so the armband doesn't disconnect during training
- Saves the raw recording to `finetune_recording.npy` for later analysis
- Restructured as a linear script: scan → record → fine-tune → live

## Other file changes
- `config.py`: `filters = [32, 64]` to match the bundled checkpoint (was `[48, 96]`,
  which failed to load)
- `requirements-myo.txt`: working environment (Python 3.9, TF 2.9.1, current Bleak)
- `requirements.txt` renamed to `requirements-original.txt`: authors' conda export,
  reference only
- `realtime_original.py`: authors' original, reference only