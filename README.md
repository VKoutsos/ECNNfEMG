# ECNNfEMG: Real-time EMG Gesture Recognition with the Myo Armband

Real-time hand gesture classification from the Thalmic Myo armband (8 EMG channels, 200 Hz),
built on the EffiE project (IEEE NER 2023, SF State MIC Lab): a small CNN pretrained on
NinaPro DB5 and fine-tuned on each user's own short recording.

Part of our Cybathlon project. Later stages will add sonomyography (ultrasound) and muscle
fatigue assessment; the pipeline is structured with that in mind.

## Setup (Windows)
```
conda create -n myo python=3.9
conda activate myo
pip install -r requirements-myo.txt
```
`requirements-original.txt` is the authors' Windows conda export, kept for reference only.
Don't install from it.

If `conda` isn't recognised in PowerShell / VS Code, run once (then restart VS Code):
```
C:\Users\<you>\miniconda3\Scripts\conda.exe init powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**NinaPro DB5:** needed for replay mode and benchmarks. Unzip the subject archives into
`Ninapro_DB5/s1 … s10` (only these folders inside `Ninapro_DB5/`).

## Quick start

### With the armband
1. Bluetooth on in Windows, but **don't pair** the Myo in Windows settings. Close Myo Connect.
   The USB dongle is not used.
2. Wake the armband and put it on (near the elbow, same position every time).
3. Record, fine-tune and go live:
   ```
   python run_live.py --participant <name>
   ```
   3 rounds × 6 gestures × 4 s (~2.5 min). At **"Get ready: X"** go into the pose and hold it
   until "Done, relax." Everything is saved to `sessions/<date>_<name>/`.
4. Reuse a saved model without recording:
   ```
   python run_live.py --load sessions/<folder>
   ```

### Without the armband
```
python run_live.py --ninapro 1 --speed 4      # DB5 subject 1 as a stand-in user
python run_live.py --replay sessions/<folder> # re-fine-tune on a saved session
python benchmark.py                           # CNN vs LDA over all DB5 subjects
```

### Main options (`run_live.py`)
| Option | Default | Meaning |
|---|---|---|
| `--gestures` | Rest, Fist, Open Hand, Ok Sign, Wrist Flexion, Wrist Extension | Gesture list (Rest first) |
| `--rounds` / `--hold` | 3 / 4.0 s | Recording rounds and seconds per gesture |
| `--threshold` | 0.6 | Below this confidence the output is Rest (avoids false triggers) |
| `--smoothing` | 5 | Majority vote over the last N predictions |
| `--speed` | 1.0 | Replay speed |
| `--myo-name` | Myo | Part of the armband's Bluetooth name |

## Recording protocol
- Forearm **and elbow resting on the table**, hand past the edge, shoulder relaxed
- **Same palm orientation for every gesture** (we use palm sideways / thumb up)
- Moderate, steady force; Rest = hand fully limp
- Wrist flexion/extension: only the wrist moves, fingers relaxed
- Same posture for recording and live use

## How it works
1. **Pretrained model** (`checkpoints/model.ckpt`): the authors' CNN trained on NinaPro DB5.
   Input: 160 ms windows (8 channels × 32 samples), standardised with `scaling_params.json`.
2. **Fine-tuning:** the output layer is replaced to match our gesture list; all layers are
   trained on the user's recording (LR 0.001, class-balanced, early stopping).
   Validation = the last recording round.
3. **Live:** a prediction every 50 ms; low-confidence predictions become Rest; majority vote
   smooths the output.

It is a classifier of static poses: it recognises *which* gesture is held, not how strongly.

## Repository structure
| File | Purpose |
|---|---|
| `run_live.py` | Main entry point: record / load / replay → fine-tune → live |
| `myo_io.py` | EMG sources with one interface: real Myo (`MyoBLE`) or replay (`ReplayStream`); shared clock |
| `recording.py` | Sessions (save/load), repeated-rounds protocol, import of old recordings |
| `ninapro.py` | Loads one DB5 subject as a session |
| `classifier.py` | Preprocessing, train/val split, fine-tuning, confusion matrix, live `Predictor` |
| `features.py` | EMG features: MAV, RMS, WL, ZC, SSC; MNF/MDF for fatigue work |
| `benchmark.py` | CNN vs LDA on all DB5 subjects, threshold comparison |
| `realtime.py` | Earlier single-script version (one round, no sessions); still works |
| `realtime_original.py` | Authors' original, reference only (needs Bleak 0.15 / Python ≤ 3.9) |
| `model.py`, `dataset.py`, `train.py`, `config.py` | From the original repo (`config.py`: `filters = [32, 64]`) |
| `tensorflow_lite.py` | Original TFLite export; has known issues, see below |
| `requirements-myo.txt` | Our environment |
| `requirements-original.txt` | Authors' conda export, reference only |
| `CHANGELOG.md` | All changes compared to the original EffiE code |

### Session format
```
sessions/<date>_<time>_<participant>/
    emg.npy       (N, 8)  raw EMG, recorded continuously
    t.npy         (N,)    timestamp per sample (epoch seconds, high resolution)
    events.json   labelled segments: gesture, round, start/end sample, times
    meta.json     participant, posture, gestures, settings
    model/        fine-tuned weights + gesture list
    results.json  validation confusion matrix and accuracy
```
`sessions/` is git-ignored (personal recordings). Benchmark results in `benchmarks/` are committed.

## Changes from the original EffiE code
See [CHANGELOG.md](CHANGELOG.md) for the full list. Main points: works with current
Bleak / Python 3.9+, fixed integer truncation in live prediction, fixed checkpoint
mismatch in `config.py`, repeated-rounds recording with proper validation, class
balancing, confidence threshold, session logging, replay mode and DB5 benchmark.

### Known issues in the original code (not fixed)
Only relevant for retraining the base model (`train.py`) or TFLite export:
- `train.py` shuffles individual samples before windowing, and train/test share recordings
  (the reported 89% is optimistic)
- `config.inital_lr = 0.2` is too high for Adam (use 0.001 before retraining)
- `tensorflow_lite.py` has hardcoded sizes and overwrites `scaling_params.json`

## Troubleshooting
| Problem | Fix |
|---|---|
| `conda` not recognised | `conda.exe init powershell` + restart VS Code (see Setup) |
| Myo not found | Charge/wake it, close Myo Connect, unpair in Windows, check `--myo-name` |
| Shape mismatch loading checkpoint | `config.filters` must be `[32, 64]` |
| CUDA / nvcuda.dll warnings | Harmless; runs on CPU |
| Gesture often confused | Hold it more consistently, or choose a more distinct gesture |
| Too many false triggers | Raise `--threshold` |
| Output flickers | Raise `--smoothing` |

## Credits and license
Based on **EffiE: Efficient Convolutional Neural Network for Real-Time EMG Pattern Recognition
System on Edge Devices** (IEEE NER 2023), SF State MIC Lab:
https://github.com/MIC-Laboratory/IEEE-NER-2023-EffiE

Pretraining data: **NinaPro DB5** (https://ninapro.hevs.ch/instructions/DB5.html).
If you use it, cite: Pizzolato et al., "Comparison of six electromyography acquisition setups
on hand movement classification tasks", PLOS One, 2017.

---

# Original EffiE README

> Kept unchanged for reference and credit. Parts are outdated for this fork:
> install from `requirements-myo.txt` (not `requirements.txt`), the main entry point is now
> `run_live.py`, and the reported 89% accuracy comes from a train/test split with data
> leakage (see "Known issues" above).


## [IEEE NER 2023 - EffiE: Efficient Convolutional Neural Network for Real-Time EMG Pattern Recognition System on Edge Devices](https://ieeexplore.ieee.org/document/10123741/)

![Fine-Tuning](visuals/finetune.png?raw=true "Fine-Tuning")
* Figure 1: Convolutional Neural Network and Fine-Tuning

## About
This project aims to achieve efficient sEMG gesture recognition through Convolutional Neural Network (CNN) and Finetuning. Finetuning is a technique of transfer learning, which, we aim to generalize model learnings from the larger dataset to a more specific, downstream dataset. In this case, we performed offline training with a large recorded sEMG dataset and retrained it with real-time collected data. Additionally, we compiled our model with Tensorflow Lite, optimized for micro-controller applications.

## Data Collections
To acquire real-time sEMG signals, we utilized Thalmic Lab's Myo Armband, with BLE connections established via. the python Bleak library. The Myo Armband contains 8 channels of raw sEMG signal sampled at 200 Hz. As for offline training data, we adopted the NinaPro DB5, which also uses Myo Armband for data acquisition. From NinaPro DB5, we utilized 7 gestures collected using the Myo Armband closest to elbow from Exercise B: 

    1. Rest (0)
    2. Thumb Up (13)
    3. Flexion of ring and little finger, extension of the others (15)
    4. Abduction of all fingers (17)
    5. Fingers flexed together in fist (18)
    6. Wrist flexion (25)
    7. Wrist Extension (26)

## Quick Start

#### Real Time Gesture Recognition
Let's perform a simple 4 gesture recognition. Enable bluetooth on your PC and have the Myo Armband attached on your arm. Then, run `realtime.py` and follow guidelines printed on the command prompt, to perform the 4 gestures: "Rest", "Fist", "Thumbs Up", and "Ok Sign". After real time sEMG is collected, the pre-trained model from folder "checkpoints" will be finetuned with these samples. Lastly, this model will perform gesture recognition! (Optional: You may also export the finetuned model's weights by controlling variable "finetuned_path")

#### Advanced
If you would like to perform other gestures, you may also finetune with other gestures from NinaPro DB5, simply by controlling the variable "targets" in `config.py`. Then, edit the "GESTURES" in `realtime.py` for realtime gesture recognition upon other gestures.

#### Microcontroller Section
Please refer to this GitHub repo for the prosthesis arm control implementation on microcontrollers: https://github.com/MIC-Laboratory/Real-time-Bionic-Arm-Control-via-CNN-on-Sony-Spresense


![sEMG Data Preprocessing](visuals/data_preprocessing.png?raw=true "sEMG Data Preprocessing")
* Figure 2: sEMG Data Preprocessing

## Data Preprocessing
As shown in the fig.2, the Myo Armband includes 8 channels, so each raw sEMG value is an 8-bit unsigned number ranging 0 - 255, so we change the sEMG values from unsigned to signed by subtracting the value by 256 if the sEMG value is greater than 127. One sEMG sample per channel sampled at 5 ms makes up an sEMG array shaped 1x8. Then, we combine 32 sEMG arrays to create an sEMG window (8 × 32). So, each window is sampled at 160 ms with 80 ms overlapping step of 16. Finally, we will have one set of mean and standard deviation for each of the 8 channels calculated from the sEMG samples, obtained from the NinaPro DB5 dataset gestures used during pre-training. And we subtract each EMG value with the local mean divided by the local standard deviation. File `dataset.py` contains all data preprocessing related code, and file `train.py` lines 19 - 50 performs sample test run.

## Offline Training
According to Wikipedia, Convolutional Neural Network is a class of artificial neural network, most commonly applied to analyze visual imagery. In this case, our "image" will be the 8x32 sEMG window. Therefore, the model input shape is [batch size, 8, 32, 1], batch_size x height x length x channels/depth. (would be 3 if image was RGB format) Our CNN model consists of 2 convolutional layers followed by a fully-connected layer. The first convolutional layer consists of 32-filters, followed by a PReLU activation function, batch normalization to accelerate model training, spatial 2D dropout to counter overfitting, and max pooling. The second convolutional layer is the same as the first layer except that it uses 64 filters instead of 32. Lastly, we will end with N neuron fully-connected layer depending on N gestures utilized during pre-training. File `model.py` contains all utility code regarding CNN, and file `train.py` lines 64 - 79 performs sample test run. To control what gestures to pre-train with from NinaPro DB5, simply edit variable "targets" in `config.py`. (Other hyperparameters also in `config.py`)

###### Special NOTE
To run realtime gesture recognition, follow the "Real Time Gesture Recognition" and "Advanced" section above.

## Tensorflow Lite (optional)
Have the finetuned model's weights stored in folder `finetuned`, and run `tensorflow_lite.py`. Then, you will see the optimized TF-Lite model in folder `tensorflow_lite`.

## Files Overview
1. Folder -> "checkpoints" contains:
    - A pretrained finetune-base model's weights.
    - 89% acc over test and train set for 7 Ninapro DB5 gestures:
        1. relax
        2. thumbs up
        3. flexion
        4. open hand
        5. fist
        6. wrist flexion
        7. wrist extension

2. Folder -> "finetuned":
    - Placeholder for a finetuned model's weights.

3. Folder -> "Ninapro_DB5" contains:
    - The Ninapro DB5 dataset I obtained from https://zenodo.org/record/1000116#.YraeZHaZNLE

4. Folder -> "tensorflow_lite" contains:
    - Placeholder for a compiled finetuned TF Lite model's weights.

5. Folder -> "visuals" contains:
    - Demonstration of how Myo Armband was weared when Ninapro DB5 collected their samples. (visuals\armband_position.png)
    - Overview of gestures and data samples: (visuals\Data.png + visuals\gestures.png)

6. `config.py` contains:
    - Configuration Variables and Parameters

7. `dataset.py` contains:
    - Utilities for extracting and preprocessing sEMG signals data

8. `model.py` contains:
    - Code for A.I. model implementation and utility functions for it.

8. `realtime.py` contains:
    - Main Program for Real-Time system which establishes BLE connection,
        defines GUI, and finetunes realtime samples from a pretrained finetune-base model.
    - OPTIONAL: Export finetuned model's weights

9. `requirements.txt` contains:
    - List of libraries I used through this github repo
    - To replicate my environment, run this under activated conda env: pip install -r requirements.txt

10. `scaling_params.json` contains: (If it exist)
    - Json with MEAN and Standard Deviation for each sensor Channel.

11. `tensorflow_lite.py` contains:
    - Code for compiling Tensorflow model to Tensorflow Lite model.

12. `train.py` contains:
    - A sample run of how to perform:
        - Data Preprocessing over Ninapro DataBase5
        - Training finetune-base model (Saving weights along the way)
        - Visualize training logs (model accuracy and loss during training)
