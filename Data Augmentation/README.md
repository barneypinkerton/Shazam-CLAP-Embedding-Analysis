# Data Augmentation Tool

## Overview

A reusable tool for augmenting audio datasets with noise mixing and musical transforms. Available as both a CLI and a GUI. Designed for evaluating audio recognition robustness under degraded conditions.

Supports two categories of augmentation:
- **Additive noise** — mixes real-world or synthetic noise into the signal at a configurable SNR level.
- **Musical transforms** — applies pitch shifting or a lo-fi bandpass filter without adding noise.

## Installation

```bash
pip install numpy soundfile tqdm
```

Tkinter is required for the GUI but ships with Python on most platforms.

## GUI

```bash
python augment_gui.py
```

The GUI provides:
- File browser dialogs for input/output directories and noise files
- A noise source list — add white noise or browse for noise files, remove entries
- Editable SNR values (comma-separated)
- Snippet duration and random seed fields
- A progress bar with real-time status
- A log panel showing per-file processing output
- Start / Cancel controls

## CLI

```bash
python augment.py --input <input_dir> --output <output_dir> \
    --noise white \
    --noise file:crowd_noise:"path/to/crowd noise.wav" \
    --noise file:street_noise:"path/to/street noise.wav" \
    --snr 20 10 0
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--input` | Yes | Path to source audio directory (expects `genre/file.wav` structure) |
| `--output` | Yes | Path to write augmented files |
| `--noise` | Yes | Noise source(s). Repeatable. `white` for generated white noise, or `file:<name>:<path>` for a noise file |
| `--snr` | Yes | SNR level(s) in dB. Repeatable. e.g. `--snr 20 10 0`. See [SNR reference](#snr-reference) below |
| `--aug` | Yes (transforms) | Transform type. `pitch_up`, `pitch_down` (semitones via `--levels`), or `lofi` (bandpass severity via `--levels`). Repeatable. |
| `--levels` | Yes (transforms) | Severity levels for `--aug`. For pitch: semitones e.g. `1 2 3`. For lo-fi: severity `1 2 3`. |
| `--snippet-duration` | No | Duration in seconds to extract from noise files (default: 30) |
| `--seed` | No | Random seed for reproducibility |
| `--workers` | No | Number of parallel workers (default: 4). Controls how many audio files are processed simultaneously using separate CPU cores. Higher values speed up processing on multi-core machines but use more memory. A value of 4 means 4 files are being read, mixed, and written at the same time. |

### Examples

**Default run (white + crowd + street at 20/10/0 dB):**
```bash
python augment.py \
    --input "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_original" \
    --output "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_augmented" \
    --noise white \
    --noise "file:crowd_noise:crowd noise.wav" \
    --noise "file:street_noise:street noise.wav" \
    --snr 20 10 0
```

**Add a new noise type later:**
```bash
python augment.py \
    --input "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_original" \
    --output "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_augmented" \
    --noise "file:rain_noise:rain.wav" \
    --snr 15 5
```

**White noise only at custom SNRs:**
```bash
python augment.py \
    --input "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_original" \
    --output "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_augmented" \
    --noise white \
    --snr 30 20 10 5 0
```

**Pitch shift (up and down, levels 1–3 semitones):**
```bash
python augment.py \
    --input "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_original" \
    --output "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_augmented" \
    --aug pitch_up \
    --aug pitch_down \
    --levels 1 2 3
```

**Lo-fi bandpass filter (levels 1–3):**
```bash
python augment.py \
    --input "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_original" \
    --output "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_augmented" \
    --aug lofi \
    --levels 1 2 3
```

## Output Structure

```
genres_augmented/
├── white_noise/
│   ├── 20dB/
│   │   ├── blues/
│   │   │   ├── blues.00000.wav
│   │   │   └── ...
│   │   ├── classical/
│   │   └── ... (all genres)
│   ├── 10dB/
│   └── 0dB/
├── crowd_noise/
│   ├── 20dB/
│   ├── 10dB/
│   └── 0dB/
└── street_noise/
    ├── 20dB/
    ├── 10dB/
    └── 0dB/
├── pitch_shift_up/
│   ├── 1/
│   ├── 2/
│   └── 3/
├── pitch_shift_down/
│   ├── 1/
│   ├── 2/
│   └── 3/
└── lofi_filter/
    ├── 1/
    ├── 2/
    └── 3/
```

For transforms, the folder number is the severity level (semitones for pitch shift, filter preset for lo-fi).

## SNR Reference

SNR (signal-to-noise ratio) measures how loud the original audio is relative to the added noise, in decibels (dB). Higher values mean cleaner audio, lower values mean noisier.

| SNR (dB) | Noise Level | Real-World Analogy |
|---|---|---|
| 20 dB | Signal is 10x louder than noise | Listening to music in a quiet room — noise barely audible |
| 10 dB | Signal is ~3x louder than noise | Listening in a busy cafe — noise clearly noticeable |
| 0 dB | Signal and noise are equal power | Trying to hear someone talk at a concert — equal loudness |
| -5 dB | Noise is louder than signal | Noise dominates, signal hard to pick out |

The three default levels (20, 10, 0) give a spread from "barely degraded" to "severely degraded", useful for testing how audio recognition holds up as conditions get worse.

## Architecture

```
augment_core.py   — shared augmentation logic (noise mixing, SNR scaling, file I/O)
augment.py        — CLI entry point (argparse)
augment_gui.py    — GUI entry point (tkinter)
```

Both the CLI and GUI call into `augment_core.py`, so all signal processing logic is in one place.

## Technical Details

- Noise files are resampled to match each recording's sample rate and channel count
- File-based noise: a fixed snippet is extracted once from the center of the source file and reused for all recordings (removes bias)
- White noise: generated fresh per file using NumPy (seeded for reproducibility)
- Existing files are skipped (safe to re-run with new noise types)
- Corrupt or unreadable files are skipped with a warning rather than halting the pipeline
- Pitch shift uses `librosa.effects.pitch_shift`; level N shifts by ±N semitones
- Lo-fi filter is a 4th-order Butterworth bandpass (`scipy.signal.butter` + `sosfiltfilt`):

| Level | High-pass | Low-pass | Character |
|---|---|---|---|
| 1 | 300 Hz | 8000 Hz | Mild — trims sub-bass and air |
| 2 | 400 Hz | 6000 Hz | Moderate — telephone / laptop speaker |
| 3 | 500 Hz | 4000 Hz | Severe — AM radio |

### How SNR Mixing Works

The core mixing logic (`mix_at_snr` in `augment_core.py`) works in four steps:

1. **Compute RMS** — the root mean square of both the original signal and the noise is calculated. RMS represents the average "loudness" (power) of a waveform.

2. **Calculate a scale factor** — the noise is scaled so its power relative to the signal matches the target SNR:
   ```
   scale = (sig_rms / noise_rms) * 10^(-SNR_dB / 20)
   ```
   - `sig_rms / noise_rms` first normalizes the noise to match the signal's power (equivalent to 0 dB SNR — equal loudness).
   - `10^(-SNR_dB / 20)` then attenuates by the target decibel amount:
     - **20 dB** → noise is 1/10th signal power (barely audible)
     - **10 dB** → noise is ~1/3rd signal power (clearly noticeable)
     - **0 dB** → noise equals signal power (equal loudness)

3. **Add sample-by-sample** — the scaled noise is added to the original signal: `mixed = signal + noise * scale`

4. **Clip** — the result is clamped to [-1.0, 1.0] to prevent overflow when saving as 16-bit WAV.

Because the scale factor is always relative to each recording's actual loudness, a quiet classical piece and a loud metal track both receive the same perceptual noise level at a given SNR — rather than a fixed amount of noise that would drown out quiet recordings but be inaudible on loud ones.

## Known Issues — GTZAN Dataset

The following file in the GTZAN dataset is corrupt and cannot be read by libsndfile:

- `jazz/jazz.00054.wav` — "Format not recognised"

This is a known issue with the GTZAN dataset. The file is automatically skipped during augmentation, resulting in 999 files per noise/SNR combination instead of 1,000 (8,991 total instead of 9,000).
