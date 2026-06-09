"""Core augmentation engine shared by CLI and GUI."""

import os
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Callable

import librosa
from scipy.signal import butter, sosfiltfilt

# Lo-fi bandpass filter frequencies (Hz) keyed by severity level.
# Each entry is (high_pass_hz, low_pass_hz).
# Level 1 (300–8000 Hz): mild — trims sub-bass and air, remains clear.
# Level 2 (400–6000 Hz): moderate — telephone / laptop-speaker quality.
# Level 3 (500–4000 Hz): severe — heavily degraded, AM-radio character.
_LOFI_BANDPASS: dict[int, tuple[int, int]] = {1: (300, 8000), 2: (400, 6000), 3: (500, 4000)}


@dataclass
class AugSource:
    name: str
    kind: str  # "white", "file", "pitch_shift_up", "pitch_shift_down", or "lofi"
    path: Optional[str] = None  # only for kind="file"
    snippet: Optional[np.ndarray] = None  # preloaded snippet (mono, target sr)
    snippet_sr: Optional[int] = None


def load_source_snippet(path: str, snippet_duration: float) -> tuple[np.ndarray, int]:
    """Load a source audio file and extract a centered snippet."""
    data, sr = sf.read(path, dtype="float64")
    # Convert to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)
    snippet_samples = int(snippet_duration * sr)
    if len(data) <= snippet_samples:
        return data, sr
    # Extract from center
    center = len(data) // 2
    start = center - snippet_samples // 2
    return data[start:start + snippet_samples], sr


def resample_simple(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample using linear interpolation (no extra deps)."""
    if orig_sr == target_sr:
        return data
    ratio = target_sr / orig_sr
    new_len = int(len(data) * ratio)
    indices = np.linspace(0, len(data) - 1, new_len)
    return np.interp(indices, np.arange(len(data)), data)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise into signal at a given SNR (dB).

    Steps:
    1. Compute RMS (root mean square) of both the signal and noise — this
       represents the average "loudness" / power of each waveform.
    2. Calculate a scale factor that adjusts the noise volume relative to the
       signal so that their power ratio matches the target SNR:
       - (sig_rms / noise_rms) normalizes noise to equal signal power (0 dB).
       - 10^(-snr_db/20) then attenuates by the desired SNR in decibels.
       - At 20 dB: noise is 1/10th the signal power (barely audible).
       - At 10 dB: noise is ~1/3rd the signal power (clearly noticeable).
       - At  0 dB: noise equals the signal power (equal loudness).
    3. Add the scaled noise to the signal sample-by-sample.
    4. Clip to [-1.0, 1.0] to prevent overflow when saving as 16-bit WAV.

    Because the scale factor is relative to each recording's actual loudness,
    a quiet classical piece and a loud metal track both receive the same
    perceptual noise level at a given SNR.
    """

    # Step 1: Compute RMS (average power) for both the original signal and the noise
    sig_rms = np.sqrt(np.mean(signal ** 2))
    noise_rms = np.sqrt(np.mean(noise ** 2))

    # If either signal is silent, mixing is meaningless — return the original
    if noise_rms == 0 or sig_rms == 0:
        return signal

    # Step 2: Scale noise so its power relative to the signal matches the target SNR.
    # (sig_rms / noise_rms) normalizes noise to equal signal power (0 dB SNR).
    # 10^(-snr_db/20) then attenuates by the desired number of decibels.
    scale = (sig_rms / noise_rms) * (10 ** (-snr_db / 20))

    # Step 3: Add the scaled noise to the original signal sample-by-sample
    mixed = signal + noise * scale

    # Step 4: Clip to [-1.0, 1.0] to prevent overflow in 16-bit WAV output
    mixed = np.clip(mixed, -1.0, 1.0)

    return mixed


def _process_single(args: tuple) -> str:
    """Process a single file — designed for multiprocessing."""
    input_path, output_path, aug_kind, source_data, source_sr, level, seed_offset = args

    # Note: jazz.00054.wav in the GTZAN dataset is corrupt ("Format not recognised")
    # and will be caught here and skipped.
    try:
        signal, sr = sf.read(input_path, dtype="float64")
    except Exception as e:
        return f"SKIP:{input_path}:{e}"

    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    if aug_kind == "white":
        rng = np.random.default_rng(seed_offset)
        noise = rng.normal(0, 1, len(signal))
        mixed = mix_at_snr(signal, noise, level)
    elif aug_kind == "pitch_shift_up":
        shifted = librosa.effects.pitch_shift(signal.astype(np.float32), sr=sr, n_steps=float(level))
        mixed = np.clip(shifted.astype(np.float64), -1.0, 1.0)
    elif aug_kind == "pitch_shift_down":
        shifted = librosa.effects.pitch_shift(signal.astype(np.float32), sr=sr, n_steps=-float(level))
        mixed = np.clip(shifted.astype(np.float64), -1.0, 1.0)
    elif aug_kind == "lofi":
        hp_hz, lp_hz = _LOFI_BANDPASS[int(level)]
        nyq = sr / 2.0
        sos = butter(4, [hp_hz / nyq, lp_hz / nyq], btype="band", output="sos")
        mixed = np.clip(sosfiltfilt(sos, signal), -1.0, 1.0)
    else:
        # File-based noise: resample snippet to match signal sr then mix at SNR
        noise = resample_simple(source_data, source_sr, sr)
        # Match length: tile or truncate
        if len(noise) < len(signal):
            repeats = (len(signal) // len(noise)) + 1
            noise = np.tile(noise, repeats)
        noise = noise[:len(signal)]
        mixed = mix_at_snr(signal, noise, level)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, mixed, sr, subtype="PCM_16")
    return output_path


def collect_audio_files(input_dir: str) -> list[tuple[str, str, str]]:
    """Collect all wav files. Returns list of (full_path, genre, filename)."""
    files = []
    for genre in sorted(os.listdir(input_dir)):
        genre_dir = os.path.join(input_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        for fname in sorted(os.listdir(genre_dir)):
            if fname.lower().endswith(".wav"):
                files.append((os.path.join(genre_dir, fname), genre, fname))
    return files


def build_job_list(
    input_dir: str,
    output_dir: str,
    sources: list[AugSource],
    levels: list[float],
    seed: Optional[int] = None,
    skip_existing: bool = True,
) -> list[tuple]:
    """Build the full list of (input, output, aug_kind, source_data, source_sr, level, seed) jobs."""
    files = collect_audio_files(input_dir)
    jobs = []
    file_idx = 0

    # Cartesian product: every augmentation source x every level x every audio file.
    # For N files, M sources, and K levels this produces N * M * K jobs.
    # e.g. 1000 files x 3 aug types x 3 levels = 9000 augmented files.
    # SNR-based types use "20dB" notation (scientifically meaningful, matches
    # existing data on disk). Transform-based types use plain numbers ("1", "2",
    # "3") since dB is not the right unit for semitones or lo-fi severity.
    _SNR_KINDS = ("white", "file")

    for src in sources:
        for level in levels:
            if src.kind in _SNR_KINDS:
                level_label = f"{int(level)}dB" if level == int(level) else f"{level}dB"
            else:
                level_label = f"{int(level)}" if level == int(level) else str(level)
            for input_path, genre, fname in files:
                out_path = os.path.join(output_dir, src.name, level_label, genre, fname)
                if skip_existing and os.path.exists(out_path):
                    continue
                seed_offset = (seed + file_idx) if seed is not None else None
                if src.kind in ("white", "pitch_shift_up", "pitch_shift_down", "lofi"):
                    jobs.append((input_path, out_path, src.kind, None, None, level, seed_offset))
                else:
                    jobs.append((input_path, out_path, "file", src.snippet, src.snippet_sr, level, seed_offset))
                file_idx += 1

    return jobs


def run_augmentation(
    input_dir: str,
    output_dir: str,
    sources: list[AugSource],
    levels: list[float],
    snippet_duration: float = 30.0,
    seed: Optional[int] = None,
    workers: int = 4,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> int:
    """
    Run the full augmentation pipeline.

    progress_callback(completed, total, last_file) is called after each file.
    cancel_check() should return True to abort.
    Returns number of files processed.
    """
    # Preload file-based source snippets
    for src in sources:
        if src.kind == "file" and src.snippet is None:
            src.snippet, src.snippet_sr = load_source_snippet(src.path, snippet_duration)

    jobs = build_job_list(input_dir, output_dir, sources, levels, seed)
    total = len(jobs)
    if total == 0:
        if progress_callback:
            progress_callback(0, 0, "Nothing to do — all files exist")
        return 0

    completed = 0
    skipped = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_single, job): job for job in jobs}
        for future in as_completed(futures):
            if cancel_check and cancel_check():
                pool.shutdown(wait=False, cancel_futures=True)
                return completed
            result = future.result()
            if result.startswith("SKIP:"):
                skipped.append(result)
                if progress_callback:
                    progress_callback(completed, total, f"[skipped] {result}")
            else:
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, result)

    if skipped and progress_callback:
        progress_callback(completed, total, f"Done. {len(skipped)} files skipped due to errors.")

    return completed
