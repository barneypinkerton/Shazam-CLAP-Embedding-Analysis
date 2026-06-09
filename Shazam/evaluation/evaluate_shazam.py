#!/usr/bin/env python3
"""Evaluate Shazam-style identification against augmented GTZAN.

Default mode walks `{augmented_root}/{aug_type}/{level}dB/{genre}/{name}.wav`.
Pass `--clean-baseline` to instead walk `{originals_root}/{genre}/{name}.wav`,
which establishes the no-augmentation reference accuracy. Clean-baseline rows are
written with `aug_type="clean"` and `level=999`, and use the same
per-filename snippet-start seed as augmented runs so windows align across
all conditions for the same track.

For every augmented file `{augmented_root}/{aug_type}/{level}dB/{genre}/{name}.wav`:
  1. Pick a deterministic random 10s window inside the 30s clip.
     The window position is seeded from the source filename only, so the
     SAME window is used across every (aug_type, level) condition for a given
     track. This isolates the effect of augmentation from snippet-position luck.
  2. Write the snippet to a temp WAV.
  3. Run identify_audio() against fingerprints_gtzan.db, timed with
     time.perf_counter().
  4. Compare the returned top-match name to the ground-truth filename
     (GTZAN's `genre.NNNNN.wav` convention means the augmented filename
     is identical to the original).
  5. Append one row to the results CSV.

Resume-safe: rows whose (aug_type, level, genre, filename) tuple is already
present in the CSV are skipped on re-run.
"""
import argparse
import csv
import hashlib
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# load_audio: file path -> (sample_rate, float32 numpy array). No playback.
# identify_audio: the Shazam matcher. Takes a WAV path, runs the full
# fingerprint+lookup+time-coherence pipeline, returns dict or None.
from src.audioprocessing import load_audio
from src.identify import identify_audio


DEFAULT_AUGMENTED = "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_augmented"
DEFAULT_ORIGINALS = "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_original"
DEFAULT_DB = str(Path(__file__).resolve().parent / "fingerprints_gtzan.db")
DEFAULT_OUT = str(Path(__file__).resolve().parent / "results" / "shazam_eval.csv")
DEFAULT_SNIPPET_SECONDS = 10.0
DEFAULT_MASTER_SEED = 20260425

# Sentinel values used when running the clean baseline (no augmentation).
# level = 999 sorts cleanly to one end of plots; aug_type = "clean" reads
# naturally in the CSV and notebook.
CLEAN_AUG_TYPE = "clean"
CLEAN_LEVEL_SENTINEL = 999

FIELDNAMES = [
    "aug_type",
    "level",
    "genre",
    "filename",
    "ground_truth_name",
    "snippet_start_s",
    "snippet_seconds",
    "identified",
    "correct",
    "elapsed_s",
    "top_match_name",
    "score",
    "confidence",
    "query_fingerprints",
]


def snippet_start_seconds(filename: str, snippet_seconds: float,
                          total_seconds: float, master_seed: int) -> float:
    """Pick a deterministic random snippet start.

    Seeded from `master_seed` + `filename` only — independent of aug_type/level
    so each underlying track uses the same window across every condition.
    """
    seed_str = f"{master_seed}|{filename}"
    seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed_int)
    max_start = max(0.0, total_seconds - snippet_seconds)
    return rng.uniform(0.0, max_start)


def write_snippet_temp(audio: np.ndarray, sr: int, start_s: float,
                       duration_s: float) -> str:
    # Slice the in-memory waveform and persist as 16-bit PCM. We write to disk
    # because identify_audio() takes a file path (matches the web-UI call site)
    # — there is no audio playback anywhere in this pipeline.
    start_idx = int(start_s * sr)
    end_idx = min(int((start_s + duration_s) * sr), len(audio))
    snippet = audio[start_idx:end_idx]
    snippet_int16 = np.clip(snippet, -32768, 32767).astype(np.int16)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="shazam_eval_")
    os.close(fd)
    wav.write(tmp_path, int(sr), snippet_int16)
    return tmp_path


def parse_augmented_path(path: str, augmented_root: str):
    """Expect `{augmented_root}/{aug_type}/{level_folder}/{genre}/{filename}`.

    level_folder is either "{n}dB" (SNR-based noise, e.g. "20dB") or a plain
    integer string (transform-based types, e.g. "1" for pitch/lofi).
    """
    rel = os.path.relpath(path, augmented_root)
    parts = rel.split(os.sep)
    if len(parts) != 4:
        return None
    aug_type, level_str, genre, filename = parts
    if level_str.lower().endswith("db"):
        try:
            level = int(level_str[:-2])
        except ValueError:
            return None
    else:
        try:
            level = int(level_str)
        except ValueError:
            return None
    return aug_type, level, genre, filename


def parse_original_path(path: str, originals_root: str):
    """Expect `{originals_root}/{genre}/{filename}` — one level shallower
    than the augmented layout. Returns the same 4-tuple shape as
    parse_augmented_path, with sentinel aug_type/level values so downstream
    code can stay uniform."""
    rel = os.path.relpath(path, originals_root)
    parts = rel.split(os.sep)
    if len(parts) != 2:
        return None
    genre, filename = parts
    return CLEAN_AUG_TYPE, CLEAN_LEVEL_SENTINEL, genre, filename


def collect_wavs(root: str):
    files = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(dirpath, f))
    files.sort()
    return files


# Back-compat alias — earlier code/imports may still reference this name.
collect_augmented_files = collect_wavs


def load_already_done(csv_path: str):
    seen = set()
    if not os.path.exists(csv_path):
        return seen
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seen.add((row["aug_type"], int(row["level"]),
                          row["genre"], row["filename"]))
            except (KeyError, ValueError):
                continue
    return seen


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--augmented-root", default=DEFAULT_AUGMENTED)
    p.add_argument("--originals-root", default=DEFAULT_ORIGINALS,
                   help="Used only with --clean-baseline.")
    p.add_argument("--clean-baseline", action="store_true",
                   help="Evaluate against the clean originals instead of the "
                        "augmented set. Rows are written with aug_type='clean' "
                        f"and level={CLEAN_LEVEL_SENTINEL}. Snippet-start seeding "
                        "uses the same per-filename hash as augmented runs, so "
                        "each track's clean snippet aligns with its augmented "
                        "counterparts in the same CSV.")
    p.add_argument("--db", default=DEFAULT_DB,
                   help="Fingerprint DB built by build_gtzan_db.py")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="Results CSV (appended in resume mode)")
    p.add_argument("--snippet-seconds", type=float, default=DEFAULT_SNIPPET_SECONDS)
    p.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N files (debug / smoke test)")
    args = p.parse_args()

    # Pick mode-specific root + path parser.
    if args.clean_baseline:
        scan_root = args.originals_root
        parse_path = lambda f: parse_original_path(f, scan_root)
    else:
        scan_root = args.augmented_root
        parse_path = lambda f: parse_augmented_path(f, scan_root)

    if not os.path.isdir(scan_root):
        sys.exit(f"Not a directory: {scan_root}")
    if not os.path.exists(args.db):
        sys.exit(f"Fingerprint DB missing: {args.db}\n"
                 f"Run build_gtzan_db.py first.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    files = collect_wavs(scan_root)
    if args.limit is not None:
        files = files[: args.limit]

    seen = load_already_done(args.out)
    pending = []
    for f in files:
        parsed = parse_path(f)
        if parsed is None or parsed in seen:
            continue
        pending.append(f)

    print(f"Augmented files found: {len(files)}")
    print(f"Already evaluated:     {len(seen)}")
    print(f"To process this run:   {len(pending)}")

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            out_f.flush()

        # === Per-file evaluation loop ===
        # For each augmented WAV: load -> slice 10s window -> write temp WAV
        # -> hand path to identify_audio() -> compare result to ground truth
        # -> append CSV row. No audio is ever played; everything is file I/O.
        for idx, filepath in enumerate(pending, 1):
            # --- Step 1: parse condition + ground truth from the path ---
            # GTZAN preserves filenames through augmentation, so the augmented
            # filename IS the ground-truth original name. In --clean-baseline
            # mode aug_type is "clean" and level is CLEAN_LEVEL_SENTINEL.
            parsed = parse_path(filepath)
            if parsed is None:
                continue
            aug_type, level, genre, filename = parsed

            # --- Step 2: read the augmented WAV from disk into memory ---
            try:
                sr, data = load_audio(filepath)
            except Exception as e:
                print(f"[{idx}/{len(pending)}] LOAD FAIL {filepath}: {e}")
                continue
            if data.ndim > 1:
                data = data.mean(axis=1)
            total_seconds = len(data) / sr

            # --- Step 3: pick the 10s window ---
            # Deterministic per filename; same window across every aug_type/level
            # condition for this track, so augmentation is the only varied factor.
            start_s = snippet_start_seconds(filename, args.snippet_seconds,
                                            total_seconds, args.master_seed)

            # --- Step 4: materialize the snippet as a temp WAV on disk ---
            try:
                snippet_path = write_snippet_temp(data, sr, start_s,
                                                  args.snippet_seconds)
            except Exception as e:
                print(f"[{idx}/{len(pending)}] SNIPPET FAIL {filepath}: {e}")
                continue

            # --- Step 5: THE ALGORITHM CALL ---
            # identify_audio() reads the temp WAV, fingerprints it, queries
            # fingerprints_gtzan.db, runs time-coherence scoring, applies the
            # confidence thresholds in Shazam/.env, and returns the top match
            # dict (or None if no match passes the gates). perf_counter wraps
            # the whole identification path for the elapsed_s metric.
            try:
                t0 = time.perf_counter()
                result = identify_audio(snippet_path, db_path=args.db)
                elapsed = time.perf_counter() - t0
            finally:
                # Always clean up the temp WAV, even if identify_audio raised.
                try:
                    os.remove(snippet_path)
                except OSError:
                    pass

            # --- Step 6: build the CSV row ---
            # `identified` = did the matcher return anything at all.
            # `correct`    = did the returned name equal the ground truth.
            if result is None:
                row = {
                    "aug_type": aug_type, "level": level,
                    "genre": genre, "filename": filename,
                    "ground_truth_name": filename,
                    "snippet_start_s": round(start_s, 3),
                    "snippet_seconds": args.snippet_seconds,
                    "identified": "no", "correct": "no",
                    "elapsed_s": round(elapsed, 4),
                    "top_match_name": "", "score": "",
                    "confidence": "", "query_fingerprints": "",
                }
            else:
                correct = "yes" if result["name"] == filename else "no"
                row = {
                    "aug_type": aug_type, "level": level,
                    "genre": genre, "filename": filename,
                    "ground_truth_name": filename,
                    "snippet_start_s": round(start_s, 3),
                    "snippet_seconds": args.snippet_seconds,
                    "identified": "yes", "correct": correct,
                    "elapsed_s": round(elapsed, 4),
                    "top_match_name": result["name"],
                    "score": result["score"],
                    "confidence": result["confidence"],
                    "query_fingerprints": result["query_fingerprints"],
                }

            # --- Step 7: persist the row immediately ---
            # flush() guarantees crash-safety: a kill mid-run loses at most
            # the in-flight file, and the resume check on next run picks up
            # exactly where we left off.
            writer.writerow(row)
            out_f.flush()
            print(f"[{idx}/{len(pending)}] {aug_type}/{level}dB/{genre}/{filename} "
                  f"-> identified={row['identified']} correct={row['correct']} "
                  f"elapsed={elapsed:.3f}s")


if __name__ == "__main__":
    main()
