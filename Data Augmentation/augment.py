#!/usr/bin/env python3
"""CLI entry point for audio data augmentation."""

import argparse
import sys
from tqdm import tqdm
from augment_core import AugSource, run_augmentation


def parse_aug(spec: str) -> AugSource:
    """Parse a --aug argument into an AugSource."""
    if spec.lower() == "white":
        return AugSource(name="white_noise", kind="white")
    if spec.lower() == "pitch_up":
        return AugSource(name="pitch_shift_up", kind="pitch_shift_up")
    if spec.lower() == "pitch_down":
        return AugSource(name="pitch_shift_down", kind="pitch_shift_down")
    if spec.lower() == "lofi":
        return AugSource(name="lofi", kind="lofi")
    if spec.startswith("file:"):
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"File source must be file:<name>:<path>, got: {spec}"
            )
        _, name, path = parts
        return AugSource(name=name, kind="file", path=path)
    raise argparse.ArgumentTypeError(
        f"Unknown augmentation spec: {spec}. Use 'white', 'pitch_up', 'pitch_down', 'lofi', "
        f"or 'file:<name>:<path>'"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Augment audio datasets with various transformations at configurable levels."
    )
    parser.add_argument("--input", required=True, help="Input audio directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--aug", action="append", required=True,
                        help=(
                            "Augmentation source. Repeatable. Options: "
                            "'white' (white noise mixed at --levels dB), "
                            "'pitch_up' (pitch shift up by --levels semitones), "
                            "'pitch_down' (pitch shift down by --levels semitones), "
                            "'lofi' (bandpass filter: L1=300–8kHz, L2=400–6kHz, L3=500–4kHz; --levels selects severity 1/2/3), "
                            "'file:<name>:<path>' (external noise file mixed at --levels dB)."
                        ))
    parser.add_argument("--levels", nargs="+", type=float, required=True,
                        help=(
                            "Augmentation levels. Meaning depends on --aug type: "
                            "dB for noise types (e.g. 20 10 0), "
                            "semitones for pitch shift (e.g. 1 2 3), "
                            "severity 1/2/3 for lofi."
                        ))
    parser.add_argument("--snippet-duration", type=float, default=30.0,
                        help="Seconds to extract from file-based sources (default: 30)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")

    args = parser.parse_args()

    sources = [parse_aug(spec) for spec in args.aug]

    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Sources: {', '.join(src.name for src in sources)}")
    print(f"Levels:  {', '.join(str(l) for l in args.levels)}")
    print(f"Workers: {args.workers}")
    print()

    pbar = tqdm(total=0, unit="file", dynamic_ncols=True)

    def progress(completed, total, last_file):
        if pbar.total != total:
            pbar.total = total
            pbar.refresh()
        pbar.update(1)
        pbar.set_postfix_str(last_file.split("/")[-1], refresh=False)

    count = run_augmentation(
        input_dir=args.input,
        output_dir=args.output,
        sources=sources,
        levels=args.levels,
        snippet_duration=args.snippet_duration,
        seed=args.seed,
        workers=args.workers,
        progress_callback=progress,
    )

    pbar.close()
    print(f"\nDone. {count} files written.")


if __name__ == "__main__":
    main()
