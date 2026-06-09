"""Extract CLAP audio embeddings for a directory of audio files.

Supports both the general-purpose LAION-CLAP checkpoint (630k-audioset-best.pt)
and the music-specialist checkpoint (music_audioset_epoch_15_esc_90.14.pt).

Usage examples:
    # General-purpose model
    python embed.py \
        --input  /path/to/Data/genres_augmented \
        --output /path/to/Embeddings/CLAP_general \
        --checkpoint /path/to/630k-audioset-best.pt

    # Music specialist model (requires --music-model flag for custom loading)
    python embed.py \
        --input  /path/to/Data/genres_augmented \
        --output /path/to/Embeddings/CLAP_music \
        --checkpoint /path/to/music_audioset_epoch_15_esc_90.14.pt \
        --music-model
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import laion_clap

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def collect_audio_files(root: Path) -> list[Path]:
    if not root.exists():
        raise ValueError(f"Input path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected a directory, got: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise ValueError(f"No supported audio files found in: {root}")
    return files


def load_model(checkpoint: Path | None, music_model: bool, use_fusion: bool) -> laion_clap.CLAP_Module:
    model = laion_clap.CLAP_Module(enable_fusion=use_fusion)
    if checkpoint is None:
        print("Loading default laion-clap checkpoint...")
        model.load_ckpt()
    elif music_model:
        # The music checkpoint has dimension mismatches with the default model spec;
        # load the state dict directly with strict=False to skip mismatched keys.
        print(f"Loading music checkpoint (strict=False): {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint (embeddings will be 512-dimensional)")
    else:
        print(f"Loading checkpoint: {checkpoint}")
        model.load_ckpt(ckpt=str(checkpoint))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="Root directory of audio files to embed.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Root directory for output .npy embedding files.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a CLAP checkpoint file. Omit to use the default.")
    parser.add_argument("--music-model", action="store_true",
                        help="Use manual torch.load + strict=False loading for the music checkpoint.")
    parser.add_argument("--fusion", action="store_true",
                        help="Enable CLAP fusion (enable_fusion=True).")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files(input_root)
    print(f"Found {len(audio_files)} audio file(s).")

    model = load_model(args.checkpoint, args.music_model, args.fusion)

    manifest = []
    for start in range(0, len(audio_files), args.batch_size):
        batch = audio_files[start:start + args.batch_size]
        batch_paths = [str(p) for p in batch]
        print(f"Embedding files {start + 1}-{start + len(batch)} / {len(audio_files)}")

        embeds = model.get_audio_embedding_from_filelist(x=batch_paths, use_tensor=False)

        for audio_path, emb in zip(batch, embeds):
            rel_path = audio_path.relative_to(input_root)
            out_path = output_root / rel_path.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, emb.astype(np.float32))

            manifest.append({
                "audio_path": str(audio_path),
                "relative_audio_path": str(rel_path),
                "embedding_path": str(out_path),
                "relative_embedding_path": str(out_path.relative_to(output_root)),
                "embedding_shape": list(emb.shape),
            })

    manifest_path = output_root / "embedding_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "input_root": str(input_root),
            "output_root": str(output_root),
            "checkpoint": str(args.checkpoint) if args.checkpoint else "default laion-clap checkpoint",
            "fusion": args.fusion,
            "music_model": args.music_model,
            "num_files": len(manifest),
            "items": manifest,
        }, f, indent=2)

    print(f"Saved per-file embeddings under: {output_root}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
