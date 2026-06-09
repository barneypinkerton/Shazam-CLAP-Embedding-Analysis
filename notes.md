# Technical Notes

## Shazam Evaluation Design

**Full track in DB, 10 s snippet on the query side** — this matches how Shazam actually works.

`build_gtzan_db.py` indexes the full 30 s of each GTZAN track: every hash → (song_id, offset) pair spanning the whole file is stored. `evaluate_shazam.py` cuts a random 10 s window from the augmented file and queries the DB. The matcher's time-coherence scoring (`identify.py`) recovers the position by finding the dominant (song_id, diff) bucket, where `diff = db_offset - snippet_offset`.

This asymmetry is the point: indexing is one-time and exhaustive; queries are short and cheap. Indexing only 10 s of each original would discard 2/3 of the reference hashes and break the scenario Shazam is designed for (snippet from anywhere in the song).

## Experimental Bias Note

Because the augmented WAVs are processed offline and the Shazam evaluation uses a clean-audio DB, the setup gives Shazam a slight advantage: no codec artifacts (MP3/AAC re-encoding) and no room impulse response on the snippet. Since both methods are evaluated against the same augmented files, the bias applies equally to both, and relative claims remain valid. Absolute accuracy claims should acknowledge this.

Cheap mitigations if absolute accuracy matters:
- Add an MP3/AAC re-encode pass to the snippet: `ffmpeg -i in.wav -b:a 64k out.aac && ffmpeg -i out.aac out.wav`
- Convolve with a short room IR before mixing noise (`scipy.signal.fftconvolve`)

## Key Findings

**Shazam**: ~91% on additive noise at all SNR levels. Drops to 0% on all pitch-shift conditions — a 1-semitone shift destroys every fingerprint hash. Lo-fi filter gives 6–8% (near-failure) because the bandpass removes the high-frequency peaks the fingerprints depend on.

**CLAP General**: Degrades gracefully across both noise and transforms. Exact retrieval: 61–77% Top-1 at 20 dB SNR, 22–33% on pitch shift, 28–60% on lo-fi. Genre classification: 86–92% at 20 dB SNR, 52–62% on pitch shift, 64–80% on lo-fi. The only system with non-zero pitch-shift robustness — a side-effect of broad training rather than pitch-invariant design. High wrong_song_right_genre_rate (38–46%) on transforms confirms semantic retrieval rather than exact matching.

**CLAP Music**: Collapses completely on transforms. Exact retrieval ~0%, genre ~10% (random chance across 10 genres). Cosine similarity falls from ~0.99 on clean audio to ~0.048 on pitch-shifted audio — the embedding lands in an empty region of the space, then predicts "metal" for every query (embedding space collapse, not genuine detection). Fine-tuning on music increased sensitivity to exact tonal features, which is catastrophic when those features are altered.

**Headline**: Shazam and CLAP Music share the same fatal weakness — exact acoustic matching. CLAP General, a generalist model not designed for music retrieval, is the most transform-robust system. Broader training > specialist fine-tuning for robustness to musical transforms.
