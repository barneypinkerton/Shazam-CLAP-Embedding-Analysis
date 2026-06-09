"""Unified evaluation entry-point for the Shazam / CLAP analysis.

Subcommands
-----------
summary     Dataset overview plots (per-genre counts, augmentation breakdown).
genre       Genre classification: train on clean embeddings, test on augmented.
retrieval   Exact-song retrieval: cosine similarity search.

Each subcommand forwards its remaining arguments to the underlying script, so
all flags documented in the individual scripts work unchanged here.

Examples
--------
    python evaluate.py summary \\
        --data-root /path/to/Embeddings/CLAP_general

    python evaluate.py genre \\
        --embedding-root /path/to/Embeddings/CLAP_general \\
        --embedding-root /path/to/Embeddings/CLAP_music \\
        --model-label "CLAP General" \\
        --model-label "CLAP Music"

    python evaluate.py retrieval \\
        --embedding-root /path/to/Embeddings/CLAP_general \\
        --embedding-root /path/to/Embeddings/CLAP_music \\
        --model-label "CLAP General" \\
        --model-label "CLAP Music"
"""

import sys
import argparse
import importlib
from pathlib import Path

# Map subcommand name → module file under "Embedding Evaluations/"
_SCRIPTS = {
    "summary":   "summarize_gtzan_data",
    "genre":     "evaluate_gtzan_retrieval",
    "retrieval": "evaluate_gtzan_exact_retrieval",
}

_EVAL_DIR = Path(__file__).parent / "Embedding Evaluations"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "subcommand",
        choices=list(_SCRIPTS),
        help="Which evaluation to run.",
    )
    # Capture all remaining args so we can pass them through.
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    module_name = _SCRIPTS[ns.subcommand]
    module_path = _EVAL_DIR / f"{module_name}.py"
    if not module_path.exists():
        sys.exit(f"Cannot find {module_path}")

    # Rewrite sys.argv so the underlying module's argparse sees the right flags.
    sys.argv = [str(module_path)] + ns.args

    # Add the eval dir to the front of sys.path so relative imports work.
    sys.path.insert(0, str(_EVAL_DIR))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
