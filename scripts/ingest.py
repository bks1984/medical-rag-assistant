"""Build the vector store from the medical manual PDF.

    python scripts/ingest.py
    python scripts/ingest.py --pdf path/to/manual.pdf --chunk-size 800 --chunk-overlap 150
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medical_rag.config import get_settings  # noqa: E402
from medical_rag.ingest import build_vectorstore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, help="Path to the manual PDF (default: MANUAL_PDF or data/medical_diagnosis_manual.pdf)")
    parser.add_argument("--chunk-size", type=int, help="Chunk size in tokens (default 500)")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap in tokens (default 50)")
    args = parser.parse_args()

    overrides = {}
    if args.pdf:
        overrides["pdf_path"] = args.pdf
    if args.chunk_size:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap

    settings = dataclasses.replace(get_settings(), **overrides)
    build_vectorstore(settings)


if __name__ == "__main__":
    main()
