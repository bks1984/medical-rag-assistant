"""Compare base LLM vs prompt engineering vs RAG with an LLM-as-a-judge.

    python scripts/evaluate.py
    python scripts/evaluate.py --questions my_questions.json   # {"Specialty": "question", ...}

Writes results/evaluation_<timestamp>.csv and prints the average scores per approach.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from medical_rag.config import get_settings  # noqa: E402
from medical_rag.evaluate import run_evaluation, summarise  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, help="JSON file mapping specialty -> question")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8")) if args.questions else None
    settings = get_settings()

    df = run_evaluation(questions, settings)

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.results_dir / f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out, index=False)

    print("\nAverage scores (1-5):")
    print(summarise(df).to_string())
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
