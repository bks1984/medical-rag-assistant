"""Ask the assistant from the command line.

    python scripts/ask.py "What are the symptoms of appendicitis?"
    python scripts/ask.py --mode compare "What is the protocol for managing sepsis?"
    python scripts/ask.py            # interactive, type 'quit' to exit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medical_rag.llm import ask_llm, ask_with_prompt_engineering  # noqa: E402
from medical_rag.rag import DISCLAIMER, answer_question  # noqa: E402

RULE = "-" * 78


def show_rag(question: str, show_context: bool) -> None:
    result = answer_question(question)
    print(f"\n{result.answer}\n")
    if result.excerpts:
        print(f"Retrieved pages: {', '.join(map(str, result.pages))}")
    if show_context:
        print(RULE)
        print(result.context)
    print(RULE)


def show_compare(question: str) -> None:
    for title, fn in (("1. Base LLM", ask_llm), ("2. Prompt engineering", ask_with_prompt_engineering)):
        print(f"\n=== {title} ===\n{fn(question)}")
    print("\n=== 3. RAG (Merck Manual) ===")
    show_rag(question, show_context=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--mode", choices=["rag", "compare"], default="rag")
    parser.add_argument("--show-context", action="store_true", help="Print the retrieved excerpts")
    args = parser.parse_args()

    print(DISCLAIMER)
    run = (lambda q: show_compare(q)) if args.mode == "compare" else (lambda q: show_rag(q, args.show_context))

    if args.question:
        run(" ".join(args.question))
        return

    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            run(q)


if __name__ == "__main__":
    main()
