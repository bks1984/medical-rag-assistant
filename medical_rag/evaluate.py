"""LLM-as-a-judge evaluation of the three answering approaches.

For each question, every answer (base LLM, prompt-engineered, RAG) is rated 1-5 on
  * groundedness - is the answer supported by the retrieved manual excerpts?
  * relevance    - does it address the main aspects of the question?

All three approaches are judged against the SAME retrieved context, so the comparison is fair.

Fixes over the notebook version:
  * scores are parsed with a regex instead of `reply[-2]`, which returned NaN whenever the
    judge ended its reply with anything other than "N}".
  * question 4 of the prompt-engineering run was scored with question 3's answer; here each
    answer is paired with its own question.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .config import SAMPLE_QUESTIONS, Settings, get_settings
from .llm import ask_llm, ask_with_prompt_engineering, chat
from .prompts import GROUNDEDNESS_SYSTEM, JUDGE_USER_TEMPLATE, RELEVANCE_SYSTEM
from .rag import answer_question
from .retriever import format_context

APPROACHES = ("base_llm", "prompt_engineering", "rag")

_SCORE_RE = {
    key: re.compile(rf"{key}\W{{0,3}}\s*([1-5])", re.IGNORECASE)
    for key in ("groundedness_score", "relevance_score")
}
_ANY_SCORE_RE = re.compile(r"\b([1-5])\b")


def parse_score(reply: str, key: str) -> int | None:
    """Pull the 1-5 score out of a judge reply. Returns None if none can be found."""
    if not reply:
        return None
    matches = _SCORE_RE[key].findall(reply)
    if matches:
        return int(matches[-1])
    # Fallback: the last standalone 1-5 digit in the final line.
    last_line = reply.strip().splitlines()[-1]
    loose = _ANY_SCORE_RE.findall(last_line)
    return int(loose[-1]) if loose else None


@dataclass
class Judgement:
    groundedness: int | None
    relevance: int | None
    groundedness_reasoning: str = ""
    relevance_reasoning: str = ""


def judge(question: str, context: str, answer: str, settings: Settings | None = None) -> Judgement:
    settings = settings or get_settings()
    user_msg = JUDGE_USER_TEMPLATE.format(question=question, context=context, answer=answer)
    g = chat(user_msg, GROUNDEDNESS_SYSTEM, model=settings.judge_model, max_tokens=600, temperature=0, settings=settings)
    r = chat(user_msg, RELEVANCE_SYSTEM, model=settings.judge_model, max_tokens=600, temperature=0, settings=settings)
    return Judgement(
        groundedness=parse_score(g, "groundedness_score"),
        relevance=parse_score(r, "relevance_score"),
        groundedness_reasoning=g,
        relevance_reasoning=r,
    )


@dataclass
class EvalRow:
    specialty: str
    question: str
    approach: str
    answer: str
    groundedness: int | None
    relevance: int | None


def run_evaluation(questions: dict[str, str] | None = None, settings: Settings | None = None, progress=print):
    """Answer every question three ways, judge each answer, return a pandas DataFrame."""
    import pandas as pd

    settings = settings or get_settings()
    questions = questions or SAMPLE_QUESTIONS
    rows: list[EvalRow] = []

    for specialty, question in questions.items():
        progress(f"[{specialty}] generating answers ...")
        rag = answer_question(question, settings=settings)
        context = format_context(rag.excerpts)
        answers = {
            "base_llm": ask_llm(question, settings),
            "prompt_engineering": ask_with_prompt_engineering(question, settings),
            "rag": rag.answer,
        }
        for approach in APPROACHES:
            progress(f"[{specialty}] judging {approach} ...")
            j = judge(question, context, answers[approach], settings)
            rows.append(EvalRow(specialty, question, approach, answers[approach], j.groundedness, j.relevance))

    return pd.DataFrame([asdict(r) for r in rows])


def summarise(df):
    """Mean groundedness and relevance per approach, in the notebook's order."""
    return (
        df.groupby("approach")[["groundedness", "relevance"]]
        .mean(numeric_only=True)
        .reindex(list(APPROACHES))
        .round(2)
    )
