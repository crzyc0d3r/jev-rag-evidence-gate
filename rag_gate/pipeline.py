"""End-to-end pipeline and CLI.

retrieve (BM25 + dense, RRF) -> judge every candidate in one request ->
filter in code -> answerability gate -> generate (or refuse).

    python -m rag_gate.pipeline "How long can I restore a deleted object?"
    python -m rag_gate.pipeline --all              # run every query in data/queries.jsonl
    python -m rag_gate.pipeline --judge jev ...    # live Jev (needs TYPESAFE_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .generate import ChatGenerator, ExtractiveGenerator
from .judges import JevJudge, JudgeError, LexicalJudge
from .policy import NOT_IN_DOCUMENTS, GatePolicy, filter_passages, should_generate
from .questions import QUESTION_SET_VERSION
from .retrieval import HybridRetriever, Passage

DATA = Path(__file__).resolve().parent.parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_corpus(path: Path = DATA / "corpus.jsonl") -> list[Passage]:
    return [Passage(r["id"], r["title"], r["text"]) for r in load_jsonl(path)]


@dataclass
class RunRecord:
    query: str
    candidates: list[str]
    scores: dict
    kept: list[str]
    dropped_irrelevant: list[str]
    dropped_injection: list[str]
    p_answerable: float | None
    generated: bool
    answer: str
    judge_calls: int
    status: str = "ok"
    question_set: str = QUESTION_SET_VERSION
    policy: dict = field(default_factory=dict)


class RagPipeline:
    def __init__(self, passages, judge, generator, policy: GatePolicy | None = None, k: int = 10):
        self.retriever = HybridRetriever(passages)
        self.judge, self.generator = judge, generator
        self.policy = policy or GatePolicy()
        self.k = k

    def run(self, query: str) -> RunRecord:
        calls_before = self.judge.calls
        candidates = self.retriever.retrieve(query, k=self.k)
        base = dict(query=query, candidates=[p.id for p in candidates], policy=asdict(self.policy))
        try:
            scores = self.judge.score_candidates(query, candidates)
            filtered = filter_passages(candidates, scores, self.policy)
            p_ans = self.judge.answerable(query, filtered.kept) if filtered.kept else 0.0
        except JudgeError as exc:
            # Fail closed: no judgment means no generation.
            return RunRecord(**base, scores={}, kept=[], dropped_irrelevant=[], dropped_injection=[],
                             p_answerable=None, generated=False, answer=NOT_IN_DOCUMENTS,
                             judge_calls=self.judge.calls - calls_before, status=f"judge_error: {exc}")
        generate = bool(filtered.kept) and should_generate(p_ans, self.policy)
        answer = self.generator.generate(query, filtered.kept) if generate else NOT_IN_DOCUMENTS
        return RunRecord(**base, scores=scores, kept=[p.id for p in filtered.kept],
                         dropped_irrelevant=filtered.dropped_irrelevant,
                         dropped_injection=filtered.dropped_injection, p_answerable=p_ans,
                         generated=generate, answer=answer,
                         judge_calls=self.judge.calls - calls_before)


def make_judge(name: str):
    return JevJudge() if name == "jev" else LexicalJudge()


def make_generator(name: str):
    return ChatGenerator() if name == "chat" else ExtractiveGenerator()


def print_record(r: RunRecord) -> None:
    print(f"\nQ: {r.query}")
    print(f"  candidates ({len(r.candidates)}): {', '.join(r.candidates)}")
    if r.scores:
        top = sorted(r.scores.items(), key=lambda kv: -kv[1]["relevance"])[:5]
        print("  top relevance: " + ", ".join(f"{pid} {s['relevance']:.2f}" for pid, s in top))
    print(f"  kept: {r.kept or '-'} | dropped (injection): {r.dropped_injection or '-'}")
    pa = "n/a" if r.p_answerable is None else f"{r.p_answerable:.2f}"
    print(f"  P(answerable)={pa} -> {'generate' if r.generated else 'refuse'} | judge calls: {r.judge_calls}")
    print(f"  A: {r.answer}")
    if r.status != "ok":
        print(f"  status: {r.status}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="question to answer")
    ap.add_argument("--all", action="store_true", help="run every query in data/queries.jsonl")
    ap.add_argument("--judge", choices=["lexical", "jev"], default="lexical")
    ap.add_argument("--generator", choices=["extractive", "chat"], default="extractive")
    ap.add_argument("--k", type=int, default=10, help="candidates passed to the judge")
    ap.add_argument("--relevance", type=float, default=GatePolicy.relevance_threshold)
    ap.add_argument("--answerable", type=float, default=GatePolicy.answerable_threshold)
    ap.add_argument("--log", type=Path, help="append JSONL run records here (e.g. results/runs.jsonl)")
    args = ap.parse_args(argv)

    if not args.query and not args.all:
        ap.error("give a query or --all")
    queries = [q["query"] for q in load_jsonl(DATA / "queries.jsonl")] if args.all else [args.query]
    try:
        judge, generator = make_judge(args.judge), make_generator(args.generator)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    policy = GatePolicy(relevance_threshold=args.relevance, answerable_threshold=args.answerable)
    pipe = RagPipeline(load_corpus(), judge, generator, policy, k=args.k)

    failed = 0
    for q in queries:
        rec = pipe.run(q)
        print_record(rec)
        failed += rec.status != "ok"
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with open(args.log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec)) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
