"""Tune the policy on a labelled query set.

For each relevance threshold it reports passage precision / recall of what
reaches the LLM, plus how often the answerability gate makes the right call
(generate for answerable queries, refuse for unanswerable ones). Judge scores
are computed once and reused, so the sweep costs one judge pass.

    python -m rag_gate.evaluate
    python -m rag_gate.evaluate --judge jev --thresholds 0.5 0.6 0.7 0.8 0.9
"""

from __future__ import annotations

import argparse

from .pipeline import DATA, load_corpus, load_jsonl, make_judge
from .policy import GatePolicy, filter_passages, should_generate
from .retrieval import HybridRetriever


def retrieval_recall(retriever, labelled, k):
    hits = total = 0
    for row in labelled:
        got = {p.id for p in retriever.retrieve(row["query"], k=k)}
        hits += len(got & set(row["relevant"]))
        total += len(row["relevant"])
    return hits / total if total else 1.0


def sweep(judge, passages, labelled, thresholds, k=10, answerable_threshold=0.6):
    retriever = HybridRetriever(passages)
    cached = []
    for row in labelled:
        cands = retriever.retrieve(row["query"], k=k)
        cached.append((row, cands, judge.score_candidates(row["query"], cands)))

    rows = []
    for t in thresholds:
        policy = GatePolicy(relevance_threshold=t, answerable_threshold=answerable_threshold)
        tp = fp = fn = gate_ok = 0
        for row, cands, scores in cached:
            kept = filter_passages(cands, scores, policy).kept
            kept_ids, gold = {p.id for p in kept}, set(row["relevant"])
            tp += len(kept_ids & gold)
            fp += len(kept_ids - gold)
            fn += len(gold - kept_ids)
            p_ans = judge.answerable(row["query"], kept) if kept else 0.0
            gate_ok += should_generate(p_ans, policy) == row["answerable"]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        rows.append({"threshold": t, "precision": precision, "recall": recall,
                     "false_pos": fp, "false_neg": fn, "gate_accuracy": gate_ok / len(cached)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=["lexical", "jev"], default="lexical")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    args = ap.parse_args(argv)

    passages, labelled = load_corpus(), load_jsonl(DATA / "queries.jsonl")
    judge = make_judge(args.judge)
    print(f"retrieval recall@{args.k} (upper bound for everything downstream): "
          f"{retrieval_recall(HybridRetriever(passages), labelled, args.k):.2f}")
    print(f"{'threshold':>9} {'precision':>9} {'recall':>7} {'FP':>4} {'FN':>4} {'gate acc':>8}")
    for r in sweep(judge, passages, labelled, args.thresholds, k=args.k):
        print(f"{r['threshold']:>9.2f} {r['precision']:>9.2f} {r['recall']:>7.2f} "
              f"{r['false_pos']:>4} {r['false_neg']:>4} {r['gate_accuracy']:>8.2f}")
    print(f"judge calls: {judge.calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
