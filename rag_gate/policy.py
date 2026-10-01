"""Stage 3: the decision lives in code.

The judge produces probabilities; this module turns them into actions using
explicit, versionable thresholds. Change the policy here without touching the
questions, and tune it on a labelled set with ``rag_gate.evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .retrieval import Passage

NOT_IN_DOCUMENTS = "Not in the documents: the retrieved passages do not contain enough evidence to answer this."


@dataclass(frozen=True)
class GatePolicy:
    relevance_threshold: float = 0.70   # keep a passage only if P(relevant) >= this
    injection_threshold: float = 0.50   # drop a passage if P(injection) >= this
    answerable_threshold: float = 0.60  # generate only if P(answerable) >= this
    max_context_passages: int = 5       # hard cap on what reaches the LLM


@dataclass
class FilterResult:
    kept: list[Passage] = field(default_factory=list)
    dropped_irrelevant: list[str] = field(default_factory=list)
    dropped_injection: list[str] = field(default_factory=list)


def filter_passages(passages: list[Passage], scores: dict[str, dict[str, float]],
                    policy: GatePolicy) -> FilterResult:
    """Apply the thresholds; kept passages are re-ordered by relevance (reranking)."""
    result = FilterResult()
    ranked = sorted(passages, key=lambda p: -scores[p.id]["relevance"])
    for p in ranked:
        s = scores[p.id]
        if s["injection"] >= policy.injection_threshold:
            # One filtering signal, not a security boundary: keep isolation and tool limits too.
            result.dropped_injection.append(p.id)
        elif s["relevance"] < policy.relevance_threshold:
            result.dropped_irrelevant.append(p.id)
        elif len(result.kept) < policy.max_context_passages:
            result.kept.append(p)
        else:
            result.dropped_irrelevant.append(p.id)
    return result


def should_generate(p_answerable: float, policy: GatePolicy) -> bool:
    return p_answerable >= policy.answerable_threshold
