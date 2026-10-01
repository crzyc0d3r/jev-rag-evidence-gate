"""Judges: who answers the typed questions.

``JevJudge`` sends the packed request to a Jev HTTP endpoint. ``LexicalJudge``
is an offline stand-in with the same interface so the pipeline, tests and the
threshold sweep run without network access or an API key. Its numbers are a
crude term-coverage heuristic and say nothing about how a real judge performs.

Both return validated probabilities; a malformed response raises ``JudgeError``
and is never turned into a passing score.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .questions import ANSWERABLE_QUESTION, build_state, candidate_questions
from .retrieval import Passage, tokenize

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
RETRYABLE = {429, 500, 502, 503, 504}


class JudgeError(RuntimeError):
    """The judge could not produce a valid answer. Callers must fail closed."""


class Judge(Protocol):
    calls: int

    def score_candidates(self, query: str, passages: list[Passage]) -> dict[str, dict[str, float]]:
        """Return {passage_id: {"relevance": p, "injection": p}}."""

    def answerable(self, query: str, passages: list[Passage]) -> float:
        """Probability that the passages are sufficient to answer the query."""


def _probability(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("noul must be a number")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("noul outside [0, 1]")
    return float(value)


def parse_noul_answers(response: dict, expected_keys: set[str]) -> dict[str, float]:
    """Validate a Jev response that should contain exactly `expected_keys` noul answers."""
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise ValueError("response has no answers object")
    answers = response["answers"]
    if set(answers) != expected_keys:
        raise ValueError("answered questions do not match the request")
    out = {}
    for key, answer in answers.items():
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise ValueError(f"{key}: expected a noul answer")
        out[key] = _probability(answer.get("noul"))
    return out


def _unpack(probs: dict[str, float], passages: list[Passage]) -> dict[str, dict[str, float]]:
    return {
        p.id: {"relevance": probs[f"rel_{p.id}"], "injection": probs.get(f"inj_{p.id}", 0.0)}
        for p in passages
    }


class JevJudge:
    """Standard-library client for a Jev endpoint (POST /v1/systemone).

    All candidate questions travel in ONE request: 20 candidates -> 1 call,
    instead of 20 query/passage calls to a cross-encoder or LLM.
    """

    def __init__(self, api_key=None, model=None, endpoint=None, timeout=30.0, retries=2,
                 check_injection=True):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self.api_key or self.api_key == "replace-me":
            raise ValueError("Set TYPESAFE_API_KEY to use the live Jev judge")
        self.model = model or os.environ.get("JEV_MODEL", "jev-latest")
        self.endpoint = endpoint or os.environ.get("JEV_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout, self.retries = timeout, retries
        self.check_injection = check_injection
        self.calls = 0

    def _post(self, payload: dict) -> dict:
        req = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def _ask(self, state: dict, questions: dict) -> dict[str, float]:
        payload = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(self.retries + 1):
            try:
                self.calls += 1
                return parse_noul_answers(self._post(payload), set(questions))
            except HTTPError as exc:  # never echo server bodies: they may contain the documents
                if exc.code not in RETRYABLE or attempt == self.retries:
                    raise JudgeError(f"Jev request failed (HTTP {exc.code})") from None
            except (URLError, TimeoutError):
                if attempt == self.retries:
                    raise JudgeError("Jev request failed: network error or timeout") from None
            except (ValueError, TypeError, KeyError):
                raise JudgeError("Jev returned an invalid response") from None
            time.sleep(min(2 ** attempt, 4))
        raise JudgeError("Jev retries exhausted")

    def score_candidates(self, query, passages):
        if not passages:
            return {}
        probs = self._ask(build_state(query, passages), candidate_questions(passages, self.check_injection))
        return _unpack(probs, passages)

    def answerable(self, query, passages):
        if not passages:
            return 0.0
        return self._ask(build_state(query, passages), ANSWERABLE_QUESTION)["answerable"]


INJECTION_PATTERNS = re.compile(
    r"ignore (all |any |your )?(previous|prior|above) instructions|"
    r"note to (any|the) (ai|assistant|llm)|you are now|system prompt|"
    r"disregard (the|all|your) (rules|instructions)",
    re.IGNORECASE,
)


class LexicalJudge:
    """Offline stand-in for Jev with the same interface.

    relevance  = logistic(coverage of the query's content terms by the passage)
    injection  = 0.95 if an injection pattern matches, else 0.02
    answerable = joint coverage of the query terms by all retained passages

    Deterministic and dependency-free; use it to exercise the pipeline, never
    as evidence of judge quality.
    """

    def __init__(self, steepness: float = 12.0, midpoint: float = 0.55):
        self.steepness, self.midpoint = steepness, midpoint
        self.calls = 0

    def _coverage(self, query: str, text: str) -> float:
        q = set(tokenize(query))
        return len(q & set(tokenize(text))) / len(q) if q else 0.0

    def _logistic(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-self.steepness * (x - self.midpoint)))

    def score_candidates(self, query, passages):
        self.calls += 1  # one packed "request" for the whole candidate set
        out = {}
        for p in passages:
            rel = self._logistic(self._coverage(query, p.content))
            inj = 0.95 if INJECTION_PATTERNS.search(p.text) else 0.02
            out[p.id] = {"relevance": round(rel, 4), "injection": inj}
        return out

    def answerable(self, query, passages):
        self.calls += 1
        if not passages:
            return 0.0
        joint = " ".join(p.content for p in passages)
        return round(self._logistic(self._coverage(query, joint)), 4)
