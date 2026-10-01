import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from rag_gate.generate import ExtractiveGenerator
from rag_gate.judges import JevJudge, JudgeError, LexicalJudge, parse_noul_answers
from rag_gate.pipeline import RagPipeline, load_corpus
from rag_gate.policy import NOT_IN_DOCUMENTS, GatePolicy, filter_passages
from rag_gate.questions import candidate_questions
from rag_gate.retrieval import BM25, HybridRetriever, Passage, reciprocal_rank_fusion

CORPUS = load_corpus()


# --- retrieval ---------------------------------------------------------------

def test_bm25_ranks_exact_match_first():
    assert BM25(CORPUS).search("rotate access keys without downtime", 3)[0][0] == "C7"


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[("a", 9), ("b", 8)], [("b", 1), ("c", 0.5)]], k=60)
    assert fused[0][0] == "b"
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_returns_k_unique_passages():
    got = HybridRetriever(CORPUS).retrieve("restore deleted object", k=8)
    assert len(got) == 8 and len({p.id for p in got}) == 8


# --- response validation -----------------------------------------------------

def _resp(answers):
    return {"model": "jev-test", "answers": {k: {"type": "noul", "noul": v} for k, v in answers.items()}}


def test_parse_accepts_valid():
    assert parse_noul_answers(_resp({"rel_C1": 0.9}), {"rel_C1"}) == {"rel_C1": 0.9}


@pytest.mark.parametrize("bad", [1.2, -0.1, True, "0.5", float("nan")])
def test_parse_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        parse_noul_answers(_resp({"rel_C1": bad}), {"rel_C1"})


def test_parse_rejects_missing_or_extra_keys():
    with pytest.raises(ValueError):
        parse_noul_answers(_resp({"rel_C1": 0.5}), {"rel_C1", "rel_C2"})


# --- live client with a fake transport ----------------------------------------

class FakeJev(JevJudge):
    def __init__(self, responder, **kw):
        super().__init__(api_key="test-key", **kw)
        self.responder, self.payloads = responder, []

    def _post(self, payload):
        self.payloads.append(payload)
        return self.responder(payload)


def echo_responder(value):
    return lambda payload: _resp({k: value for k in payload["questions"]})


def test_all_candidates_scored_in_one_request():
    judge = FakeJev(echo_responder(0.8))
    cands = CORPUS[:12]
    scores = judge.score_candidates("q", cands)
    assert judge.calls == 1 and len(judge.payloads) == 1
    assert set(scores) == {p.id for p in cands}
    assert len(judge.payloads[0]["questions"]) == 24  # relevance + injection per passage


def test_questions_are_self_contained():
    qs = candidate_questions(CORPUS[:2])
    for key, q in qs.items():
        assert key.split("_")[1] in q["instructions"]  # names its passage; keys are not read


def test_auth_error_not_retried_and_body_not_echoed():
    def boom(_):
        raise HTTPError("u", 401, "secret body", {}, BytesIO(b"doc text"))
    judge = FakeJev(boom, retries=3)
    with pytest.raises(JudgeError) as exc:
        judge.score_candidates("q", CORPUS[:2])
    assert judge.calls == 1 and "doc text" not in str(exc.value)


def test_malformed_response_raises():
    judge = FakeJev(lambda p: {"answers": {}})
    with pytest.raises(JudgeError):
        judge.score_candidates("q", CORPUS[:2])


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        JevJudge()


# --- policy --------------------------------------------------------------------

def test_filter_applies_thresholds_and_injection():
    ps = [Passage(i, i, "t") for i in ("A", "B", "C")]
    scores = {"A": {"relevance": 0.9, "injection": 0.0},
              "B": {"relevance": 0.95, "injection": 0.9},
              "C": {"relevance": 0.4, "injection": 0.0}}
    r = filter_passages(ps, scores, GatePolicy(relevance_threshold=0.7))
    assert [p.id for p in r.kept] == ["A"]
    assert r.dropped_injection == ["B"] and r.dropped_irrelevant == ["C"]


def test_filter_caps_context_and_reranks():
    ps = [Passage(str(i), "", "t") for i in range(6)]
    scores = {p.id: {"relevance": 0.5 + int(p.id) / 20, "injection": 0.0} for p in ps}
    r = filter_passages(ps, scores, GatePolicy(relevance_threshold=0.5, max_context_passages=3))
    assert [p.id for p in r.kept] == ["5", "4", "3"]


# --- end to end ------------------------------------------------------------------

def test_pipeline_answers_with_citation():
    rec = RagPipeline(CORPUS, LexicalJudge(), ExtractiveGenerator()).run(
        "How do I rotate access keys without downtime?")
    assert rec.generated and "[C7]" in rec.answer and rec.judge_calls == 2


def test_pipeline_refuses_unanswerable():
    rec = RagPipeline(CORPUS, LexicalJudge(), ExtractiveGenerator()).run(
        "Does Harbor support object locking for compliance (WORM)?")
    assert not rec.generated and rec.answer == NOT_IN_DOCUMENTS


def test_injection_passage_never_reaches_context():
    rec = RagPipeline(CORPUS, LexicalJudge(), ExtractiveGenerator()).run(
        "How long can I restore a deleted object?")
    assert "C14" not in rec.kept


def test_pipeline_fails_closed_on_judge_error():
    def boom(_):
        raise HTTPError("u", 500, "x", {}, BytesIO(b""))
    judge = FakeJev(boom, retries=0)
    rec = RagPipeline(CORPUS, judge, ExtractiveGenerator()).run("anything")
    assert rec.status.startswith("judge_error") and not rec.generated
    json.dumps(rec.__dict__)  # record stays serialisable for logging
