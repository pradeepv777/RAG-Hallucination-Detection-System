"""
Unit tests for scoring and verdict resolution module (score.py).
"""

import pytest
from score import FaithfulnessScorer


def test_scorer_all_supported():
    scorer = FaithfulnessScorer()
    verified_claims = [
        {
            "claim_id": 0,
            "claim": "The sky is blue.",
            "verifications": [
                {
                    "chunk_id": 0,
                    "evidence": "On a clear day, the sky appears blue.",
                    "evidence_similarity": 0.88,
                    "entailment_prob": 0.94,
                    "contradiction_prob": 0.02,
                    "neutral_prob": 0.04,
                }
            ],
        },
        {
            "claim_id": 1,
            "claim": "Grass is green.",
            "verifications": [
                {
                    "chunk_id": 1,
                    "evidence": "Chlorophyll makes grass green.",
                    "evidence_similarity": 0.85,
                    "entailment_prob": 0.91,
                    "contradiction_prob": 0.03,
                    "neutral_prob": 0.06,
                }
            ],
        },
    ]

    report = scorer.score(verified_claims, raw_answer="The sky is blue and grass is green.", raw_context="...")

    assert report.total_claims == 2
    assert report.supported_claims == 2
    assert report.contradicted_claims == 0
    assert report.unsupported_claims == 0
    assert report.faithfulness_score == 1.0
    assert report.is_faithful is True
    assert report.is_hallucinated is False


def test_scorer_with_contradiction():
    scorer = FaithfulnessScorer()
    verified_claims = [
        {
            "claim_id": 0,
            "claim": "The event happened in 2022.",
            "verifications": [
                {
                    "chunk_id": 0,
                    "evidence": "The event officially concluded in 1945.",
                    "evidence_similarity": 0.82,
                    "entailment_prob": 0.05,
                    "contradiction_prob": 0.90,
                    "neutral_prob": 0.05,
                }
            ],
        }
    ]

    report = scorer.score(verified_claims, raw_answer="The event happened in 2022.", raw_context="...")

    assert report.total_claims == 1
    assert report.contradicted_claims == 1
    assert report.claims[0].verdict == "contradicted"
    assert report.is_hallucinated is True
    assert report.is_faithful is False


def test_scorer_with_unsupported_neutral():
    scorer = FaithfulnessScorer()
    verified_claims = [
        {
            "claim_id": 0,
            "claim": "The company has 500 employees.",
            "verifications": [
                {
                    "chunk_id": 0,
                    "evidence": "The company produces high-end bicycles.",
                    "evidence_similarity": 0.40,
                    "entailment_prob": 0.10,
                    "contradiction_prob": 0.05,
                    "neutral_prob": 0.85,
                }
            ],
        }
    ]

    report = scorer.score(verified_claims, raw_answer="The company has 500 employees.", raw_context="...")

    assert report.total_claims == 1
    assert report.unsupported_claims == 1
    assert report.claims[0].verdict == "unsupported"
    assert report.is_hallucinated is True


def test_scorer_empty_edge_cases():
    scorer = FaithfulnessScorer()

    # Empty answer
    report_empty_ans = scorer.score([], raw_answer="", raw_context="Some context")
    assert report_empty_ans.total_claims == 0
    assert report_empty_ans.is_faithful is True

    # Empty context
    claim_no_ctx = [{
        "claim_id": 0,
        "claim": "A random statement.",
        "verifications": []
    }]
    report_no_ctx = scorer.score(claim_no_ctx, raw_answer="A random statement.", raw_context="")
    assert report_no_ctx.total_claims == 1
    assert report_no_ctx.unsupported_claims == 1
    assert report_no_ctx.is_hallucinated is True
