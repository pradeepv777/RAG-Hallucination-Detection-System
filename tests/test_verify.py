"""
Unit tests for NLI verification module (verify.py).
"""

import numpy as np
import pytest
from verify import NLIVerifier, softmax


def test_softmax_properties():
    # 1D array
    x = np.array([2.0, 1.0, 0.1])
    probs = softmax(x)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]

    # 2D array
    x2 = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    probs2 = softmax(x2, axis=-1)
    assert np.allclose(np.sum(probs2, axis=-1), [1.0, 1.0])
    assert probs2[0, 0] > 0.99
    assert probs2[1, 1] > 0.99


def test_verifier_empty_pairs():
    verifier = NLIVerifier()
    res = verifier.verify_pairs([])
    assert res == []


def test_verifier_entailment_and_contradiction():
    verifier = NLIVerifier()

    premise = "The Eiffel Tower is in Paris, France."
    entailed_claim = "Paris has the Eiffel Tower."
    contradicted_claim = "The Eiffel Tower is located in Berlin, Germany."

    pairs = [
        (premise, entailed_claim),
        (premise, contradicted_claim),
    ]

    results = verifier.verify_pairs(pairs)
    assert len(results) == 2

    # Entailment pair
    assert results[0]["entailment_prob"] > results[0]["contradiction_prob"]
    assert results[0]["entailment_prob"] > 0.6

    # Contradiction pair
    assert results[1]["contradiction_prob"] > results[1]["entailment_prob"]
    assert results[1]["contradiction_prob"] > 0.6


def test_verifier_verify_claims_structure():
    verifier = NLIVerifier()
    retrieved_claims = [
        {
            "claim_id": 0,
            "claim": "Water boils at 100 degrees Celsius at sea level.",
            "evidence_chunks": [
                {
                    "chunk_id": 0,
                    "evidence": "At standard atmospheric pressure at sea level, the boiling point of water is 100 °C.",
                    "similarity": 0.88,
                }
            ],
            "top_evidence": "At standard atmospheric pressure at sea level, the boiling point of water is 100 °C.",
            "top_similarity": 0.88,
        }
    ]

    verified = verifier.verify_claims(retrieved_claims)
    assert len(verified) == 1
    assert "verifications" in verified[0]
    verifs = verified[0]["verifications"]
    assert len(verifs) == 1
    assert verifs[0]["entailment_prob"] > 0.7
    assert verifs[0]["evidence_similarity"] == 0.88
