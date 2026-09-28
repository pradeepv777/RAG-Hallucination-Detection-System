"""
Unit tests for evidence chunking and retrieval (retrieve.py).
"""

import pytest
from retrieve import ContextChunker, EvidenceRetriever


def test_chunker_basic():
    chunker = ContextChunker(target_chunk_chars=100, overlap_sentences=1)
    context = (
        "Sentence one is here. Sentence two follows it. "
        "Sentence three is next. Sentence four is the last one."
    )
    chunks = chunker.chunk(context)
    assert len(chunks) >= 2
    # Ensure chunks are non-empty strings
    for ch in chunks:
        assert len(ch) > 0


def test_chunker_empty():
    chunker = ContextChunker()
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_chunker_qa_passages():
    chunker = ContextChunker()
    context = "passage 1: First passage context.\npassage 2: Second passage context."
    chunks = chunker.chunk(context)
    assert len(chunks) == 2
    assert "First passage" in chunks[0]
    assert "Second passage" in chunks[1]


def test_retriever_pipeline():
    retriever = EvidenceRetriever()
    context = (
        "The Eiffel Tower is located in Paris, France. "
        "It was constructed in 1889 as the entrance arch for the World's Fair. "
        "The Statue of Liberty was a gift from France to the United States."
    )
    claims = [
        {"claim_id": 0, "claim": "The Eiffel Tower is in Paris, France."},
        {"claim_id": 1, "claim": "The Statue of Liberty is in New York."},
    ]

    retrieved = retriever.retrieve(context, claims, top_k=2)

    assert len(retrieved) == 2
    for r in retrieved:
        assert "evidence_chunks" in r
        assert "top_evidence" in r
        assert "top_similarity" in r
        assert len(r["evidence_chunks"]) <= 2

    # Claim 0 should strongly match Eiffel Tower
    assert "Eiffel Tower" in retrieved[0]["top_evidence"]
    assert retrieved[0]["top_similarity"] > 0.6
