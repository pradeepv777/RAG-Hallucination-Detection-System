"""
Unit tests for atomic claim extraction module (claims.py).
"""

import pytest
from claims import ClaimExtractor, split_into_sentences, clean_text, is_conversational_filler


def test_clean_text():
    raw = "  This   is   a   test.\n\nWith newlines   "
    assert clean_text(raw) == "This is a test. With newlines"


def test_is_conversational_filler():
    assert is_conversational_filler("Sure!") is True
    assert is_conversational_filler("Here is the answer:") is True
    assert is_conversational_filler("Hope that helps!") is True
    assert is_conversational_filler("Panera Bread has 2.5 stars.") is False


def test_split_into_sentences_preserves_decimals():
    text = "The restaurant has a rating of 2.5 stars based on 3 reviews. It was founded in 1993."
    sents = split_into_sentences(text)
    assert len(sents) == 2
    assert "2.5 stars" in sents[0]


def test_split_into_sentences_handles_abbreviations():
    text = "Dr. Smith met with Mr. Johnson at 10 a.m. in the U.S. capital."
    sents = split_into_sentences(text)
    assert len(sents) == 1
    assert "Dr. Smith" in sents[0]


def test_split_into_sentences_strips_conversational_prefix():
    text = "Sure! Based on the provided text, the company was founded in 1980."
    sents = split_into_sentences(text)
    assert len(sents) == 1
    assert "Sure!" not in sents[0]
    assert "the company was founded in 1980" in sents[0]


def test_claim_extractor_output_format():
    extractor = ClaimExtractor()
    text = "Anne Frank died in 1945. She was 15 years old; her sister Margot also died."
    claims = extractor.extract_claims(text)

    assert len(claims) >= 2
    for i, c in enumerate(claims):
        assert "claim_id" in c
        assert "claim" in c
        assert c["claim_id"] == i
        assert len(c["claim"]) > 10


def test_claim_extractor_empty_input():
    extractor = ClaimExtractor()
    assert extractor.extract_claims("") == []
    assert extractor.extract_claims("   \n\n  ") == []
