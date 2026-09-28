"""
Claim Extraction Module.

Decomposes an LLM-generated answer into verifiable atomic claims.
Provides a deterministic rule-based extractor as the primary reliable baseline,
with an optional interface for LLM-assisted atomic claim decomposition.
"""

import os
import re
from typing import Any, Dict, List, Optional


# Conversational prefixes and conversational fluff to skip or strip
DISCARD_PATTERNS = [
    r"^(sure!?|certainly!?|of course!?)\s*",
    r"^here (is|are) (a summary|an overview|the answer|the information)[^:.,]*[:.,]\s*",
    r"^based on the (provided|given) (text|context|passages?|data|documents?)[^:.,]*[:.,]\s*",
    r"^according to the (provided|given) (text|context|passages?|data|documents?)[^:.,]*[:.,]\s*",
    r"^(in summary|in conclusion|overall),?\s*",
    r"^please let me know if you have any (further )?questions\.?$",
    r"^hope (this|that) helps!?$",
]


def clean_text(text: str) -> str:
    """Removes extra whitespace and normalizes text."""
    return re.sub(r"\s+", " ", text).strip()


def is_conversational_filler(sentence: str) -> bool:
    """Checks if a sentence is merely conversational framing without factual content."""
    s = sentence.strip().lower()
    s_clean = s.rstrip("!.:")
    if len(s_clean) < 10 and not any(char.isdigit() for char in s_clean):
        return True
    filler_phrases = {
        "sure",
        "here is the answer",
        "here's the answer",
        "here is what i found",
        "let me know if you need more help",
        "hope this helps",
        "hope that helps",
    }
    if s_clean in filler_phrases:
        return True
    for fp in filler_phrases:
        if s.startswith(fp + ":") or s.startswith(fp + "."):
            return True
    return False


def split_into_sentences(text: str) -> List[str]:
    """
    Robust sentence boundary detection handling decimals, abbreviations,
    bullet points, and newlines without external NLTK/SpaCy dependencies.
    """
    if not text or not text.strip():
        return []

    # Clean text but keep paragraph structure
    text = text.strip()

    # Pre-process common non-boundary periods (e.g. e.g., i.e., Dr., U.S., etc.)
    # Protect decimals like 2.5
    text = re.sub(r"(\d+)\.(\d+)", r"\1<DECIMAL_POINT>\2", text)

    # Protect common abbreviations and initialisms (e.g., a.m., p.m., Dr., etc.)
    abbrs = [
        "dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "co", "vs", "etc",
        "u.s", "u.k", "e.g", "i.e", "a.m", "p.m"
    ]
    for abbr in abbrs:
        def _replace_abbr(match):
            return match.group(0).replace(".", "<ABBR_DOT>")
        pattern = re.compile(rf"\b{re.escape(abbr)}\.", re.IGNORECASE)
        text = pattern.sub(_replace_abbr, text)

    # Protect single capital initials like "John F. Kennedy"
    text = re.sub(r"\b([A-Z])\.\s+", r"\1<ABBR_DOT> ", text)

    # Split on sentence terminals (. ! ?) followed by whitespace or line breaks
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    sentences = []
    for raw in raw_sentences:
        # Restore protected characters
        s = raw.replace("<DECIMAL_POINT>", ".").replace("<ABBR_DOT>", ".")
        s = clean_text(s)

        # Strip bullet points or numbered lists at start
        s = re.sub(r"^(\d+[\.\)]|\*|\-|\•)\s*", "", s)

        # Strip conversational intro if present
        for pattern in DISCARD_PATTERNS:
            s = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()

        if s and len(s) > 8 and not is_conversational_filler(s):
            sentences.append(s)

    return sentences


def decompose_compound_sentence(sentence: str) -> List[str]:
    """
    Splits compound sentences joined by semicolons into discrete claims.
    Avoids splitting on simple commas to preserve semantic context.
    """
    if ";" in sentence:
        sub_clauses = [clean_text(part) for part in sentence.split(";")]
        claims = [c for c in sub_clauses if len(c) > 10 and not is_conversational_filler(c)]
        if len(claims) > 1:
            return claims
    return [sentence]


class ClaimExtractor:
    """
    Deterministic claim extractor that breaks generated answers into atomic,
    independently verifiable factual statements.
    """

    def __init__(self, min_claim_chars: int = 12):
        self.min_claim_chars = min_claim_chars

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts atomic claims from input text.

        Returns:
            List of dicts: [{'claim_id': int, 'claim': str}]
        """
        if not text or not text.strip():
            return []

        sentences = split_into_sentences(text)
        claims = []
        claim_id = 0

        for sent in sentences:
            decomposed = decompose_compound_sentence(sent)
            for item in decomposed:
                item = clean_text(item)
                if len(item) >= self.min_claim_chars and not is_conversational_filler(item):
                    claims.append({
                        "claim_id": claim_id,
                        "claim": item,
                    })
                    claim_id += 1

        return claims


class LLMClaimExtractor:
    """
    Optional LLM-assisted claim extractor for fine-grained atomic proposition extraction.
    Used when an OpenAI or Anthropic API key is provided and high-resolution
    propositional decomposition is desired.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.fallback = ClaimExtractor()

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts atomic claims using LLM if available; otherwise falls back gracefully.
        """
        if not self.api_key:
            return self.fallback.extract_claims(text)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            prompt = (
                "Decompose the following text into distinct, atomic factual claims. "
                "Each claim should be a standalone self-contained statement. "
                "Output one claim per line without bullets or numbering.\n\n"
                f"Text:\n{text}\n\nClaims:"
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            lines = [clean_text(line) for line in content.splitlines() if clean_text(line)]
            claims = []
            for i, line in enumerate(lines):
                # Remove any leading digits or dashes
                line = re.sub(r"^(\d+[\.\)]|\*|\-|\•)\s*", "", line)
                if len(line) >= 10:
                    claims.append({"claim_id": i, "claim": line})
            return claims if claims else self.fallback.extract_claims(text)
        except Exception:
            return self.fallback.extract_claims(text)


def extract_claims(text: str) -> List[Dict[str, Any]]:
    """Convenience function using the default deterministic extractor."""
    extractor = ClaimExtractor()
    return extractor.extract_claims(text)
