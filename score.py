"""
Faithfulness Scoring and Verdict Resolution Module.

Aggregates claim-level NLI verification outputs into claim verdicts and computes
answer-level faithfulness metrics, handling edge cases systematically.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClaimResult:
    """Represents the final verdict and evidence for a single claim."""
    claim_id: int
    claim: str
    verdict: str  # 'supported', 'contradicted', 'unsupported'
    score: float  # Confidence score (0.0 to 1.0)
    evidence: str
    evidence_similarity: float
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "verdict": self.verdict,
            "score": round(self.score, 4),
            "evidence": self.evidence,
            "evidence_similarity": round(self.evidence_similarity, 4),
            "entailment_prob": round(self.entailment_prob, 4),
            "contradiction_prob": round(self.contradiction_prob, 4),
            "neutral_prob": round(self.neutral_prob, 4),
        }


@dataclass
class FaithfulnessReport:
    """Comprehensive answer-level evaluation report."""
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    faithfulness_score: float  # [0.0, 1.0]
    is_faithful: bool
    is_hallucinated: bool
    claims: List[ClaimResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness_score": round(self.faithfulness_score, 4),
            "is_faithful": self.is_faithful,
            "is_hallucinated": self.is_hallucinated,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "contradicted_claims": self.contradicted_claims,
            "unsupported_claims": self.unsupported_claims,
            "notes": self.notes,
            "claims": [c.to_dict() for c in self.claims],
        }


class FaithfulnessScorer:
    """
    Computes claim-level verdicts and answer-level faithfulness metrics.
    """

    def __init__(
        self,
        entailment_threshold: float = 0.50,
        contradiction_threshold: float = 0.40,
        faithfulness_threshold: float = 0.99,
    ):
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.faithfulness_threshold = faithfulness_threshold

    def resolve_claim_verdict(self, verified_claim: Dict[str, Any]) -> ClaimResult:
        """
        Resolves the final verdict for a claim across its top-k verified evidence chunks.

        Decision Logic:
        1. If ANY chunk has contradiction_prob >= contradiction_threshold:
           -> 'contradicted' (direct factual conflict takes priority)
        2. Else if ANY chunk has entailment_prob >= entailment_threshold:
           -> 'supported' (sufficient grounding found)
        3. Otherwise:
           -> 'unsupported' (neutral or lack of evidence in context)
        """
        claim_id = verified_claim.get("claim_id", 0)
        claim_text = verified_claim.get("claim", "")
        chunk_verifications = verified_claim.get("verifications", [])

        if not chunk_verifications:
            # No evidence was retrieved (e.g. empty context)
            return ClaimResult(
                claim_id=claim_id,
                claim=claim_text,
                verdict="unsupported",
                score=0.0,
                evidence="",
                evidence_similarity=0.0,
                entailment_prob=0.0,
                contradiction_prob=0.0,
                neutral_prob=1.0,
            )

        # Evaluate supporting vs contradicting evidence:
        # A claim is SUPPORTED if any retrieved chunk directly entails it
        supporting_chunks = [
            cv for cv in chunk_verifications
            if cv.get("entailment_prob", 0.0) >= self.entailment_threshold
        ]

        contradicting_chunks = [
            cv for cv in chunk_verifications
            if cv.get("contradiction_prob", 0.0) >= self.contradiction_threshold
        ]

        # If a chunk directly entails the claim, verify if it's the strongest signal
        if supporting_chunks:
            best_support = max(supporting_chunks, key=lambda x: x.get("entailment_prob", 0.0))
            
            # Check if there is a genuine direct conflict that has higher semantic relevance
            if contradicting_chunks:
                best_contra = max(contradicting_chunks, key=lambda x: x.get("contradiction_prob", 0.0))
                # Only flag as contradiction if the contradictory chunk is more relevant or has higher contradiction
                if best_contra.get("evidence_similarity", 0.0) > best_support.get("evidence_similarity", 0.0) and best_contra.get("contradiction_prob", 0.0) > best_support.get("entailment_prob", 0.0):
                    return ClaimResult(
                        claim_id=claim_id,
                        claim=claim_text,
                        verdict="contradicted",
                        score=best_contra.get("contradiction_prob", 0.0),
                        evidence=best_contra.get("evidence", ""),
                        evidence_similarity=best_contra.get("evidence_similarity", 0.0),
                        entailment_prob=best_contra.get("entailment_prob", 0.0),
                        contradiction_prob=best_contra.get("contradiction_prob", 0.0),
                        neutral_prob=best_contra.get("neutral_prob", 0.0),
                    )

            # Grounded by supporting chunk
            return ClaimResult(
                claim_id=claim_id,
                claim=claim_text,
                verdict="supported",
                score=best_support.get("entailment_prob", 0.0),
                evidence=best_support.get("evidence", ""),
                evidence_similarity=best_support.get("evidence_similarity", 0.0),
                entailment_prob=best_support.get("entailment_prob", 0.0),
                contradiction_prob=best_support.get("contradiction_prob", 0.0),
                neutral_prob=best_support.get("neutral_prob", 0.0),
            )

        # If not supported, check if any chunk contradicts it
        if contradicting_chunks:
            best_contra = max(contradicting_chunks, key=lambda x: x.get("contradiction_prob", 0.0))
            return ClaimResult(
                claim_id=claim_id,
                claim=claim_text,
                verdict="contradicted",
                score=best_contra.get("contradiction_prob", 0.0),
                evidence=best_contra.get("evidence", ""),
                evidence_similarity=best_contra.get("evidence_similarity", 0.0),
                entailment_prob=best_contra.get("entailment_prob", 0.0),
                contradiction_prob=best_contra.get("contradiction_prob", 0.0),
                neutral_prob=best_contra.get("neutral_prob", 0.0),
            )

        # 3. Otherwise unsupported / neutral
        # Pick the chunk with highest semantic similarity to show what was found
        best = max(chunk_verifications, key=lambda x: x.get("evidence_similarity", 0.0))
        return ClaimResult(
            claim_id=claim_id,
            claim=claim_text,
            verdict="unsupported",
            score=best.get("neutral_prob", 0.0),
            evidence=best.get("evidence", ""),
            evidence_similarity=best.get("evidence_similarity", 0.0),
            entailment_prob=best.get("entailment_prob", 0.0),
            contradiction_prob=best.get("contradiction_prob", 0.0),
            neutral_prob=best.get("neutral_prob", 0.0),
        )

    def score(
        self,
        verified_claims: List[Dict[str, Any]],
        raw_answer: str = "",
        raw_context: str = "",
    ) -> FaithfulnessReport:
        """
        Aggregates claim verdicts into a full answer-level faithfulness report.
        """
        notes = []

        # Edge case: Empty answer
        if not raw_answer or not raw_answer.strip():
            notes.append("Empty response provided.")
            return FaithfulnessReport(
                total_claims=0,
                supported_claims=0,
                contradicted_claims=0,
                unsupported_claims=0,
                faithfulness_score=1.0,
                is_faithful=True,
                is_hallucinated=False,
                claims=[],
                notes=notes,
            )

        # Edge case: Empty context
        if not raw_context or not raw_context.strip():
            notes.append("Empty context provided; all claims treated as unsupported.")

        claim_results: List[ClaimResult] = [
            self.resolve_claim_verdict(vc) for vc in verified_claims
        ]

        total_claims = len(claim_results)

        # Edge case: No factual claims extracted (e.g. greeting or conversational acknowledgment)
        if total_claims == 0:
            notes.append("No verifiable factual claims detected in response.")
            return FaithfulnessReport(
                total_claims=0,
                supported_claims=0,
                contradicted_claims=0,
                unsupported_claims=0,
                faithfulness_score=1.0,
                is_faithful=True,
                is_hallucinated=False,
                claims=[],
                notes=notes,
            )

        supported_count = sum(1 for c in claim_results if c.verdict == "supported")
        contradicted_count = sum(1 for c in claim_results if c.verdict == "contradicted")
        unsupported_count = sum(1 for c in claim_results if c.verdict == "unsupported")

        faithfulness_score = supported_count / total_claims

        # An answer is hallucinated if any claim is contradicted or unsupported
        is_hallucinated = (contradicted_count > 0) or (unsupported_count > 0) or (faithfulness_score < self.faithfulness_threshold)
        is_faithful = not is_hallucinated

        return FaithfulnessReport(
            total_claims=total_claims,
            supported_claims=supported_count,
            contradicted_claims=contradicted_count,
            unsupported_claims=unsupported_count,
            faithfulness_score=faithfulness_score,
            is_faithful=is_faithful,
            is_hallucinated=is_hallucinated,
            claims=claim_results,
            notes=notes,
        )
