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

    @staticmethod
    def _create_claim_result(
        claim_id: int,
        claim: str,
        verdict: str,
        score: float,
        chunk: Optional[Dict[str, Any]] = None,
    ) -> ClaimResult:
        ch = chunk or {}
        return ClaimResult(
            claim_id=claim_id,
            claim=claim,
            verdict=verdict,
            score=score,
            evidence=ch.get("evidence", ""),
            evidence_similarity=ch.get("evidence_similarity", 0.0),
            entailment_prob=ch.get("entailment_prob", 0.0),
            contradiction_prob=ch.get("contradiction_prob", 0.0),
            neutral_prob=ch.get("neutral_prob", 1.0 if not ch else 0.0),
        )

    @staticmethod
    def _empty_report(notes: List[str]) -> FaithfulnessReport:
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

    def resolve_claim_verdict(self, verified_claim: Dict[str, Any]) -> ClaimResult:
        """
        Resolves the final verdict for a claim across its top-k verified evidence chunks.
        """
        claim_id = verified_claim.get("claim_id", 0)
        claim_text = verified_claim.get("claim", "")
        chunk_verifications = verified_claim.get("verifications", [])

        if not chunk_verifications:
            return self._create_claim_result(claim_id, claim_text, "unsupported", 0.0)

        supporting_chunks = [
            cv for cv in chunk_verifications
            if cv.get("entailment_prob", 0.0) >= self.entailment_threshold
        ]
        contradicting_chunks = [
            cv for cv in chunk_verifications
            if cv.get("contradiction_prob", 0.0) >= self.contradiction_threshold
        ]

        if supporting_chunks:
            best_support = max(supporting_chunks, key=lambda x: x.get("entailment_prob", 0.0))
            if contradicting_chunks:
                best_contra = max(contradicting_chunks, key=lambda x: x.get("contradiction_prob", 0.0))
                if (
                    best_contra.get("evidence_similarity", 0.0) > best_support.get("evidence_similarity", 0.0)
                    and best_contra.get("contradiction_prob", 0.0) > best_support.get("entailment_prob", 0.0)
                ):
                    return self._create_claim_result(
                        claim_id, claim_text, "contradicted", best_contra.get("contradiction_prob", 0.0), best_contra
                    )

            return self._create_claim_result(
                claim_id, claim_text, "supported", best_support.get("entailment_prob", 0.0), best_support
            )

        if contradicting_chunks:
            best_contra = max(contradicting_chunks, key=lambda x: x.get("contradiction_prob", 0.0))
            return self._create_claim_result(
                claim_id, claim_text, "contradicted", best_contra.get("contradiction_prob", 0.0), best_contra
            )

        best = max(chunk_verifications, key=lambda x: x.get("evidence_similarity", 0.0))
        return self._create_claim_result(
            claim_id, claim_text, "unsupported", best.get("neutral_prob", 0.0), best
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

        if not raw_answer or not raw_answer.strip():
            notes.append("Empty response provided.")
            return self._empty_report(notes)

        if not raw_context or not raw_context.strip():
            notes.append("Empty context provided; all claims treated as unsupported.")

        claim_results: List[ClaimResult] = [
            self.resolve_claim_verdict(vc) for vc in verified_claims
        ]
        total_claims = len(claim_results)

        if total_claims == 0:
            notes.append("No verifiable factual claims detected in response.")
            return self._empty_report(notes)

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
