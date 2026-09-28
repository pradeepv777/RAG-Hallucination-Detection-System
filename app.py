"""
FastAPI Microservice for RAG Hallucination Detection.

Exposes REST endpoints to decompose answers into atomic claims, retrieve supporting
context evidence using vector search (FAISS), and evaluate claim-level faithfulness via NLI.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from claims import ClaimExtractor
from retrieve import EvidenceRetriever
from score import FaithfulnessReport, FaithfulnessScorer
from verify import NLIVerifier


# --- Pydantic Request & Response Schemas ---

class VerificationRequest(BaseModel):
    question: Optional[str] = Field(default="", description="The user prompt or query.")
    context: str = Field(..., description="Retrieved source context / reference text against which to verify.")
    answer: str = Field(..., description="LLM-generated answer to be inspected for hallucinations.")
    top_k_evidence: int = Field(default=3, ge=1, le=10, description="Number of evidence chunks to retrieve per claim.")


class ClaimDetail(BaseModel):
    claim_id: int
    claim: str
    verdict: str  # 'supported', 'contradicted', 'unsupported'
    score: float
    evidence: str
    evidence_similarity: float
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float


class VerificationResponse(BaseModel):
    faithfulness_score: float
    is_faithful: bool
    is_hallucinated: bool
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    claims: List[ClaimDetail]
    notes: List[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# --- Lifecycle & App Initialization ---

models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and resources on startup."""
    print("Initializing RAG Hallucination Detector Pipeline...")
    models["claim_extractor"] = ClaimExtractor()
    models["retriever"] = EvidenceRetriever()
    models["verifier"] = NLIVerifier()
    models["scorer"] = FaithfulnessScorer()
    print("Models initialized successfully.")
    yield
    models.clear()


app = FastAPI(
    title="RAG Hallucination Detection API",
    description="Fine-grained claim-level evidence verification and faithfulness scoring for RAG answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint to verify service readiness."""
    return HealthResponse(
        status="healthy",
        service="rag-hallucination-detector",
        version="1.0.0",
    )


@app.post("/verify", response_model=VerificationResponse, tags=["Verification"])
async def verify_answer(req: VerificationRequest):
    """
    Analyzes an LLM-generated answer against provided context.
    
    1. Extracts atomic claims from the answer.
    2. Retrieves top-k evidence chunks from context using FAISS.
    3. Verifies each claim via NLI Cross-Encoder.
    4. Computes claim verdicts and aggregate faithfulness score.
    """
    if not req.answer.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The answer field cannot be empty.",
        )

    try:
        extractor: ClaimExtractor = models.get("claim_extractor") or ClaimExtractor()
        retriever: EvidenceRetriever = models.get("retriever") or EvidenceRetriever()
        verifier: NLIVerifier = models.get("verifier") or NLIVerifier()
        scorer: FaithfulnessScorer = models.get("scorer") or FaithfulnessScorer()

        # Step 1: Extract atomic claims
        claims = extractor.extract_claims(req.answer)

        # Step 2: Retrieve evidence from context
        retrieved_claims = retriever.retrieve(req.context, claims, top_k=req.top_k_evidence)

        # Step 3: NLI verification
        verified_claims = verifier.verify_claims(retrieved_claims)

        # Step 4: Scoring & verdicts
        report: FaithfulnessReport = scorer.score(
            verified_claims, raw_answer=req.answer, raw_context=req.context
        )

        return VerificationResponse(
            faithfulness_score=round(report.faithfulness_score, 4),
            is_faithful=report.is_faithful,
            is_hallucinated=report.is_hallucinated,
            total_claims=report.total_claims,
            supported_claims=report.supported_claims,
            contradicted_claims=report.contradicted_claims,
            unsupported_claims=report.unsupported_claims,
            claims=[ClaimDetail(**c.to_dict()) for c in report.claims],
            notes=report.notes,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
