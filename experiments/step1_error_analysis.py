"""
Step 1: Error Analysis on Dev Set (Train split, seed 42, 100 samples).
Identifies False Positive answers (y_true == False, y_pred == True),
analyzes each ungrounded claim, categorizes root causes, and computes category counts.
"""

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from claims import ClaimExtractor
from dataset_loader import RAGTruthLoader
from retrieve import EvidenceRetriever
from score import FaithfulnessScorer
from verify import NLIVerifier


def categorize_claim_error(claim_text: str, evidence_text: str, full_context: str, probs: Dict[str, float], sim: float) -> str:
    claim_lower = claim_text.lower()
    
    # 1. Filler claim / conversational framing
    filler_indicators = [
        "overall", "in conclusion", "in summary", "to summarize", "furthermore",
        "moreover", "in addition", "as mentioned", "as stated", "it is important to note",
        "note that", "the following is", "below is", "here is"
    ]
    if any(claim_lower.startswith(fi) for fi in filler_indicators) or len(claim_text.split()) <= 4:
        return "filler claim"

    # 2. Pronoun / context loss
    pronoun_pattern = r"^(it|he|she|they|this|these|that|its|their|his|her)\b"
    if re.search(pronoun_pattern, claim_lower.strip()):
        return "pronoun/context loss"

    # 3. Numeric / date mismatch
    has_number = bool(re.search(r"\b\d+([.,]\d+)?\b", claim_text))
    if has_number:
        # Check if numbers in claim are in context
        claim_nums = re.findall(r"\b\d+([.,]\d+)?\b", claim_text)
        context_nums = re.findall(r"\b\d+([.,]\d+)?\b", full_context)
        # If number is present in full context, but NLI scored neutral -> numerical/date reasoning limitation
        if any(cn in full_context for cn in claim_nums):
            return "numeric/date mismatch"

    # 4. Retrieval miss: fact exists in full context but not in retrieved evidence chunk
    # Check if key words from claim appear in full context outside evidence
    claim_words = set(w for w in re.findall(r"\b\w{4,}\b", claim_lower) if w not in ["with", "from", "that", "this", "were", "have", "been"])
    words_in_evidence = sum(1 for w in claim_words if w in evidence_text.lower())
    words_in_full = sum(1 for w in claim_words if w in full_context.lower())
    
    if words_in_full > words_in_evidence + 1 or sim < 0.50:
        return "retrieval miss"

    # 5. Paraphrase scored neutral: high word overlap with evidence or context, but NLI assigned neutral
    if probs["neutral_prob"] >= 0.50 and words_in_evidence >= 2:
        return "paraphrase scored neutral"

    return "other"


def main():
    torch.set_num_threads(10)
    print("Loading Dev set...")
    data_dir = str(PROJECT_ROOT / "dataset") if (PROJECT_ROOT / "dataset").exists() else "dataset"
    loader = RAGTruthLoader(data_dir)
    dev_examples = loader.load_dataset(split="train", quality="good", limit=100, random_seed=42)

    extractor = ClaimExtractor()
    retriever = EvidenceRetriever()
    verifier = NLIVerifier(batch_size=64)
    scorer = FaithfulnessScorer(entailment_threshold=0.50, contradiction_threshold=0.40, faithfulness_threshold=0.99)

    print("Running inference on Dev set...")
    fp_records = []
    category_counts = {
        "pronoun/context loss": 0,
        "retrieval miss": 0,
        "filler claim": 0,
        "numeric/date mismatch": 0,
        "paraphrase scored neutral": 0,
        "other": 0,
    }

    # Batch all pairs first
    ex_claims_list = []
    all_pairs = []
    pair_mapping = []

    for ex_idx, ex in enumerate(dev_examples):
        claims = extractor.extract_claims(ex.response)
        retrieved = retriever.retrieve(ex.context, claims, top_k=3)
        ex_claims_list.append((ex, retrieved))

        for c_idx, rc in enumerate(retrieved):
            c_text = rc.get("claim", "")
            for ch_idx, chunk in enumerate(rc.get("evidence_chunks", [])):
                all_pairs.append((chunk.get("evidence", ""), c_text))
                pair_mapping.append((ex_idx, c_idx, ch_idx, chunk))

    print(f"Total pairs to verify: {len(all_pairs)}...")
    probs = verifier.verify_pairs(all_pairs)

    # Reconstruct
    verified_data = []
    for ex, retrieved in ex_claims_list:
        v_claims = []
        for rc in retrieved:
            v_claims.append({**rc, "verifications": []})
        verified_data.append((ex, v_claims))

    for (ex_idx, c_idx, ch_idx, chunk), p in zip(pair_mapping, probs):
        verified_data[ex_idx][1][c_idx]["verifications"].append({
            "chunk_id": chunk.get("chunk_id", ch_idx),
            "evidence": chunk.get("evidence", ""),
            "evidence_similarity": chunk.get("similarity", 0.0),
            "entailment_prob": p["entailment_prob"],
            "contradiction_prob": p["contradiction_prob"],
            "neutral_prob": p["neutral_prob"],
        })

    # Evaluate each example
    for ex, v_claims in verified_data:
        report = scorer.score(v_claims, raw_answer=ex.response, raw_context=ex.context)
        y_true = ex.is_hallucinated
        y_pred = report.is_hallucinated

        # Check for False Positive (Clean in ground truth, but predicted hallucinated)
        if (not y_true) and y_pred:
            fp_claims = []
            for cr in report.claims:
                if cr.verdict in ["unsupported", "contradicted"]:
                    cat = categorize_claim_error(
                        cr.claim,
                        cr.evidence,
                        ex.context,
                        {
                            "entailment_prob": cr.entailment_prob,
                            "contradiction_prob": cr.contradiction_prob,
                            "neutral_prob": cr.neutral_prob,
                        },
                        cr.evidence_similarity,
                    )
                    category_counts[cat] += 1
                    fp_claims.append({
                        "claim_id": cr.claim_id,
                        "claim": cr.claim,
                        "verdict": cr.verdict,
                        "confidence_score": cr.score,
                        "evidence": cr.evidence,
                        "evidence_similarity": cr.evidence_similarity,
                        "entailment_prob": cr.entailment_prob,
                        "contradiction_prob": cr.contradiction_prob,
                        "neutral_prob": cr.neutral_prob,
                        "categorized_cause": cat,
                    })

            fp_records.append({
                "example_id": ex.id,
                "task_type": ex.task_type,
                "model": ex.model,
                "response": ex.response,
                "context_preview": ex.context[:250] + ("..." if len(ex.context) > 250 else ""),
                "total_claims": report.total_claims,
                "supported_claims": report.supported_claims,
                "faithfulness_score": report.faithfulness_score,
                "culprit_claims": fp_claims,
            })

    output_summary = {
        "dev_samples": len(dev_examples),
        "total_false_positive_answers": len(fp_records),
        "total_unsupported_claims_in_fp_answers": sum(len(r["culprit_claims"]) for r in fp_records),
        "category_counts": category_counts,
        "false_positive_details": fp_records,
    }

    out_path = EXPERIMENTS_DIR / "error_analysis_dev.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print("\n" + "="*50)
    print("STEP 1: ERROR ANALYSIS RESULTS (DEV SET)")
    print("="*50)
    print(f"Total Dev Answers Evaluated: {len(dev_examples)}")
    print(f"False Positive Answers: {len(fp_records)} out of {67} true negative answers")
    print(f"Total Culprit Claims in FP Answers: {sum(len(r['culprit_claims']) for r in fp_records)}")
    print("\nCategorized Causes Breakdown:")
    for cat, count in category_counts.items():
        pct = (count / max(1, sum(category_counts.values()))) * 100
        print(f"  - {cat:<28}: {count:>3} claims ({pct:>5.1f}%)")
    print("="*50)


if __name__ == "__main__":
    main()
