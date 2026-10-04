"""
NLI Verification Module.

Uses a Cross-Encoder Natural Language Inference model (e.g., cross-encoder/nli-deberta-v3-base)
to evaluate factual consistency between retrieved context evidence (premise) and
extracted atomic claims (hypothesis).
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
from sentence_transformers import CrossEncoder


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Computes stable softmax probabilities over an array."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class NLIVerifier:
    """
    NLI Verifier that evaluates evidence-claim pairs using a DeBERTa cross-encoder.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name or os.getenv(
            "NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-base"
        )
        self.batch_size = batch_size
        self.device = device
        self._model: Optional[CrossEncoder] = None
        self._label_mapping: Dict[str, int] = {}

    @property
    def model(self) -> CrossEncoder:
        """Lazy loads the CrossEncoder model."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self.device)
            self._init_label_mapping()
        return self._model

    def _init_label_mapping(self) -> None:
        """
        Dynamically detects id2label mapping from the underlying HF config
        to ensure exact correspondence between class indices and NLI classes.
        """
        hf_config = getattr(self.model.model, "config", None)
        id2label = getattr(hf_config, "id2label", None) if hf_config else None

        # Standard default for cross-encoder/nli-deberta-v3-base:
        # {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        if id2label and isinstance(id2label, dict):
            for idx, label_name in id2label.items():
                lbl = str(label_name).lower()
                if "entail" in lbl:
                    self._label_mapping["entailment"] = int(idx)
                elif "contra" in lbl:
                    self._label_mapping["contradiction"] = int(idx)
                elif "neut" in lbl:
                    self._label_mapping["neutral"] = int(idx)
        else:
            self._label_mapping = {
                "contradiction": 0,
                "entailment": 1,
                "neutral": 2,
            }

        self._indices = (
            self._label_mapping.get("entailment", 1),
            self._label_mapping.get("contradiction", 0),
            self._label_mapping.get("neutral", 2),
        )

    def verify_pairs(self, premise_hypothesis_pairs: List[tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Runs batch inference over (premise, hypothesis) pairs.

        Returns:
            List of dicts: [
                {'entailment_prob': float, 'contradiction_prob': float, 'neutral_prob': float}
            ]
        """
        if not premise_hypothesis_pairs:
            return []

        logits = self.model.predict(
            premise_hypothesis_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # If single pair, shape is (3,), reshape to (1, 3)
        if len(logits.shape) == 1:
            logits = np.expand_dims(logits, axis=0)

        probs = softmax(logits, axis=-1)
        ent_idx, con_idx, neu_idx = self._indices

        return [
            {
                "entailment_prob": round(float(p[ent_idx]), 4),
                "contradiction_prob": round(float(p[con_idx]), 4),
                "neutral_prob": round(float(p[neu_idx]), 4),
            }
            for p in probs
        ]

    def verify_claims(
        self,
        retrieved_claims: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Verifies all claims against their retrieved evidence chunks in an efficient batch.

        Args:
            retrieved_claims: Claims enriched by EvidenceRetriever.retrieve()

        Returns:
            List of claims enriched with 'verifications' list.
        """
        # Collect all (evidence, claim) pairs across all claims for batching
        pairs = []
        pair_metadata = []  # tracks (claim_index, chunk_index)

        for c_idx, c in enumerate(retrieved_claims):
            claim_text = c.get("claim", "")
            evidence_chunks = c.get("evidence_chunks", [])
            for ch_idx, chunk in enumerate(evidence_chunks):
                evidence_text = chunk.get("evidence", "")
                # Premise is the retrieved context; Hypothesis is the generated claim
                pairs.append((evidence_text, claim_text))
                pair_metadata.append((c_idx, ch_idx, chunk))

        # Run batched inference
        all_probs = self.verify_pairs(pairs)

        # Reconstruct verified results back into claims structure
        verified_claims = []
        for c in retrieved_claims:
            verified_claims.append({
                **c,
                "verifications": [],
            })

        for (c_idx, ch_idx, chunk), prob_dict in zip(pair_metadata, all_probs):
            verification_entry = {
                "chunk_id": chunk.get("chunk_id", ch_idx),
                "evidence": chunk.get("evidence", ""),
                "evidence_similarity": chunk.get("similarity", 0.0),
                "entailment_prob": prob_dict["entailment_prob"],
                "contradiction_prob": prob_dict["contradiction_prob"],
                "neutral_prob": prob_dict["neutral_prob"],
            }
            verified_claims[c_idx]["verifications"].append(verification_entry)

        return verified_claims
