"""
Context Chunking and FAISS Vector Retrieval Module.

Indexes source context chunks using sentence-transformers and FAISS,
and retrieves the top-k most semantically relevant evidence chunks for each claim.
"""

import re
from typing import Any, Dict, List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from claims import split_into_sentences, clean_text

PASSAGE_SPLIT_PATTERN = re.compile(r"(?:passage\s+\d+:\s*)", flags=re.IGNORECASE)


class ContextChunker:
    """
    Chunks textual context into coherent, overlapping passages suitable
    for serving as NLI premises without breaking across mid-sentence boundaries.
    """

    def __init__(self, target_chunk_chars: int = 400, overlap_sentences: int = 1):
        self.target_chunk_chars = target_chunk_chars
        self.overlap_sentences = overlap_sentences

    def chunk(self, context: str) -> List[str]:
        """
        Splits context into chunks of approximate target_chunk_chars with sentence overlap.
        """
        if not context or not context.strip():
            return []

        # Check if already partitioned into explicit passage markers (common in RAG benchmarks)
        if "passage " in context.lower():
            parts = PASSAGE_SPLIT_PATTERN.split(context)
            passages = [clean_text(p) for p in parts if clean_text(p)]
            if len(passages) > 1:
                return passages

        # Check paragraph breaks
        paragraphs = [clean_text(p) for p in context.split("\n\n") if clean_text(p)]
        if len(paragraphs) > 1 and all(len(p) <= self.target_chunk_chars * 1.5 for p in paragraphs):
            return paragraphs

        # Sentence-based sliding window chunking
        sentences = split_into_sentences(context)
        if not sentences:
            return [clean_text(context)] if clean_text(context) else []

        if len(sentences) <= 2:
            return [clean_text(context)]

        chunks = []
        current_sentences: List[str] = []
        current_len = 0

        for sent in sentences:
            current_sentences.append(sent)
            current_len += len(sent) + 1

            if current_len >= self.target_chunk_chars:
                chunks.append(" ".join(current_sentences))
                # Maintain overlap
                current_sentences = current_sentences[-self.overlap_sentences:]
                current_len = sum(len(s) + 1 for s in current_sentences)

        if current_sentences:
            remaining = " ".join(current_sentences)
            if not chunks or remaining != chunks[-1]:
                chunks.append(remaining)

        return chunks


class EvidenceRetriever:
    """
    FAISS-based dense retriever using SentenceTransformers to index context
    and retrieve top-k supporting evidence chunks for claims.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunker: Optional[ContextChunker] = None,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunker = chunker or ContextChunker()

    def build_index(self, chunks: List[str]) -> tuple[Optional[faiss.IndexFlatIP], Optional[np.ndarray]]:
        """
        Embeds chunks and creates a normalized FAISS cosine similarity index.
        """
        if not chunks:
            return None, None

        embeddings = self.model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        # Normalize for Inner Product (cosine similarity)
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index, embeddings

    def retrieve_for_claim(
        self,
        claim: str,
        chunks: List[str],
        index: Optional[faiss.IndexFlatIP],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k evidence chunks for a given claim.

        Returns:
            List of dicts: [
                {'chunk_id': int, 'evidence': str, 'similarity': float}
            ]
        """
        if not chunks or index is None or not claim or not claim.strip():
            return []

        k = min(top_k, len(chunks))
        if k <= 0:
            return []

        claim_embedding = self.model.encode([claim], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(claim_embedding)

        similarities, indices = index.search(claim_embedding, k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and idx < len(chunks):
                results.append({
                    "chunk_id": int(idx),
                    "evidence": chunks[idx],
                    "similarity": round(float(sim), 4),
                })

        return results

    def retrieve(
        self,
        context: str,
        claims: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates chunking, indexing, and retrieval across all claims.

        Args:
            context: Raw context string
            claims: Output of ClaimExtractor ([{'claim_id': int, 'claim': str}])
            top_k: Number of evidence chunks per claim

        Returns:
            Enriched list of claims, each with an 'evidence' key containing top-k chunks.
        """
        chunks = self.chunker.chunk(context)
        empty_defaults = [
            {
                **c,
                "evidence_chunks": [],
                "top_evidence": "",
                "top_similarity": 0.0,
            }
            for c in claims
        ]
        if not chunks or not claims:
            return empty_defaults

        index, _ = self.build_index(chunks)
        if index is None:
            return empty_defaults

        claim_texts = [c.get("claim", "") for c in claims]
        valid_indices = [i for i, t in enumerate(claim_texts) if t and t.strip()]
        if not valid_indices:
            return empty_defaults

        k = min(top_k, len(chunks))
        valid_texts = [claim_texts[i] for i in valid_indices]

        # Vectorized batch encoding and single matrix search in FAISS
        claim_embeddings = self.model.encode(
            valid_texts, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(claim_embeddings)
        all_sims, all_idxs = index.search(claim_embeddings, k)

        enriched = list(empty_defaults)
        for batch_pos, orig_idx in enumerate(valid_indices):
            ev_chunks = [
                {
                    "chunk_id": int(ch_idx),
                    "evidence": chunks[ch_idx],
                    "similarity": round(float(sim), 4),
                }
                for sim, ch_idx in zip(all_sims[batch_pos], all_idxs[batch_pos])
                if 0 <= ch_idx < len(chunks)
            ]
            enriched[orig_idx] = {
                **claims[orig_idx],
                "evidence_chunks": ev_chunks,
                "top_evidence": ev_chunks[0]["evidence"] if ev_chunks else "",
                "top_similarity": ev_chunks[0]["similarity"] if ev_chunks else 0.0,
            }

        return enriched
