"""
RAGTruth Dataset Loader and Preprocessing Module.

Loads and joins RAGTruth response and source data, preserving the official train/test
splits and deriving clean answer-level and claim-level ground truth labels.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union


@dataclass
class RAGTruthExample:
    """Represents a unified, preprocessed RAGTruth example."""
    id: str
    source_id: str
    model: str
    temperature: float
    split: str
    quality: str
    task_type: str
    prompt: str
    question: str
    context: str
    response: str
    labels: List[Dict[str, Any]] = field(default_factory=list)
    is_hallucinated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "model": self.model,
            "temperature": self.temperature,
            "split": self.split,
            "quality": self.quality,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "question": self.question,
            "context": self.context,
            "response": self.response,
            "labels": self.labels,
            "is_hallucinated": self.is_hallucinated,
        }


def format_source_info(source_info: Any, task_type: str) -> tuple[str, str]:
    """
    Extracts the (context, question) tuple from task-specific source_info.
    
    Args:
        source_info: The raw source_info field from source_info.jsonl
        task_type: One of 'QA', 'Summary', 'Data2txt'
        
    Returns:
        (context, question) strings
    """
    if task_type == "QA" and isinstance(source_info, dict):
        question = source_info.get("question", "")
        passages = source_info.get("passages", "")
        if isinstance(passages, list):
            passages = "\n\n".join(str(p) for p in passages)
        return str(passages), str(question)

    if task_type == "Summary":
        context = source_info if isinstance(source_info, str) else str(source_info)
        return context, ""

    if task_type == "Data2txt":
        if isinstance(source_info, dict):
            # Format structured business/entity info into coherent textual context
            parts = []
            for k, v in source_info.items():
                if v:
                    parts.append(f"{k}: {v}")
            return "\n".join(parts), ""
        return str(source_info), ""

    # Fallback
    if isinstance(source_info, dict):
        return json.dumps(source_info, indent=2), ""
    return str(source_info), ""


class RAGTruthLoader:
    """Loader for the RAGTruth dataset with strict split isolation and ground-truth derivation."""

    def __init__(self, data_dir: Union[str, Path] = "dataset"):
        self.data_dir = Path(data_dir)
        self.response_file = self.data_dir / "response.jsonl"
        self.source_file = self.data_dir / "source_info.jsonl"

        if not self.response_file.exists():
            raise FileNotFoundError(f"Response file not found at: {self.response_file}")
        if not self.source_file.exists():
            raise FileNotFoundError(f"Source info file not found at: {self.source_file}")

    def load_sources(self) -> Dict[str, Dict[str, Any]]:
        """Loads all source items indexed by source_id."""
        sources = {}
        with open(self.source_file, "r", encoding="utf-8") as f:
            for line in f:
                if stripped := line.strip():
                    item = json.loads(stripped)
                    sources[str(item["source_id"])] = item
        return sources

    def load_dataset(
        self,
        split: Optional[str] = "test",
        quality: Optional[str] = "good",
        task_type: Optional[str] = None,
        model: Optional[str] = None,
        limit: Optional[int] = None,
        random_seed: Optional[int] = 42,
    ) -> List[RAGTruthExample]:
        """
        Loads preprocessed examples filtered by split, quality, task, and model.

        Args:
            split: 'train', 'test', or None for all splits
            quality: 'good', or None for all qualities
            task_type: 'QA', 'Summary', 'Data2txt', or None for all
            model: Filter by specific LLM (e.g. 'gpt-4-0613') or None
            limit: Maximum number of examples to load (subsampled if specified)
            random_seed: Seed for reproducible subsampling if limit is provided

        Returns:
            List of RAGTruthExample objects
        """
        sources = self.load_sources()
        examples: List[RAGTruthExample] = []

        with open(self.response_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                # Filter by split and quality
                if split is not None and item.get("split") != split:
                    continue
                if quality is not None and item.get("quality") != quality:
                    continue
                if model is not None and item.get("model") != model:
                    continue

                source_id = str(item.get("source_id"))
                src_data = sources.get(source_id)
                if not src_data:
                    continue

                curr_task = src_data.get("task_type", "")
                if task_type is not None and curr_task != task_type:
                    continue

                context, question = format_source_info(src_data.get("source_info"), curr_task)
                prompt = src_data.get("prompt", "")
                if not question and prompt:
                    question = prompt

                labels = item.get("labels", [])
                is_hallucinated = len(labels) > 0

                example = RAGTruthExample(
                    id=str(item.get("id")),
                    source_id=source_id,
                    model=item.get("model", ""),
                    temperature=float(item.get("temperature", 0.0)),
                    split=item.get("split", ""),
                    quality=item.get("quality", ""),
                    task_type=curr_task,
                    prompt=prompt,
                    question=question,
                    context=context,
                    response=item.get("response", ""),
                    labels=labels,
                    is_hallucinated=is_hallucinated,
                )
                examples.append(example)

        if limit is not None and len(examples) > limit:
            rng = random.Random(random_seed)
            examples = rng.sample(examples, limit)

        return examples

    @staticmethod
    def claim_overlaps_ground_truth(
        claim_text: str,
        full_response: str,
        ground_truth_labels: List[Dict[str, Any]],
        min_overlap_chars: int = 5,
    ) -> bool:
        """
        Determines whether an extracted claim overlaps with any ground-truth hallucination span.

        Args:
            claim_text: Extracted claim string
            full_response: The full LLM response text
            ground_truth_labels: List of annotated span dicts containing 'start', 'end', 'text'
            min_overlap_chars: Minimum character overlap required to mark as hallucinated

        Returns:
            True if the claim overlaps a labeled hallucinated span, False otherwise.
        """
        if not ground_truth_labels:
            return False

        # Find position of claim in full_response
        claim_start = full_response.find(claim_text)
        if claim_start == -1:
            # Substring exact match failed, fall back to checking if any label text is in claim
            # or claim is in label text
            for label in ground_truth_labels:
                lbl_text = label.get("text", "").strip()
                if lbl_text and (lbl_text in claim_text or claim_text in lbl_text):
                    return True
            return False

        claim_end = claim_start + len(claim_text)

        for label in ground_truth_labels:
            lbl_start = label.get("start", -1)
            lbl_end = label.get("end", -1)
            if lbl_start == -1 or lbl_end == -1:
                continue

            # Calculate intersection of intervals [claim_start, claim_end] and [lbl_start, lbl_end]
            overlap_start = max(claim_start, lbl_start)
            overlap_end = min(claim_end, lbl_end)
            if overlap_end - overlap_start >= min_overlap_chars:
                return True

        return False
