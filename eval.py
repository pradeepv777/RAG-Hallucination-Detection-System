"""
Evaluation and Benchmarking Harness against RAGTruth.

Evaluates hallucination detection across both answer-level and claim-level metrics
against the published RAGTruth test split ground truth.
Compares:
  - Baseline 1: Embedding Similarity Threshold
  - Baseline 2: Full NLI Verification Pipeline (DeBERTa cross-encoder)
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from claims import ClaimExtractor
from dataset_loader import RAGTruthExample, RAGTruthLoader
from retrieve import EvidenceRetriever
from score import FaithfulnessScorer
from verify import NLIVerifier


@dataclass
class EvaluationMetrics:
    method: str
    level: str  # 'answer' or 'claim'
    task: str   # 'overall', 'QA', 'Summary', 'Data2txt'
    total_samples: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "level": self.level,
            "task": self.task,
            "total_samples": self.total_samples,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


class BenchmarkRunner:
    """
    Runs systematic evaluation of hallucination detection baselines against RAGTruth.
    """

    def __init__(
        self,
        retriever: Optional[EvidenceRetriever] = None,
        verifier: Optional[NLIVerifier] = None,
        scorer: Optional[FaithfulnessScorer] = None,
        similarity_threshold: float = 0.65,
    ):
        self.claim_extractor = ClaimExtractor()
        self.retriever = retriever or EvidenceRetriever()
        self.verifier = verifier or NLIVerifier()
        self.scorer = scorer or FaithfulnessScorer()
        self.similarity_threshold = similarity_threshold

    def evaluate_dataset(
        self,
        examples: List[RAGTruthExample],
        methods: List[str] = ["embedding_similarity", "nli"],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Runs evaluation on a collection of examples across specified methods.
        """
        results_by_method: Dict[str, Any] = {}

        for method in methods:
            print(f"\n=======================================================")
            print(f"Running Evaluation for Method: {method.upper()}")
            print(f"Total Examples: {len(examples)}")
            print(f"=======================================================")

            start_time = time.time()
            records = []
            claim_records = []

            for i, ex in enumerate(examples):
                claims = self.claim_extractor.extract_claims(ex.response)
                retrieved_claims = self.retriever.retrieve(ex.context, claims, top_k=top_k)

                if method == "embedding_similarity":
                    # Baseline 1: Flag claim as unsupported if top similarity < threshold
                    claim_preds = []
                    for rc in retrieved_claims:
                        sim = rc.get("top_similarity", 0.0)
                        is_claim_hallu = sim < self.similarity_threshold
                        claim_preds.append({
                            "claim_id": rc.get("claim_id", 0),
                            "claim": rc.get("claim", ""),
                            "verdict": "unsupported" if is_claim_hallu else "supported",
                            "score": sim,
                            "is_pred_hallu": is_claim_hallu,
                        })

                    # Answer is hallucinated if any claim is unsupported
                    ans_pred_hallu = any(cp["is_pred_hallu"] for cp in claim_preds) if claim_preds else False

                elif method == "nli":
                    # Full NLI Pipeline
                    verified = self.verifier.verify_claims(retrieved_claims)
                    report = self.scorer.score(verified, raw_answer=ex.response, raw_context=ex.context)
                    ans_pred_hallu = report.is_hallucinated

                    claim_preds = [
                        {
                            "claim_id": cr.claim_id,
                            "claim": cr.claim,
                            "verdict": cr.verdict,
                            "score": cr.score,
                            "is_pred_hallu": cr.verdict in ["contradicted", "unsupported"],
                        }
                        for cr in report.claims
                    ]
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Answer-level ground truth
                ans_gt_hallu = ex.is_hallucinated

                records.append({
                    "id": ex.id,
                    "task_type": ex.task_type,
                    "model": ex.model,
                    "y_true": ans_gt_hallu,
                    "y_pred": ans_pred_hallu,
                })

                # Claim-level ground truth mapping: check if claim overlaps labeled spans
                for cp in claim_preds:
                    gt_claim_hallu = RAGTruthLoader.claim_overlaps_ground_truth(
                        cp["claim"], ex.response, ex.labels
                    )
                    claim_records.append({
                        "id": ex.id,
                        "task_type": ex.task_type,
                        "claim_id": cp["claim_id"],
                        "y_true": gt_claim_hallu,
                        "y_pred": cp["is_pred_hallu"],
                    })

                if (i + 1) % 25 == 0 or (i + 1) == len(examples):
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"[{method}] Processed {i+1}/{len(examples)} examples ({rate:.1f} ex/sec)")

            # Compute answer-level metrics
            df_ans = pd.DataFrame(records)
            ans_metrics_overall = self._calculate_metrics(df_ans["y_true"], df_ans["y_pred"], method, "answer", "overall")

            ans_metrics_by_task = {}
            for t in ["QA", "Summary", "Data2txt"]:
                subset = df_ans[df_ans["task_type"] == t]
                if len(subset) > 0:
                    ans_metrics_by_task[t] = self._calculate_metrics(
                        subset["y_true"], subset["y_pred"], method, "answer", t
                    )

            # Compute claim-level metrics
            df_claims = pd.DataFrame(claim_records)
            claim_metrics_overall = None
            if len(df_claims) > 0:
                claim_metrics_overall = self._calculate_metrics(
                    df_claims["y_true"], df_claims["y_pred"], method, "claim", "overall"
                )

            results_by_method[method] = {
                "elapsed_seconds": round(time.time() - start_time, 2),
                "answer_metrics": {
                    "overall": ans_metrics_overall.to_dict(),
                    "by_task": {k: v.to_dict() for k, v in ans_metrics_by_task.items()},
                },
                "claim_metrics": claim_metrics_overall.to_dict() if claim_metrics_overall else None,
            }

        return results_by_method

    @staticmethod
    def _calculate_metrics(
        y_true: pd.Series, y_pred: pd.Series, method: str, level: str, task: str
    ) -> EvaluationMetrics:
        """Calculates standard classification metrics."""
        y_t = np.array(y_true, dtype=bool)
        y_p = np.array(y_pred, dtype=bool)

        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        acc = float(accuracy_score(y_t, y_p))

        cm = confusion_matrix(y_t, y_p, labels=[False, True])
        tn, fp, fn, tp = cm.ravel()

        return EvaluationMetrics(
            method=method,
            level=level,
            task=task,
            total_samples=len(y_t),
            precision=prec,
            recall=rec,
            f1=f1,
            accuracy=acc,
            tp=int(tp),
            fp=int(fp),
            fn=int(fn),
            tn=int(tn),
        )


def format_results_table(benchmark_results: Dict[str, Any]) -> str:
    """Formats benchmark results as a Markdown table."""
    lines = [
        "| Method | Level | Task | Precision | Recall | F1 Score | Accuracy | Samples |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for method, data in benchmark_results.items():
        ans_overall = data["answer_metrics"]["overall"]
        lines.append(
            f"| {method} | Answer | Overall | {ans_overall['precision']:.3f} | "
            f"{ans_overall['recall']:.3f} | {ans_overall['f1']:.3f} | "
            f"{ans_overall['accuracy']:.3f} | {ans_overall['total_samples']} |"
        )
        for task, t_data in data["answer_metrics"]["by_task"].items():
            lines.append(
                f"| {method} | Answer | {task} | {t_data['precision']:.3f} | "
                f"{t_data['recall']:.3f} | {t_data['f1']:.3f} | "
                f"{t_data['accuracy']:.3f} | {t_data['total_samples']} |"
            )
        if data.get("claim_metrics"):
            cm = data["claim_metrics"]
            lines.append(
                f"| {method} | Claim | Overall | {cm['precision']:.3f} | "
                f"{cm['recall']:.3f} | {cm['f1']:.3f} | "
                f"{cm['accuracy']:.3f} | {cm['total_samples']} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hallucination Detector on RAGTruth")
    parser.add_argument("--data_dir", default="dataset", help="Path to RAGTruth dataset directory")
    parser.add_argument("--split", default="test", help="Dataset split ('test', 'train')")
    parser.add_argument("--task_type", default=None, help="Filter by task ('QA', 'Summary', 'Data2txt')")
    parser.add_argument("--limit", type=int, default=100, help="Number of examples to evaluate (default: 100)")
    parser.add_argument("--output_file", default="benchmark_results.json", help="Path to save output JSON")
    parser.add_argument("--methods", nargs="+", default=["embedding_similarity", "nli"], help="Methods to benchmark")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed for subsampling")
    args = parser.parse_args()

    print(f"Loading RAGTruth split '{args.split}' from {args.data_dir} (limit={args.limit})...")
    loader = RAGTruthLoader(args.data_dir)
    examples = loader.load_dataset(
        split=args.split,
        quality="good",
        task_type=args.task_type,
        limit=args.limit,
        random_seed=args.random_seed,
    )
    print(f"Loaded {len(examples)} examples.")

    runner = BenchmarkRunner()
    results = runner.evaluate_dataset(examples, methods=args.methods)

    # Save results JSON
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics to {args.output_file}")

    # Print summary table
    table_str = format_results_table(results)
    print("\n================== BENCHMARK SUMMARY TABLE ==================")
    print(table_str)
    print("=============================================================\n")


if __name__ == "__main__":
    main()
