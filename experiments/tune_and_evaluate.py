"""
Tune and Evaluate Experiment Script.

Strict experimental protocol:
1. Load Dev set (train split, seed 42, 100 samples) and Test set (test split, seed 42, 100 samples).
   No sample or source overlap between Dev and Test.
2. Run extraction and retrieval (top_k=5) + batched NLI inference once on Dev.
3. Sweep grid of hyperparameters ONLY on Dev:
   - For NLI:
     * faithfulness_threshold in [0.5, 0.6, 0.7, 0.8, 0.9]
     * entailment_threshold in [0.30, 0.35, 0.40, 0.45, 0.50]
     * contradiction_threshold in [0.35, 0.40]
     * top_k in [3, 5]
     * answer rule variants:
       - 'contradicted_only'
       - 'contradicted_or_faithfulness'
       - 'contradicted_or_unsupported_ge_2'
       - 'combined'
       - 'strict_unsupported' (old rule)
   - For Embedding Baseline:
     * similarity_threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
     * top_k in [3, 5]
     * rule variants:
       - 'any_below' (old rule)
       - 'unsupported_ge_2'
       - 'ratio_below'
       - 'ratio_or_ge_2'
4. Select the single best configuration for NLI and Baseline based on Dev F1 (tie-breaker: precision).
5. Touch the Test set exactly once at the end with the selected configurations.
6. Record full confusion matrices and per-task metrics.
7. Save everything to tuned_benchmark_results.json.
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from claims import ClaimExtractor
from dataset_loader import RAGTruthExample, RAGTruthLoader
from retrieve import EvidenceRetriever
from verify import NLIVerifier


def extract_and_cache_dataset(
    examples: List[RAGTruthExample],
    extractor: ClaimExtractor,
    retriever: EvidenceRetriever,
    verifier: NLIVerifier,
    max_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Extracts claims, retrieves max_k chunks, and runs NLI inference over all pairs in batch.
    Returns fully cached feature representations for fast parameter sweeping.
    """
    cached_data = []

    # 1. Extract claims & retrieve chunks for all examples
    all_pairs = []
    pair_index_map = []  # (ex_idx, claim_idx, chunk_idx)

    for ex_idx, ex in enumerate(examples):
        claims = extractor.extract_claims(ex.response)
        retrieved_claims = retriever.retrieve(ex.context, claims, top_k=max_k)
        
        ex_entry = {
            "id": ex.id,
            "task_type": ex.task_type,
            "model": ex.model,
            "y_true": ex.is_hallucinated,
            "labels": ex.labels,
            "response": ex.response,
            "claims": [],
        }

        for c_idx, rc in enumerate(retrieved_claims):
            claim_text = rc.get("claim", "")
            chunks = rc.get("evidence_chunks", [])
            
            c_entry = {
                "claim_id": rc.get("claim_id", c_idx),
                "claim": claim_text,
                "chunks": [],
            }

            for ch_idx, chunk in enumerate(chunks):
                evidence_text = chunk.get("evidence", "")
                similarity = chunk.get("similarity", 0.0)
                all_pairs.append((evidence_text, claim_text))
                pair_index_map.append((ex_idx, c_idx, ch_idx, similarity, evidence_text))
                
                c_entry["chunks"].append({
                    "chunk_id": ch_idx,
                    "evidence": evidence_text,
                    "similarity": similarity,
                    "entailment_prob": 0.0,
                    "contradiction_prob": 0.0,
                    "neutral_prob": 0.0,
                })

            ex_entry["claims"].append(c_entry)
        cached_data.append(ex_entry)

    # 2. Batch NLI inference
    print(f"Running batched NLI inference across {len(all_pairs)} (evidence, claim) pairs...")
    t0 = time.time()
    probs = verifier.verify_pairs(all_pairs)
    t1 = time.time()
    print(f"NLI inference finished in {t1 - t0:.2f}s ({(t1 - t0)/max(1, len(all_pairs))*1000:.1f}ms/pair).")

    # 3. Populate cached probabilities
    for (ex_idx, c_idx, ch_idx, sim, evid), p in zip(pair_index_map, probs):
        cached_data[ex_idx]["claims"][c_idx]["chunks"][ch_idx]["entailment_prob"] = p["entailment_prob"]
        cached_data[ex_idx]["claims"][c_idx]["chunks"][ch_idx]["contradiction_prob"] = p["contradiction_prob"]
        cached_data[ex_idx]["claims"][c_idx]["chunks"][ch_idx]["neutral_prob"] = p["neutral_prob"]

    return cached_data


def evaluate_nli_config(
    cached_data: List[Dict[str, Any]],
    top_k: int,
    entailment_threshold: float,
    contradiction_threshold: float,
    faithfulness_threshold: float,
    answer_rule: str,
) -> Dict[str, Any]:
    """Evaluates a single NLI configuration using cached probabilities."""
    y_true_ans = []
    y_pred_ans = []
    task_types = []
    
    y_true_claims = []
    y_pred_claims = []

    for ex in cached_data:
        ans_gt = ex["y_true"]
        claims = ex["claims"]
        
        if not claims:
            # No verifiable claims -> default faithful
            y_true_ans.append(ans_gt)
            y_pred_ans.append(False)
            task_types.append(ex["task_type"])
            continue

        supported_count = 0
        contradicted_count = 0
        unsupported_count = 0

        for c in claims:
            # Take top_k chunks
            active_chunks = c["chunks"][:top_k]
            
            # Resolve claim verdict
            if not active_chunks:
                c_verdict = "unsupported"
            else:
                supporting = [ch for ch in active_chunks if ch["entailment_prob"] >= entailment_threshold]
                contradicting = [ch for ch in active_chunks if ch["contradiction_prob"] >= contradiction_threshold]

                if supporting:
                    best_sup = max(supporting, key=lambda x: x["entailment_prob"])
                    if contradicting:
                        best_con = max(contradicting, key=lambda x: x["contradiction_prob"])
                        if best_con["similarity"] > best_sup["similarity"] and best_con["contradiction_prob"] > best_sup["entailment_prob"]:
                            c_verdict = "contradicted"
                        else:
                            c_verdict = "supported"
                    else:
                        c_verdict = "supported"
                elif contradicting:
                    c_verdict = "contradicted"
                else:
                    c_verdict = "unsupported"

            if c_verdict == "supported":
                supported_count += 1
            elif c_verdict == "contradicted":
                contradicted_count += 1
            else:
                unsupported_count += 1

            # Claim-level ground truth
            gt_claim = RAGTruthLoader.claim_overlaps_ground_truth(
                c["claim"], ex["response"], ex["labels"], min_overlap_chars=5
            )
            y_true_claims.append(gt_claim)
            y_pred_claims.append(c_verdict in ["contradicted", "unsupported"])

        total_claims = len(claims)
        faith_score = supported_count / total_claims if total_claims > 0 else 1.0

        # Answer-level rule variants
        if answer_rule == "contradicted_only":
            is_hallu = (contradicted_count > 0)
        elif answer_rule == "contradicted_or_faithfulness":
            is_hallu = (contradicted_count > 0) or (faith_score < faithfulness_threshold)
        elif answer_rule == "contradicted_or_unsupported_ge_2":
            is_hallu = (contradicted_count > 0) or (unsupported_count >= 2)
        elif answer_rule == "combined":
            is_hallu = (contradicted_count > 0) or (unsupported_count >= 2) or (faith_score < faithfulness_threshold)
        elif answer_rule == "strict_unsupported":
            is_hallu = (contradicted_count > 0) or (unsupported_count > 0) or (faith_score < faithfulness_threshold)
        else:
            raise ValueError(f"Unknown rule: {answer_rule}")

        y_true_ans.append(ans_gt)
        y_pred_ans.append(is_hallu)
        task_types.append(ex["task_type"])

    # Metrics
    metrics = compute_metrics_dict(y_true_ans, y_pred_ans, task_types, y_true_claims, y_pred_claims)
    return metrics


def evaluate_baseline_config(
    cached_data: List[Dict[str, Any]],
    top_k: int,
    similarity_threshold: float,
    baseline_rule: str,
    faithfulness_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Evaluates a single Embedding Similarity Baseline configuration."""
    y_true_ans = []
    y_pred_ans = []
    task_types = []
    
    y_true_claims = []
    y_pred_claims = []

    for ex in cached_data:
        ans_gt = ex["y_true"]
        claims = ex["claims"]
        
        if not claims:
            y_true_ans.append(ans_gt)
            y_pred_ans.append(False)
            task_types.append(ex["task_type"])
            continue

        below_count = 0
        above_count = 0

        for c in claims:
            active_chunks = c["chunks"][:top_k]
            top_sim = active_chunks[0]["similarity"] if active_chunks else 0.0
            is_claim_hallu = top_sim < similarity_threshold

            if is_claim_hallu:
                below_count += 1
            else:
                above_count += 1

            gt_claim = RAGTruthLoader.claim_overlaps_ground_truth(
                c["claim"], ex["response"], ex["labels"], min_overlap_chars=5
            )
            y_true_claims.append(gt_claim)
            y_pred_claims.append(is_claim_hallu)

        total_claims = len(claims)
        ratio_above = above_count / total_claims if total_claims > 0 else 1.0

        if baseline_rule == "any_below":
            is_hallu = below_count > 0
        elif baseline_rule == "unsupported_ge_2":
            is_hallu = below_count >= 2
        elif baseline_rule == "ratio_below":
            is_hallu = ratio_above < faithfulness_threshold
        elif baseline_rule == "ratio_or_ge_2":
            is_hallu = (below_count >= 2) or (ratio_above < faithfulness_threshold)
        else:
            raise ValueError(f"Unknown baseline rule: {baseline_rule}")

        y_true_ans.append(ans_gt)
        y_pred_ans.append(is_hallu)
        task_types.append(ex["task_type"])

    metrics = compute_metrics_dict(y_true_ans, y_pred_ans, task_types, y_true_claims, y_pred_claims)
    return metrics


def compute_metrics_dict(
    y_true_ans: List[bool],
    y_pred_ans: List[bool],
    task_types: List[str],
    y_true_claims: List[bool],
    y_pred_claims: List[bool],
) -> Dict[str, Any]:
    """Calculates precision, recall, F1, accuracy, and confusion matrix."""
    y_t = np.array(y_true_ans, dtype=bool)
    y_p = np.array(y_pred_ans, dtype=bool)

    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    acc = float(accuracy_score(y_t, y_p))
    cm = confusion_matrix(y_t, y_p, labels=[False, True])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    by_task = {}
    df = pd.DataFrame({"y_true": y_t, "y_pred": y_p, "task": task_types})
    for t in ["QA", "Summary", "Data2txt"]:
        sub = df[df["task"] == t]
        if len(sub) > 0:
            st_t = np.array(sub["y_true"], dtype=bool)
            st_p = np.array(sub["y_pred"], dtype=bool)
            sub_cm = confusion_matrix(st_t, st_p, labels=[False, True])
            s_tn, s_fp, s_fn, s_tp = [int(v) for v in sub_cm.ravel()]
            by_task[t] = {
                "total_samples": len(st_t),
                "precision": round(float(precision_score(st_t, st_p, zero_division=0)), 4),
                "recall": round(float(recall_score(st_t, st_p, zero_division=0)), 4),
                "f1": round(float(f1_score(st_t, st_p, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(st_t, st_p)), 4),
                "tp": s_tp, "fp": s_fp, "fn": s_fn, "tn": s_tn,
            }

    claim_metrics = {}
    if y_true_claims:
        yc_t = np.array(y_true_claims, dtype=bool)
        yc_p = np.array(y_pred_claims, dtype=bool)
        c_cm = confusion_matrix(yc_t, yc_p, labels=[False, True])
        c_tn, c_fp, c_fn, c_tp = [int(v) for v in c_cm.ravel()]
        claim_metrics = {
            "total_samples": len(yc_t),
            "precision": round(float(precision_score(yc_t, yc_p, zero_division=0)), 4),
            "recall": round(float(recall_score(yc_t, yc_p, zero_division=0)), 4),
            "f1": round(float(f1_score(yc_t, yc_p, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(yc_t, yc_p)), 4),
            "tp": c_tp, "fp": c_fp, "fn": c_fn, "tn": c_tn,
        }

    return {
        "overall": {
            "total_samples": len(y_t),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "by_task": by_task,
        "claim_metrics": claim_metrics,
    }


def main():
    torch.set_num_threads(10)
    print("="*60)
    print("STARTING STRICT TUNING & EVALUATION PIPELINE")
    print("="*60)

    # 1. Dataset Loading
    data_dir = str(PROJECT_ROOT / "dataset") if (PROJECT_ROOT / "dataset").exists() else "dataset"
    loader = RAGTruthLoader(data_dir)
    print("Loading DEV set (from 'train' split, quality='good', limit=100, seed=42)...")
    dev_examples = loader.load_dataset(split="train", quality="good", limit=100, random_seed=42)
    print("Loading TEST set (from 'test' split, quality='good', limit=100, seed=42)...")
    test_examples = loader.load_dataset(split="test", quality="good", limit=100, random_seed=42)

    # Verify zero overlap
    dev_ids = set(x.id for x in dev_examples)
    test_ids = set(x.id for x in test_examples)
    dev_srcs = set(x.source_id for x in dev_examples)
    test_srcs = set(x.source_id for x in test_examples)
    assert len(dev_ids & test_ids) == 0, "FATAL: ID overlap between Dev and Test!"
    assert len(dev_srcs & test_srcs) == 0, "FATAL: Source overlap between Dev and Test!"
    print(f"Verified: 0 ID overlap, 0 Source overlap. Dev={len(dev_examples)}, Test={len(test_examples)}")

    # Initialize modules
    extractor = ClaimExtractor()
    retriever = EvidenceRetriever()
    verifier = NLIVerifier(batch_size=64)

    # 2. Extract & cache DEV set
    print("\n--- EXTRACTING & CACHING DEV SET ---")
    dev_cached = extract_and_cache_dataset(dev_examples, extractor, retriever, verifier, max_k=5)

    # 3. Parameter Grid Sweep on DEV for NLI System
    print("\n--- SWEEPING NLI HYPERPARAMETERS ON DEV ---")
    nli_grid = []
    top_k_list = [3, 5]
    entail_list = [0.30, 0.35, 0.40, 0.45, 0.50]
    contra_list = [0.35, 0.40]
    faith_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    rule_list = [
        "contradicted_only",
        "contradicted_or_faithfulness",
        "contradicted_or_unsupported_ge_2",
        "combined",
        "strict_unsupported",
    ]

    for k in top_k_list:
        for ent in entail_list:
            for con in contra_list:
                for rule in rule_list:
                    if rule in ["contradicted_only", "contradicted_or_unsupported_ge_2"]:
                        # faithfulness threshold does not affect these rules
                        res = evaluate_nli_config(dev_cached, k, ent, con, 0.8, rule)
                        nli_grid.append({
                            "config": {
                                "top_k": k,
                                "entailment_threshold": ent,
                                "contradiction_threshold": con,
                                "faithfulness_threshold": 0.8,
                                "answer_rule": rule,
                            },
                            "overall": res["overall"],
                        })
                    else:
                        for faith in faith_list:
                            res = evaluate_nli_config(dev_cached, k, ent, con, faith, rule)
                            nli_grid.append({
                                "config": {
                                    "top_k": k,
                                    "entailment_threshold": ent,
                                    "contradiction_threshold": con,
                                    "faithfulness_threshold": faith,
                                    "answer_rule": rule,
                                },
                                "overall": res["overall"],
                            })

    # Sort NLI configs by F1 descending, then precision descending
    nli_grid.sort(key=lambda x: (x["overall"]["f1"], x["overall"]["precision"], x["overall"]["accuracy"]), reverse=True)
    best_nli_dev = nli_grid[0]
    print(f"Total NLI configs evaluated on Dev: {len(nli_grid)}")
    print(f"Top NLI Config on Dev: {best_nli_dev['config']}")
    print(f"Dev Performance: Prec={best_nli_dev['overall']['precision']}, Rec={best_nli_dev['overall']['recall']}, "
          f"F1={best_nli_dev['overall']['f1']}, Acc={best_nli_dev['overall']['accuracy']} "
          f"(TP={best_nli_dev['overall']['tp']}, FP={best_nli_dev['overall']['fp']}, TN={best_nli_dev['overall']['tn']}, FN={best_nli_dev['overall']['fn']})")

    # 4. Parameter Grid Sweep on DEV for Embedding Baseline
    print("\n--- SWEEPING BASELINE HYPERPARAMETERS ON DEV ---")
    baseline_grid = []
    sim_list = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    base_rules = ["any_below", "unsupported_ge_2", "ratio_below", "ratio_or_ge_2"]

    for k in top_k_list:
        for sim in sim_list:
            for rule in base_rules:
                if rule in ["any_below", "unsupported_ge_2"]:
                    res = evaluate_baseline_config(dev_cached, k, sim, rule, 0.8)
                    baseline_grid.append({
                        "config": {
                            "top_k": k,
                            "similarity_threshold": sim,
                            "baseline_rule": rule,
                            "faithfulness_threshold": 0.8,
                        },
                        "overall": res["overall"],
                    })
                else:
                    for faith in faith_list:
                        res = evaluate_baseline_config(dev_cached, k, sim, rule, faith)
                        baseline_grid.append({
                            "config": {
                                "top_k": k,
                                "similarity_threshold": sim,
                                "baseline_rule": rule,
                                "faithfulness_threshold": faith,
                            },
                            "overall": res["overall"],
                        })

    baseline_grid.sort(key=lambda x: (x["overall"]["f1"], x["overall"]["precision"], x["overall"]["accuracy"]), reverse=True)
    best_base_dev = baseline_grid[0]
    print(f"Total Baseline configs evaluated on Dev: {len(baseline_grid)}")
    print(f"Top Baseline Config on Dev: {best_base_dev['config']}")
    print(f"Dev Performance: Prec={best_base_dev['overall']['precision']}, Rec={best_base_dev['overall']['recall']}, "
          f"F1={best_base_dev['overall']['f1']}, Acc={best_base_dev['overall']['accuracy']} "
          f"(TP={best_base_dev['overall']['tp']}, FP={best_base_dev['overall']['fp']}, TN={best_base_dev['overall']['tn']}, FN={best_base_dev['overall']['fn']})")

    # 5. TOUCH TEST SET EXACTLY ONCE
    print("\n=======================================================")
    print("TOUCHING TEST SET EXACTLY ONCE WITH CHOSEN CONFIGURATIONS")
    print("=======================================================")
    test_cached = extract_and_cache_dataset(test_examples, extractor, retriever, verifier, max_k=5)

    # Evaluate Tuned NLI on Test
    tuned_nli_test_res = evaluate_nli_config(
        test_cached,
        top_k=best_nli_dev["config"]["top_k"],
        entailment_threshold=best_nli_dev["config"]["entailment_threshold"],
        contradiction_threshold=best_nli_dev["config"]["contradiction_threshold"],
        faithfulness_threshold=best_nli_dev["config"]["faithfulness_threshold"],
        answer_rule=best_nli_dev["config"]["answer_rule"],
    )

    # Evaluate Tuned Baseline on Test
    tuned_base_test_res = evaluate_baseline_config(
        test_cached,
        top_k=best_base_dev["config"]["top_k"],
        similarity_threshold=best_base_dev["config"]["similarity_threshold"],
        baseline_rule=best_base_dev["config"]["baseline_rule"],
        faithfulness_threshold=best_base_dev["config"]["faithfulness_threshold"],
    )

    # Also evaluate Untuned NLI (original defaults: top_k=3, ent=0.5, con=0.4, faith=0.99, strict_unsupported)
    orig_nli_test_res = evaluate_nli_config(
        test_cached,
        top_k=3,
        entailment_threshold=0.50,
        contradiction_threshold=0.40,
        faithfulness_threshold=0.99,
        answer_rule="strict_unsupported",
    )

    # Also evaluate Untuned Baseline (original defaults: top_k=3, sim=0.65, any_below)
    orig_base_test_res = evaluate_baseline_config(
        test_cached,
        top_k=3,
        similarity_threshold=0.65,
        baseline_rule="any_below",
    )

    # Package output
    final_output = {
        "metadata": {
            "dev_samples": len(dev_examples),
            "dev_split": "train (seed=42)",
            "test_samples": len(test_examples),
            "test_split": "test (seed=42)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chosen_nli_config": best_nli_dev["config"],
            "chosen_baseline_config": best_base_dev["config"],
            "dev_best_nli": best_nli_dev["overall"],
            "dev_best_baseline": best_base_dev["overall"],
        },
        "tuned_nli_system": tuned_nli_test_res,
        "tuned_baseline": tuned_base_test_res,
        "original_untuned_nli_system": orig_nli_test_res,
        "original_untuned_baseline": orig_base_test_res,
    }

    out_path = EXPERIMENTS_DIR / "tuned_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved all tuned experiment results to {out_path}.")

    # Print Comparison Table
    print("\n================== HELD-OUT TEST SET EVALUATION ==================")
    header = f"{'Method':<28} | {'Task':<10} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Acc':<7} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4}"
    print(header)
    print("-" * len(header))

    def print_row(name, data, task="overall"):
        d = data["overall"] if task == "overall" else data["by_task"][task]
        print(f"{name:<28} | {task:<10} | {d['precision']:<7.4f} | {d['recall']:<7.4f} | {d['f1']:<7.4f} | {d['accuracy']:<7.4f} | {d['tp']:<4} | {d['fp']:<4} | {d['tn']:<4} | {d['fn']:<4}")

    print_row("Original Baseline (Untuned)", orig_base_test_res, "overall")
    print_row("Original NLI (Untuned)", orig_nli_test_res, "overall")
    print_row("Tuned Baseline", tuned_base_test_res, "overall")
    print_row("Tuned NLI System", tuned_nli_test_res, "overall")

    print("\n--- By Task: Tuned NLI vs Tuned Baseline ---")
    for t in ["QA", "Summary", "Data2txt"]:
        print_row("Tuned Baseline", tuned_base_test_res, t)
        print_row("Tuned NLI System", tuned_nli_test_res, t)

    print("==================================================================")


if __name__ == "__main__":
    main()
