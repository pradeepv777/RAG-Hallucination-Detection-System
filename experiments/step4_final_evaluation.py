"""
Step 4: Final Evaluation on Fresh Held-Out Test Set (300 Samples).
Protocol:
- NEW final test set drawn from RAGTruth test split (seed 789).
- Strictly disjoint from ALL prior sets:
  * 0 ID & 0 source overlap with training pool (900 samples)
  * 0 ID & 0 source overlap with old 100-sample test
  * 0 ID & 0 source overlap with old 300-sample fresh test
- Touched EXACTLY ONCE.
- Baselines evaluated:
  1. flag-all
  2. flag-none
  3. original_strict (unimproved heuristic: faith < 0.99 or con > 0 or unsup > 0)
  4. tuned_embedding_baseline (per-task CV tuned: QA=0.3637, Summary=0.3444, Data2txt=0.3890)
  5. global_threshold_improved (global logistic regression, threshold t=0.70)
  6. new_per_task_system (per-task calibrated models, threshold R>=0.60: QA=0.5467, Summary=0.4452, Data2txt=0.5110)
  7. new_per_task_interaction_system (interaction model, threshold R>=0.60: QA=0.3646, Summary=0.3466, Data2txt=0.7527)
- 1,000-iteration bootstrap 95% CIs for all metrics per task and overall.
- Save to per_task_calibrated_results.json without touching existing JSON files.
"""

import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

def resolve_cache_path(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    cand1 = PROJECT_ROOT / ".cache" / os.path.basename(filename)
    if cand1.exists():
        return str(cand1)
    cand2 = Path(".cache") / os.path.basename(filename)
    if cand2.exists():
        return str(cand2)
    return str(cand1) if (PROJECT_ROOT / ".cache").is_dir() else filename

from claims import ClaimExtractor
from dataset_loader import RAGTruthExample, RAGTruthLoader
from retrieve import EvidenceRetriever
from verify import NLIVerifier
from step2_ablations import (
    cache_dataset_nli,
    extract_tabular_features,
)
from step3_train_pool_and_cv import (
    extract_base_12_features,
    extract_interaction_features,
    extract_baseline_feature,
)


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Computes precision, recall, f1, accuracy and 1000-iteration bootstrap 95% CIs."""
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    rng = np.random.RandomState(seed)
    f1s, precs, recs, accs = [], [], [], []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        f1s.append(f1_score(yt_b, yp_b, zero_division=0))
        precs.append(precision_score(yt_b, yp_b, zero_division=0))
        recs.append(recall_score(yt_b, yp_b, zero_division=0))
        accs.append(accuracy_score(yt_b, yp_b))

    def _ci(arr):
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_ci": _ci(precs),
        "recall_ci": _ci(recs),
        "f1_ci": _ci(f1s),
        "accuracy_ci": _ci(accs),
    }


def evaluate_system_full(y_true: np.ndarray, y_pred: np.ndarray, tasks: List[str]) -> Dict[str, Any]:
    """Generates overall and per-task evaluation metrics with bootstrap 95% CIs."""
    res = {
        "overall": {
            "total_samples": len(y_true),
            "positives": int(np.sum(y_true)),
            **bootstrap_metrics(y_true, y_pred),
        },
        "by_task": {},
    }

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "task": tasks})
    for t in ["QA", "Summary", "Data2txt"]:
        sub = df[df["task"] == t]
        yt_t = np.array(sub["y_true"], dtype=bool)
        yp_t = np.array(sub["y_pred"], dtype=bool)
        res["by_task"][t] = {
            "total_samples": len(yt_t),
            "positives": int(np.sum(yt_t)),
            **bootstrap_metrics(yt_t, yp_t),
        }
    return res


def main():
    torch.set_num_threads(10)
    print("=" * 80)
    print("STEP 4: FINAL EVALUATION ON FRESH HELD-OUT TEST SET (300 SAMPLES)")
    print("=" * 80)

    data_dir = str(PROJECT_ROOT / "dataset") if (PROJECT_ROOT / "dataset").exists() else "dataset"
    loader = RAGTruthLoader(data_dir)

    # 1. Load Training Pool to train final frozen models
    with open(resolve_cache_path("cache_train_pool_900.json"), "r", encoding="utf-8") as f:
        train_pool_900 = json.load(f)

    # 2. Strict Data Hygiene: Verify and construct NEW final test set
    with open(resolve_cache_path("cache_test_300.json"), "r", encoding="utf-8") as f:
        prev_fresh_test_300 = json.load(f)
    old_test_100 = loader.load_dataset(split="test", quality="good", limit=100, random_seed=42)

    used_test_ids = set(x["id"] for x in prev_fresh_test_300) | set(x.id for x in old_test_100)
    used_test_srcs = set(x["source_id"] for x in prev_fresh_test_300) | set(x.source_id for x in old_test_100)

    all_test = loader.load_dataset(split="test", quality="good", limit=None)
    avail_test = [x for x in all_test if x.id not in used_test_ids and x.source_id not in used_test_srcs]

    rng_test = random.Random(789)
    qa_avail = [x for x in avail_test if x.task_type == "QA"]
    sum_avail = [x for x in avail_test if x.task_type == "Summary"]
    d2t_avail = [x for x in avail_test if x.task_type == "Data2txt"]

    new_test_qa = rng_test.sample(qa_avail, 100)
    new_test_sum = rng_test.sample(sum_avail, 100)
    new_test_d2t = rng_test.sample(d2t_avail, 100)
    new_test_examples = new_test_qa + new_test_sum + new_test_d2t

    # Verify zero overlap across all sets
    train_pool_ids = set(x["id"] for x in train_pool_900)
    train_pool_srcs = set(x["source_id"] for x in train_pool_900)
    new_test_ids = set(x.id for x in new_test_examples)
    new_test_srcs = set(x.source_id for x in new_test_examples)

    assert len(train_pool_ids & new_test_ids) == 0, "Train-test ID overlap detected!"
    assert len(train_pool_srcs & new_test_srcs) == 0, "Train-test Source overlap detected!"
    assert len(used_test_ids & new_test_ids) == 0, "Prev-test ID overlap detected!"
    assert len(used_test_srcs & new_test_srcs) == 0, "Prev-test Source overlap detected!"

    print("Strict Data Hygiene Check PASSED: Zero overlap across all sets.")
    print(f"Sampled NEW Final Test (300 samples): QA={len(new_test_qa)}, Summary={len(new_test_sum)}, Data2txt={len(new_test_d2t)}")

    # 3. Cache New Final Test NLI Data
    cache_path = resolve_cache_path("cache_new_final_test_300.json")
    extractor = ClaimExtractor()
    retriever = EvidenceRetriever()
    verifier = NLIVerifier(batch_size=64)

    cache_final_test = cache_dataset_nli(
        new_test_examples,
        extractor,
        retriever,
        verifier,
        cache_path=cache_path,
        use_contextualization=True,
        use_improved_premise=True,
        use_data2txt_nat=True,
    )

    y_test = np.array([x["y_true"] for x in cache_final_test], dtype=bool)
    test_tasks = [x["task_type"] for x in cache_final_test]

    print(f"\nFinal Test Ground Truth: Total={len(y_test)}, Positives={int(np.sum(y_test))} ({np.mean(y_test):.1%})")
    for t in ["QA", "Summary", "Data2txt"]:
        pos_t = sum(1 for x in cache_final_test if x["task_type"] == t and x["y_true"])
        print(f"  {t:<10}: Positives = {pos_t} / 100 ({pos_t}%)")

    # 4. Train Final Frozen Models on 900 Training Pool
    # (a) Per-task Separate Logistic Regression
    final_separate_models = {}
    for t in ["QA", "Summary", "Data2txt"]:
        t_data = [x for x in train_pool_900 if x["task_type"] == t]
        X_t = np.array([extract_base_12_features(x) for x in t_data])
        y_t = np.array([x["y_true"] for x in t_data], dtype=bool)
        clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=300, random_state=42)
        clf.fit(X_t, y_t)
        final_separate_models[t] = clf

    # (b) Global Logistic Regression from Step 2 (trained on 14-dim tabular features)
    X_train_global = np.array([extract_tabular_features(x) for x in train_pool_900])
    y_train_global = np.array([x["y_true"] for x in train_pool_900], dtype=bool)
    clf_global = LogisticRegression(class_weight="balanced", C=0.5, max_iter=300, random_state=42)
    clf_global.fit(X_train_global, y_train_global)

    # (c) Task-Interaction Model (38-dim features)
    X_train_inter = np.array([extract_interaction_features(x) for x in train_pool_900])
    clf_inter = LogisticRegression(class_weight="balanced", C=0.5, max_iter=400, random_state=42)
    clf_inter.fit(X_train_inter, y_train_global)

    # 5. Execute Predictions for All Systems
    # System 1: Flag-all
    pred_flag_all = np.ones(len(y_test), dtype=bool)

    # System 2: Flag-none
    pred_flag_none = np.zeros(len(y_test), dtype=bool)

    # System 3: Original strict system (faith < 0.99 or con > 0 or unsup > 0)
    def eval_orig_strict(cached_list, ent_th=0.50, con_th=0.40, faith_th=0.99):
        preds = []
        for ex in cached_list:
            claims = ex.get("claims", [])
            if not claims:
                preds.append(False)
                continue
            sup = sum(1 for c in claims if c.get("entailment_prob", 0.0) >= ent_th)
            con = sum(1 for c in claims if c.get("contradiction_prob", 0.0) >= con_th)
            unsup = len(claims) - sup - con
            faith = sup / len(claims)
            is_h = (con > 0) or (unsup > 0) or (faith < faith_th)
            preds.append(is_h)
        return np.array(preds, dtype=bool)

    pred_orig_strict = eval_orig_strict(cache_final_test)

    # System 4: Tuned embedding baseline (per-task tuned via CV for R=0.60)
    # Thresholds: QA=0.3637, Summary=0.3444, Data2txt=0.3890
    emb_thresholds = {"QA": 0.3637, "Summary": 0.3444, "Data2txt": 0.3890}
    pred_tuned_emb = []
    for ex in cache_final_test:
        score = extract_baseline_feature(ex)
        th = emb_thresholds[ex["task_type"]]
        pred_tuned_emb.append(score >= th)
    pred_tuned_emb = np.array(pred_tuned_emb, dtype=bool)

    # System 5: Previous global-threshold improved system (t = 0.70)
    X_test_global = np.array([extract_tabular_features(x) for x in cache_final_test])
    probs_global = clf_global.predict_proba(X_test_global)[:, 1]
    pred_global_improved = probs_global >= 0.70

    # System 6: New Per-Task System (Separate models, CV-calibrated for R=0.60)
    # Thresholds: QA=0.5467, Summary=0.4452, Data2txt=0.5110
    task_thresholds_sep = {"QA": 0.5467, "Summary": 0.4452, "Data2txt": 0.5110}
    pred_per_task = []
    for ex in cache_final_test:
        t = ex["task_type"]
        feat = extract_base_12_features(ex).reshape(1, -1)
        prob = final_separate_models[t].predict_proba(feat)[0, 1]
        pred_per_task.append(prob >= task_thresholds_sep[t])
    pred_per_task = np.array(pred_per_task, dtype=bool)

    # System 7: New Per-Task Interaction System (Interaction model, CV-calibrated for R=0.60)
    # Thresholds: QA=0.3646, Summary=0.3466, Data2txt=0.7527
    task_thresholds_inter = {"QA": 0.3646, "Summary": 0.3466, "Data2txt": 0.7527}
    X_test_inter = np.array([extract_interaction_features(x) for x in cache_final_test])
    probs_inter = clf_inter.predict_proba(X_test_inter)[:, 1]
    pred_per_task_inter = []
    for i, ex in enumerate(cache_final_test):
        t = ex["task_type"]
        pred_per_task_inter.append(probs_inter[i] >= task_thresholds_inter[t])
    pred_per_task_inter = np.array(pred_per_task_inter, dtype=bool)

    # 6. Compute Full Metrics and Bootstrap 95% CIs
    print("\n" + "=" * 80)
    print("COMPUTING FINAL METRICS WITH 1,000 BOOTSTRAP ITERATIONS...")
    print("=" * 80)

    systems = {
        "flag_all": pred_flag_all,
        "flag_none": pred_flag_none,
        "original_strict": pred_orig_strict,
        "tuned_embedding_baseline": pred_tuned_emb,
        "global_threshold_improved": pred_global_improved,
        "new_per_task_system": pred_per_task,
        "new_per_task_interaction_system": pred_per_task_inter,
    }

    results = {}
    for name, preds in systems.items():
        results[name] = evaluate_system_full(y_test, preds, test_tasks)

    # 7. Print Comprehensive Formatted Report
    def print_table(title, get_metrics_fn):
        print("\n" + "=" * 95)
        print(title)
        print("=" * 95)
        hdr = "{:<32} | {:<7} | {:<7} | {:<7} | {:<7} | {:<4} | {:<4} | {:<4} | {:<4}"
        print(hdr.format("System", "Prec", "Rec", "F1", "Acc", "TP", "FP", "TN", "FN"))
        print("-" * 95)
        for name in [
            "flag_all",
            "flag_none",
            "original_strict",
            "tuned_embedding_baseline",
            "global_threshold_improved",
            "new_per_task_system",
            "new_per_task_interaction_system",
        ]:
            m = get_metrics_fn(name)
            print(hdr.format(name, m["precision"], m["recall"], m["f1"], m["accuracy"], m["tp"], m["fp"], m["tn"], m["fn"]))
            print(f"   95% CI: P [{m['precision_ci'][0]:.4f}, {m['precision_ci'][1]:.4f}] | R [{m['recall_ci'][0]:.4f}, {m['recall_ci'][1]:.4f}] | F1 [{m['f1_ci'][0]:.4f}, {m['f1_ci'][1]:.4f}]")

    print_table("OVERALL TEST SET PERFORMANCE (N=300, Positives=129)", lambda name: results[name]["overall"])

    for t in ["QA", "Summary", "Data2txt"]:
        pos_t = results["flag_all"]["by_task"][t]["positives"]
        print_table(f"TASK: {t} (N=100, Positives={pos_t})", lambda name: results[name]["by_task"][t])

    # 8. Save Everything to per_task_calibrated_results.json
    final_output = {
        "metadata": {
            "training_pool_samples": len(train_pool_900),
            "final_test_samples": len(cache_final_test),
            "chosen_target_recall": 0.60,
            "chosen_rationale": "R=0.60 guarantees non-zero sensitivity on QA and Summary while preserving maximum precision and high overall accuracy, avoiding the global threshold collapse.",
            "thresholds_per_task_separate": task_thresholds_sep,
            "thresholds_per_task_interaction": task_thresholds_inter,
            "thresholds_per_task_embedding_baseline": emb_thresholds,
            "global_threshold_previous": 0.70,
        },
        "results": results,
    }

    out_path = EXPERIMENTS_DIR / "per_task_calibrated_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSuccessfully saved all final results to {out_path}.")


if __name__ == "__main__":
    main()
