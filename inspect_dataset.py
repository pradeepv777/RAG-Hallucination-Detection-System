import json
from collections import Counter
from pathlib import Path

def inspect():
    resp_path = Path("dataset/response.jsonl")
    src_path = Path("dataset/source_info.jsonl")

    print("=== INSPECTING RAGTRUTH DATASET ===")
    
    # 1. Inspect source_info
    src_count = 0
    task_types = Counter()
    sample_src_by_task = {}
    source_ids = set()

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            src_count += 1
            item = json.loads(line)
            source_ids.add(item["source_id"])
            tt = item.get("task_type", "unknown")
            task_types[tt] += 1
            if tt not in sample_src_by_task:
                sample_src_by_task[tt] = item

    print(f"\nTotal source items: {src_count}")
    print(f"Unique source_ids: {len(source_ids)}")
    print(f"Task type distribution in source_info: {dict(task_types)}")

    for tt, sample in sample_src_by_task.items():
        print(f"\n--- Sample source_info for task_type: {tt} ---")
        print(f"Keys: {list(sample.keys())}")
        print(f"source_id: {sample.get('source_id')}")
        src_info = sample.get("source_info")
        if isinstance(src_info, dict):
            print(f"source_info keys: {list(src_info.keys())}")
            if "question" in src_info:
                print(f"Sample question: {src_info['question'][:100]}...")
            if "passages" in src_info:
                print(f"Passages count/type: {type(src_info['passages'])}, len: {len(src_info['passages'])}")
        elif isinstance(src_info, str):
            print(f"source_info preview (str): {src_info[:150]}...")
        else:
            print(f"source_info type: {type(src_info)}")

    # 2. Inspect response.jsonl
    resp_count = 0
    split_counts = Counter()
    quality_counts = Counter()
    model_counts = Counter()
    label_types = Counter()
    hallu_count = 0
    clean_count = 0
    samples_with_labels = []

    with open(resp_path, "r", encoding="utf-8") as f:
        for line in f:
            resp_count += 1
            item = json.loads(line)
            split_counts[item.get("split")] += 1
            quality_counts[item.get("quality")] += 1
            model_counts[item.get("model")] += 1
            
            labels = item.get("labels", [])
            if len(labels) > 0:
                hallu_count += 1
                for lbl in labels:
                    label_types[lbl.get("label_type")] += 1
                if len(samples_with_labels) < 2:
                    samples_with_labels.append(item)
            else:
                clean_count += 1

    print(f"\nTotal response items: {resp_count}")
    print(f"Split distribution: {dict(split_counts)}")
    print(f"Quality distribution: {dict(quality_counts)}")
    print(f"Model distribution: {dict(model_counts)}")
    print(f"Hallucinated responses (labels > 0): {hallu_count} ({hallu_count/resp_count:.1%})")
    print(f"Clean responses (labels == 0): {clean_count} ({clean_count/resp_count:.1%})")
    print(f"Label type distribution: {dict(label_types)}")

    print("\n--- Sample Hallucinated Response ---")
    if samples_with_labels:
        s = samples_with_labels[0]
        print(f"ID: {s.get('id')}, Source ID: {s.get('source_id')}, Model: {s.get('model')}")
        print(f"Response: {s.get('response')[:200]}...")
        print(f"Labels: {s.get('labels')}")

if __name__ == "__main__":
    inspect()
