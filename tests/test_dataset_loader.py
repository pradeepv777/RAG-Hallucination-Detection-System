"""
Unit tests for dataset loader and preprocessing (dataset_loader.py).
"""

from dataset_loader import RAGTruthLoader, format_source_info


def test_format_source_info_qa():
    qa_source = {
        "question": "What is gravity?",
        "passages": "passage 1: Gravity is a fundamental interaction."
    }
    context, question = format_source_info(qa_source, "QA")
    assert "Gravity is a fundamental interaction" in context
    assert question == "What is gravity?"


def test_format_source_info_summary():
    summary_source = "A long article about space exploration..."
    context, question = format_source_info(summary_source, "Summary")
    assert context == summary_source
    assert question == ""


def test_format_source_info_data2txt():
    data_source = {"name": "Cafe Roma", "city": "Rome", "stars": 4.5}
    context, question = format_source_info(data_source, "Data2txt")
    assert "name: Cafe Roma" in context
    assert "stars: 4.5" in context


def test_dataset_loader_splits():
    loader = RAGTruthLoader("dataset")
    test_examples = loader.load_dataset(split="test", limit=10)
    assert len(test_examples) == 10
    for ex in test_examples:
        assert ex.split == "test"
        assert ex.quality == "good"
        assert len(ex.context) > 0
        assert len(ex.response) > 0


def test_claim_overlaps_ground_truth():
    response = "The team won the championship in 2020 and celebrated in June."
    labels = [
        {"start": 34, "end": 38, "text": "2020", "label_type": "Evident Conflict"}
    ]

    # Overlapping claim
    claim1 = "The team won the championship in 2020."
    assert RAGTruthLoader.claim_overlaps_ground_truth(claim1, response, labels) is True

    # Non-overlapping claim
    claim2 = "They celebrated in June."
    assert RAGTruthLoader.claim_overlaps_ground_truth(claim2, response, labels) is False
