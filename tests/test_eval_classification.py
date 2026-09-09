from unittest.mock import MagicMock

import eval_classification as evaluation


def test_scoring_and_confusion_matrix_preserve_misroute_direction(monkeypatch):
    classifier = MagicMock(side_effect=[{"classification": "GENERAL"}, {"classification": "CALCULATION"}])
    monkeypatch.setattr(evaluation, "classify_question", classifier)
    questions = [
        {"id": "test-biomed", "question": "test biomedical question", "gold_category": "BIOMED", "trap_class": "test trap"},
        {"id": "test-calc", "question": "test arithmetic question", "gold_category": "CALCULATION", "trap_class": "test trap"},
    ]
    results = [evaluation.run_one(question) for question in questions]
    assert [result["correct"] for result in results] == [False, True]
    assert results[0]["id"] == "test-biomed"
    classifier.assert_any_call({"resolved_question": "test biomedical question"})
    assert evaluation.build_confusion_matrix(results) == {
        "BIOMED": {"BIOMED": 0, "CALCULATION": 0, "GENERAL": 1},
        "CALCULATION": {"BIOMED": 0, "CALCULATION": 1, "GENERAL": 0},
        "GENERAL": {"BIOMED": 0, "CALCULATION": 0, "GENERAL": 0},
    }
