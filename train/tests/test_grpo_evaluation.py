from train.src.training.grpo_evaluation import comparison, summarize


def row(index, tp, fp, fn):
    return dict(sample_index=index, status='complete', tp=tp, fp=fp, fn=fn, unparsed_attempts=0)


def test_missing_prediction_cannot_look_like_fp_improvement():
    baseline = [row(0, 2, 3, 1), row(1, 1, 99, 0)]
    current = [row(0, 2, 1, 1), dict(sample_index=1, status='failed')]
    result = comparison(baseline, current)
    assert result['paired_qa'] == 1
    assert result['delta']['fp'] == -2
    assert result['delta']['recall'] == 0


def test_silence_drops_recall_and_empty_intersection_is_not_zero_score():
    result = comparison([row(0, 2, 9, 0)], [row(0, 0, 0, 2)])
    assert result['delta']['fp'] == -9
    assert result['current']['recall'] == result['current']['f1'] == 0
    assert comparison([row(0, 2, 0, 0)], [])['current'] is None
    assert summarize([row(0, 2, 1, 2)])['f1'] == 4/7
