import pytest
from humor_generator_v35.training.checkpoint_selection import (
    selection_metric, selection_score, resume_best, selection_record,
)


def test_margin_only_improvement_does_not_win_caption_selection():
    before = dict(total=3.0, caption_nll=1.0)
    after = dict(total=2.0, caption_nll=1.1)
    metric = selection_metric({'training': {'stage': 'caption', 'checkpoint_selection_metric': 'caption_nll'}})
    assert selection_score(after, metric) > selection_score(before, metric)
    assert selection_score(after, 'total') < selection_score(before, 'total')


def test_selection_roundtrip_and_legacy_resume():
    assert selection_metric({'training': {}}) == 'total'
    assert resume_best(None, 'caption_nll') == float('inf')
    assert resume_best({'best_validation_total': 2.0}, 'total') == 2.0
    for metric in ('total', 'caption_nll'):
        record = selection_record(metric, 1.2)
        assert resume_best(record, metric) == 1.2
    assert 'best_validation_total' not in selection_record('caption_nll', 1.2)
    with pytest.raises(ValueError, match='legacy'):
        resume_best({'best_validation_total': 2.0}, 'caption_nll')
    with pytest.raises(ValueError, match='mismatch'):
        resume_best(selection_record('total', 1.2), 'caption_nll')


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), True, '1.2'])
def test_selection_fails_on_invalid_metrics(value):
    with pytest.raises(ValueError):
        selection_score({'caption_nll': value}, 'caption_nll')
