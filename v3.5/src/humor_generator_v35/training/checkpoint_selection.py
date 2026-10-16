"""Explicit validation-only selection, separate from training loss weights."""
import math


def selection_metric(config):
    metric = config['training'].get('checkpoint_selection_metric', 'total')
    if metric not in {'total', 'caption_nll'}:
        raise ValueError(f'unsupported checkpoint selection metric: {metric}')
    if metric == 'caption_nll' and config['training']['stage'] != 'caption':
        raise ValueError('caption_nll selection requires caption stage')
    return metric


def selection_score(metrics, metric):
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'missing or non-finite checkpoint selection metric: {metric}')
    return float(value)


def resume_best(state, metric):
    if state is None:
        return float('inf')
    saved = state.get('checkpoint_selection')
    if saved is None:
        if metric != 'total':
            raise ValueError('legacy checkpoint cannot resume with a different selection metric')
        return selection_score({'total': state.get('best_validation_total')}, 'total')
    if saved.get('metric') != metric:
        raise ValueError('resume checkpoint selection metric mismatch')
    return selection_score({metric: saved.get('best_value')}, metric)


def selection_record(metric, best):
    result = {'checkpoint_selection': {'metric': metric, 'direction': 'min', 'best_value': best}}
    if metric == 'total':
        result['best_validation_total'] = best
    return result
