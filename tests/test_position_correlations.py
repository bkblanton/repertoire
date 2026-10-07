import json

import httpx
import numpy as np
import pytest
from helpers import position

from repertoire_score.depth import prepared_depth_values
from repertoire_score.evaluate import COMPLETED, backward, reaches
from repertoire_score.model import Branch, ModelNode
from repertoire_score.position_correlations import analyze, metrics, weighted_rank
from repertoire_score.stats import correlation


def test_depth_and_reach_merge_transposed_continuations_in_each_joint_draw():
    model = {
        'root': ModelNode('own', [Branch(target='left', weight=0.5), Branch(target='right', weight=0.5)]),
        'left': ModelNode('opponent', [Branch(target='shared'), Branch(counts=[50, 0, 50])]),
        'right': ModelNode('opponent', [Branch(target='shared'), Branch(counts=[50, 0, 50])]),
        'shared': ModelNode('own', [Branch(target='end', weight=1.0)]),
        'end': ModelNode('stop', [Branch(counts=[50, 0, 50])]),
    }
    order = ['end', 'shared', 'left', 'right', 'root']
    p, q = np.array([0.6, 0.8]), np.array([0.2, 0.4])
    outcome = np.array([0.7, 0.9])
    sampled = {
        'root': [(0.5, None), (0.5, None)],
        'left': [(p, None), (1 - p, 0.5)],
        'right': [(q, None), (1 - q, 0.5)],
        'shared': [(1.0, None)],
        'end': [(1.0, outcome)],
    }
    depth = {k: low for k, (low, _) in prepared_depth_values(model, order, sampled).items()}
    reach = reaches(model, order, sampled, {'root': 1.0})
    assert np.allclose(depth['root'], [1.4, 1.6])
    assert np.allclose(depth['left'], p)  # future depth excludes the root's selected move
    assert np.allclose(depth['shared'], 1)
    assert np.allclose(depth['end'], 0)
    assert np.allclose(reach['shared'], [0.4, 0.6])
    assert np.allclose(reach['end'], reach['shared'])
    values = backward(model, order, sampled, 30, 2)
    assert np.allclose(values['root'][COMPLETED], reach['shared'] * outcome + (1 - reach['shared']) * 0.5)


def test_weighted_ranks_and_correlations_ignore_artificial_branch_duplication():
    x, y, w = np.array([0.0, 1.0, 2.0]), np.array([3.0, 1.0, 5.0]), np.array([1.0, 2.0, 7.0])
    assert weighted_rank(x, w) == pytest.approx([0.05, 0.2, 0.65])
    original = metrics(x, y, w)
    split = metrics([0.0, 1.0, 1.0, 2.0], [3.0, 1.0, 1.0, 5.0], [1.0, 1.0, 1.0, 7.0])
    for key in ('reach_weighted_pearson', 'reach_weighted_spearman', 'slope_pp_per_move'):
        assert original[key] == pytest.approx(split[key])
    assert original['reach_weighted_pearson'] == pytest.approx(correlation(x, y, w))
    assert original['reach_weighted_pearson'] != pytest.approx(original['pearson'])
    assert metrics([1.0, 1.0], [2.0, 3.0], [1.0, 1.0])['reach_weighted_pearson'] is None
    assert all(value is None for value in metrics([], [], []).values())
    assert all(value is None for value in metrics([0.0, 1.0], [0.0, 1.0], [0.0, 0.0]).values())


def test_saved_analysis_is_cache_only_reproducible_and_uses_overall_policy(complete, monkeypatch):
    path, report = complete
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('Must use cached evidence'))
    original = {p: p.read_bytes() for p in path.parent.glob('*.json')}
    source = path.parent / 'fixture.pgn'
    pgn = source.read_bytes()
    output = path.parent / 'prepared-depth-gain-correlation.json'
    first = analyze([path], output, path.parent / 'cache')
    second = analyze([path], output, path.parent / 'cache')
    assert first['network_requests'] == 0
    assert first['results'] == second['results']
    result = first['results']['white']
    assert result['validation']['root_depth_reproduced'] == pytest.approx(
        report['overall']['prepared_depth']['expected_moves']
    )
    assert result['validation']['root_score_reproduced']
    assert len({r['id'] for r in result['decisions']}) == result['n']
    # Alternative Tarrasch decisions belong in their chapter report, not in this overall-policy correlation.
    assert all(r['move'] != 'b1d2' for r in result['decisions'])
    assert result['position_depths'][position('')] == pytest.approx(
        report['overall']['prepared_depth']['expected_moves']
    )
    saved = json.loads(path.with_suffix('.vulnerabilities.json').read_bytes())
    rows = {r['id']: r for r in saved['overall']['all_signed_rows']}
    for row in result['decisions']:
        assert row['depth'] == result['position_depths'][row['target']]
        assert row['reach'] == pytest.approx(rows[row['id']]['branch_reach'])
        assert row['future_preparation_gain_pp'] == pytest.approx(
            100 * (row['repertoire_score'] - row['move_database_score'])
        )
        # Total gain is immediate move gain plus future preparation gain.
        total = 100 * (row['repertoire_score'] - rows[row['id']]['reference_score'])
        immediate = 100 * (row['move_database_score'] - rows[row['id']]['reference_score'])
        assert total == pytest.approx(immediate + row['future_preparation_gain_pp'])
    assert 'confidence_intervals_95' not in result
    assert source.read_bytes() == pgn
    assert all(p.read_bytes() == content for p, content in original.items())


def test_sparse_samples_and_stale_inputs_are_rejected(complete):
    path, _ = complete
    cache, output = path.parent / 'cache', path.parent / 'prepared-depth-gain-correlation.json'
    companion = path.with_suffix('.vulnerabilities.json')
    saved = json.loads(companion.read_bytes())
    own = [r for r in saved['overall']['all_signed_rows'] if r['kind'] == 'own']
    for row in own:
        row['parent_sparse'] = True
    companion.write_text(json.dumps(saved), encoding='utf-8')
    result = analyze([path], output, cache)['results']['white']
    assert result['n'] == 0 and result['exclusions']['sparse'] == len(own)
    assert result['reach_weighted_pearson'] is None
    saved['manifest']['report_sha256'] = 'wrong'
    companion.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='snapshot'):
        analyze([path], output, cache)
    (path.parent / 'fixture.pgn').write_text('changed source')
    with pytest.raises(ValueError, match='PGN changed'):
        analyze([path], output, cache)
