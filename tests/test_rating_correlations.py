import hashlib
import json

import numpy as np
import pytest

from repertoire_score.rating_correlations import chapter_association, estimate, load_color, within_parent


def replies():
    # Parent offsets reverse the pooled relationship; every local slope is negative.
    return [
        dict(position=str(i), rating=rating + i * 400, score=score + i * 0.2, weight=weight)
        for i in range(4)
        for rating, score, weight in ((1500, 0.6, 1), (1600, 0.55, 2), (1700, 0.5, 1))
    ]


def test_within_parent_removes_position_offsets():
    result = within_parent(replies())
    assert result['correlation'] == pytest.approx(-1)
    assert result['slope_percent_per_100_rating'] == pytest.approx(-5)
    assert result['replies'] == 12
    assert result['parents'] == 4
    assert result == within_parent(replies())
    assert np.corrcoef([r['rating'] for r in replies()], [r['score'] for r in replies()])[0, 1] > 0


def test_no_variation_and_singletons_are_not_invented_associations():
    rows = [dict(position='a', rating=1500, score=0.5, weight=1), dict(position='b', rating=1600, score=0.4, weight=1)]
    assert within_parent(rows)['correlation'] is None
    rows[1]['position'] = 'a'
    result = within_parent(rows)
    assert result['correlation'] == pytest.approx(-1)
    rows[1]['score'] = 0.5
    assert within_parent(rows)['correlation'] is None


def test_grouped_chapter_moments_reproduce_weighted_correlation_and_slope():
    rows = [
        dict(rating=x, score=y, reach=w, cluster=str(i // 2))
        for i, (x, y, w) in enumerate(
            (
                (1500, 0.65, 0.1),
                (1600, 0.6, 0.3),
                (1700, 0.62, 0.4),
                (1800, 0.55, 0.2),
                (1900, 0.5, 0.5),
                (2000, 0.52, 0.1),
            )
        )
    ]
    x, y, w = (np.array([r[k] for r in rows]) for k in ('rating', 'score', 'reach'))
    dx, dy = x - np.average(x, weights=w), y - np.average(y, weights=w)
    result = chapter_association(rows, 'score', True)
    assert result['correlation'] == pytest.approx(
        np.sum(w * dx * dy) / np.sqrt(np.sum(w * dx * dx) * np.sum(w * dy * dy))
    )
    assert result['slope_percent_per_100_rating'] == pytest.approx(10000 * np.sum(w * dx * dy) / np.sum(w * dx * dx))
    assert result['clusters'] == 3
    assert result['chapters'] == 6


def test_pooled_estimate_from_group_moments():
    result = estimate([[1, 1, 1], [1, -1, 1]], True)
    assert result['correlation'] == pytest.approx(0)
    assert result['clusters'] == 2 and 'confidence_intervals_95' not in result
    assert estimate([], True)['status'] == 'too_few_groups'


def fixture(tmp_path):
    source = tmp_path / 'study.pgn'
    source.write_text('1. e4 e5 *', encoding='utf-8')
    path = tmp_path / 'white.json'

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    input_hash = sha(source)
    path.write_text(
        json.dumps(
            dict(
                color='white',
                manifest=dict(input_path=str(source), input_sha256=input_hash, filters={}, sparse_threshold=30),
                chapters=[
                    dict(
                        id='c',
                        name='Test',
                        prepared_positions=['a'],
                        score=dict(raw_empirical_score=0.6, entry_probability=0.2),
                        entry_baseline=dict(difference_pp=5, raw_score=0.55),
                    )
                ],
            )
        ),
        encoding='utf-8',
    )
    manifest = dict(report_sha256=sha(path), input_sha256=input_hash, filters={})
    rows = [
        dict(
            id=f'a|{i}',
            kind='opponent',
            position='a',
            sample_count=n,
            sparse=n < 30,
            move_score=0.6,
            branch_reach=0.1,
            prepared=i == 0,
            reference_score=0.55,
            line='line',
            chapters=['c'],
        )
        for i, n in enumerate((100, 30, 29))
    ]
    vulnerabilities = path.with_suffix('.vulnerabilities.json')
    vulnerabilities.write_text(
        json.dumps(dict(color='white', manifest=manifest, overall={'all_signed_rows': rows})), encoding='utf-8'
    )
    ratings = path.with_suffix('.ratings.json')
    ratings.write_text(
        json.dumps(
            dict(
                color='white',
                manifest=dict(manifest, supporting_sha256={'vulnerabilities': sha(vulnerabilities)}),
                scopes=[
                    dict(id='overall', moves={r['id']: dict(mean=1500, parent_mean=1550) for r in rows}),
                    dict(id='c', score_evidence=dict(mean=1600, known_coverage=1)),
                ],
            )
        ),
        encoding='utf-8',
    )
    return path, ratings, source


def test_saved_inputs_filter_sparse_rows_and_guard_provenance(tmp_path):
    path, ratings, source = fixture(tmp_path)
    result = load_color(path)
    assert len(result['replies']) == 2
    assert result['exclusions']['replies'] == {'sparse': 1}
    assert result['chapters'][0]['delta'] == pytest.approx(0.05)
    assert result['chapters'][0]['rating'] == 1600
    data = json.loads(ratings.read_text())
    data['manifest']['report_sha256'] = 'wrong'
    ratings.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='does not match'):
        load_color(path)


def test_source_changed_guard(tmp_path):
    path, ratings, source = fixture(tmp_path)
    source.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='PGN changed'):
        load_color(path)
