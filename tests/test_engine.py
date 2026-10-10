import json

import pytest
from helpers import position
from test_free_transpositions import scored  # noqa: F401

from repertoire.engine import analyze, exits_summary, expected_score, judgement
from repertoire.evals import Evaluation, Store
from repertoire.preparation import analyze as preparation
from repertoire.vulnerabilities import analyze as vulnerabilities


def evaluation(cp=None, mate=None, depth=30):
    line = {'line': 'e2e4'} | ({'cp': cp} if cp is not None else {'mate': mate})
    return Evaluation(depth=depth, knodes=1, pvs=[line], source='export', retrieved_at='2026-10-05')


def test_expected_score_follows_the_lichess_curve_from_your_side():
    assert expected_score(evaluation(cp=0), True) == pytest.approx(0.5)
    white = expected_score(evaluation(cp=100), True)
    assert white == pytest.approx(1 / (1 + 2.718281828459045 ** (-0.368208)))
    assert expected_score(evaluation(cp=100), False) == pytest.approx(1 - white)
    assert expected_score(evaluation(mate=3), True) == 1.0
    assert expected_score(evaluation(mate=-3), True) == 0.0
    assert expected_score(evaluation(mate=-3), False) == 1.0
    assert expected_score(None, True) is None


def test_judgements_use_the_lichess_thresholds():
    assert [judgement(x) for x in (None, 4.9, 5, 10, 15)] == [None, None, 'inaccuracy', 'mistake', 'blunder']


def test_exits_keep_evaluations_floors_and_missing_apart():
    stops = [
        dict(position='a', parent_position='p', reach=0.5, score=0.6),
        dict(position='b', parent_position='p', reach=0.3, score=0.5),
        dict(position='c', parent_position='q', reach=0.2, score=0.4),
    ]
    summary = exits_summary(stops, {'a': 0.55, 'p': 0.52})
    assert summary['evaluated_reach'] == 0.5 and summary['floored_reach'] == 0.3
    assert summary['missing_reach'] == pytest.approx(0.2)
    assert summary['engine_score'] == pytest.approx(0.55) and summary['database_score'] == pytest.approx(0.6)
    assert summary['practical_minus_engine_pp'] == pytest.approx(5)
    assert summary['engine_floor'] == pytest.approx((0.5 * 0.55 + 0.3 * 0.52) / 0.8)


def companions(path, cache):
    for name, analyze_family in (('vulnerabilities', vulnerabilities), ('preparation', preparation)):
        path.with_suffix(f'.{name}.json').write_text(json.dumps(analyze_family(path, cache)))


def test_engine_view_scores_moves_exits_and_transpositions(scored, evaluation_store):  # noqa: F811
    path, cache, _ = scored
    companions(path, cache)
    before_nd5 = position('e4 e5 Nc3 Qf6')
    with Store(evaluation_store) as store:
        store.put(before_nd5, evaluation(cp=90))
        store.put(position('e4 e5 Nc3 Qf6 Nd5'), evaluation(cp=20))
        store.put(position('e4 e5 Nc3 Qf6 Nd5 Qd8'), evaluation(cp=80))
        store.put(position('e4 e5 Nc3'), evaluation(cp=30))
    result = analyze(path, cache)
    assert result['status'] == 'resolved'
    nd5 = next(v for k, v in result['moves'].items() if k.startswith(before_nd5))
    assert nd5['loss_pp'] > 5 and nd5['judgement'] == 'inaccuracy'
    qd8 = next(v for k, v in result['moves'].items() if k.endswith('|f6d8'))
    assert qd8['swing_pp'] > 0
    (transposition,) = result['transpositions'].values()
    assert transposition['loss_pp'] == pytest.approx(100 * (transposition['before'] - transposition['after']))
    overall = next(s for s in result['scopes'] if s['id'] == 'overall')['exits']
    assert overall['evaluated_reach'] > 0 and overall['floored_reach'] > 0
    total = overall['evaluated_reach'] + overall['floored_reach'] + overall['missing_reach']
    assert total == pytest.approx(overall['reach'])
    assert {e['basis'] for e in result['exits']} == {'evaluated', 'floor'}


def test_without_a_store_the_engine_view_is_unresolved(scored):  # noqa: F811
    path, cache, _ = scored
    companions(path, cache)
    result = analyze(path, cache)
    assert result['status'] == 'unresolved_evidence'
    assert all(v['loss_pp'] is None for v in result['moves'].values() if 'loss_pp' in v)


def test_report_shows_the_engine_view_and_the_summary_line(scored, evaluation_store):  # noqa: F811
    from repertoire.report.bundle import load
    from repertoire.report.links import bundle_refs
    from repertoire.report.sections import engine_section, engine_sentence

    path, cache, _ = scored
    companions(path, cache)
    with Store(evaluation_store) as store:
        store.put(position('e4 e5 Nc3 Qf6'), evaluation(cp=90))
        store.put(position('e4 e5 Nc3 Qf6 Nd5'), evaluation(cp=20))
        store.put(position('e4 e5 Nc3 Qf6 Nd5 Qd8'), evaluation(cp=80))
    path.with_suffix('.engine.json').write_text(json.dumps(analyze(path, cache)))
    (bundle,) = load([path], strict=False)
    assert 'engine' in bundle
    text = '\n'.join(engine_section(bundle, bundle_refs(bundle), 5))
    assert '### Engine view' in text and '**Your moves the engine questions**' in text
    assert 'inaccuracy' in text
    assert engine_sentence(bundle).startswith('Where preparation ends, the engine gives you')
    path.with_suffix('.engine.json').unlink()
    assert engine_sentence(load([path], strict=False)[0]) is None
