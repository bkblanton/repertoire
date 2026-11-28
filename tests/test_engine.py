import json

import pytest
from helpers import position
from test_free_transpositions import scored  # noqa: F401

from repertoire.engine import (
    analyze,
    exits_summary,
    expected_score,
    is_trivial_recapture,
    judgement,
    normalize_pv_uci,
    position_sharpness,
)
from repertoire.evals import Evaluation, Store
from repertoire.preparation import analyze as preparation
from repertoire.vulnerabilities import analyze as vulnerabilities


def evaluation(cp=None, mate=None, depth=30, line='e2e4'):
    pv = {'line': line} | ({'cp': cp} if cp is not None else {'mate': mate})
    return Evaluation(depth=depth, knodes=1, pvs=[pv], source='export', retrieved_at='2026-10-05')


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
    assert set(result['positions']) == {
        position('e4 e5 Nc3 Qf6'),
        position('e4 e5 Nc3 Qf6 Nd5'),
        position('e4 e5 Nc3 Qf6 Nd5 Qd8'),
        position('e4 e5 Nc3'),
    }
    assert result['positions'][position('e4 e5 Nc3')]['cp'] == 30


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
    assert '**Opponent mistakes**' not in text
    assert '**Prepared opponent mistakes**' not in text  # Qd8 has no prepared answer
    unpunished = text.split('**Unpunished opponent mistakes**', 1)[1]
    assert 'Nd5 Qd8' in unpunished
    assert engine_sentence(bundle).startswith('Where preparation ends, the engine gives you')
    path.with_suffix('.engine.json').unlink()
    assert engine_sentence(load([path], strict=False)[0]) is None


def test_missing_evaluations_are_mentioned_once_per_page(complete, tmp_path):
    from repertoire.report.generate import generate
    from repertoire.report.sections import NO_EVALUATIONS

    path, _ = complete
    out = tmp_path / 'out'
    generate([path], out / 'report.md', out / 'summary.md')
    for name in ('summary.md', 'report.md'):
        assert (out / name).read_text(encoding='utf-8').count(NO_EVALUATIONS) == 1
    pages = list((out / 'chapters').glob('*.md'))
    assert pages and all(p.read_text(encoding='utf-8').count(NO_EVALUATIONS) == 1 for p in pages)
    assert 'Engine where prep ends' not in (out / 'report.md').read_text(encoding='utf-8')


def test_castling_normalization_sharpness_and_recapture_helpers():
    castling_pos = position('e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6')
    assert normalize_pv_uci(castling_pos, 'e1h1') == 'e1g1'
    assert normalize_pv_uci(position('e4 e5 Ke2'), 'e1h1') == 'e1h1'  # no castling right -> unchanged
    assert normalize_pv_uci(castling_pos, 'e2e4') == 'e2e4'  # non-castling UCI unchanged

    # Multi-PV preferred over children; 1-ply child eval fallback works when multi-PV is absent.
    pos = position('e4 e5 Nf3')
    evals_multi = {
        pos: Evaluation(
            depth=30,
            knodes=10,
            pvs=[{'cp': -90, 'line': 'b8c6'}, {'cp': 35, 'line': 'g8f6'}],
            source='export',
            retrieved_at='now',
        )
    }
    assert position_sharpness(pos, True, evals_multi) is None
    sh_multi = position_sharpness(pos, False, evals_multi, allow_children=False)
    assert sh_multi is not None and sh_multi['best_move'] == 'b8c6' and sh_multi['source'] == 'multipv'
    assert sh_multi['gap_pp'] > 10

    evals_children = {
        pos: evaluation(cp=-90, line='b8c6'),
        position('e4 e5 Nf3 Nf6'): evaluation(cp=35, line='d2d4'),
    }
    assert position_sharpness(pos, False, evals_children, allow_children=False) is None
    sh_child = position_sharpness(pos, False, evals_children, allow_children=True)
    assert sh_child is not None and sh_child['best_move'] == 'b8c6' and sh_child['source'] == 'children'
    assert sh_child['gap_pp'] > 10

    # Obvious single recapture vs real 2-piece recapture choice.
    assert is_trivial_recapture(position('e4 d5'), 'e4d5', 'd8d5', 'g8f6')
    assert not is_trivial_recapture(position('e4 e5 Nf3 Nc6 Bb5 a6 Bxc6'), 'b5c6', 'd7c6', 'b7c6')
    assert not is_trivial_recapture(position('e4 d5'), 'e4d5', 'g8f6', 'd8d5')


def test_engine_audit_finds_tests_sharp_replies_cliffs_and_refutations(scored, evaluation_store):  # noqa: F811
    from repertoire.report.bundle import load
    from repertoire.report.generate import generate
    from repertoire.report.links import bundle_refs
    from repertoire.report.pages import summary_report
    from repertoire.report.sections import engine_section, summary_critical_engine_section

    path, cache, _ = scored
    companions(path, cache)
    before_nd5 = position('e4 e5 Nc3 Qf6')
    after_nd5 = position('e4 e5 Nc3 Qf6 Nd5')
    qd8 = position('e4 e5 Nc3 Qf6 Nd5 Qd8')
    c6 = position('e4 e5 Nc3 Qf6 Nd5 c6')
    with Store(evaluation_store) as store:
        # 1. Own move 3.Nd5 loses > 5pp; pv0 is f6d8 (unprepared refutation).
        store.put(before_nd5, evaluation(cp=120, line='f1c4'))
        store.put(after_nd5, evaluation(cp=20, line='f6d8'))
        # 2. Unprepared opponent reply 3...Qd8 lands on an only-move tactic (4.Nc3 vs 4.d4).
        store.put(
            qd8,
            Evaluation(
                depth=32,
                knodes=200,
                pvs=[{'cp': 90, 'line': 'd5c3'}, {'cp': -40, 'line': 'd2d4'}],
                source='export',
                retrieved_at='now',
            ),
        )
        # 3. Unprepared reply 3...c6 has 1-ply child evals loaded via lookahead!
        store.put(c6, evaluation(cp=80, line='d5c3'))
        store.put(position('e4 e5 Nc3 Qf6 Nd5 c6 Nc3'), evaluation(cp=80, line='d7d5'))
        store.put(position('e4 e5 Nc3 Qf6 Nd5 c6 d4'), evaluation(cp=-20, line='c6d5'))

    result = analyze(path, cache)
    # `positions` keeps only direct engine positions (not 1-ply lookahead children).
    assert position('e4 e5 Nc3 Qf6 Nd5 c6 Nc3') not in result['positions']
    nd5_id = next(k for k in result['moves'] if k.startswith(before_nd5))
    assert result['moves'][nd5_id]['refutation'] == {'move': 'f6d8', 'move_san': 'Qd8', 'prepared': False}

    overall_crit = next(s for s in result['scopes'] if s['id'] == 'overall')['critical']
    tests = overall_crit['unprepared_engine_tests']
    assert any(
        t['move'] == 'f6d8' and t['is_refutation'] and t['reply_sharpness']['best_move_san'] == 'Nc3' for t in tests
    )
    sharp = overall_crit['sharp_unprepared_replies']
    assert {r['move'] for r in sharp} >= {'f6d8', 'c7c6'}
    c6_row = next(r for r in sharp if r['move'] == 'c7c6')
    assert c6_row['sharpness']['source'] == 'children' and c6_row['sharpness']['best_move_san'] == 'Nc3'

    path.with_suffix('.engine.json').write_text(json.dumps(result))
    bundle = load([path], strict=False)[0]
    refs = bundle_refs(bundle)
    eng_md = '\n'.join(engine_section(bundle, refs, 10))
    assert '**Unprepared opponent engine tests**' in eng_md
    assert '**Sharp unprepared replies and traps**' in eng_md
    assert 'Refutation' in eng_md and 'Qd8 (unprepared)' in eng_md
    sum_crit = '\n'.join(summary_critical_engine_section(bundle, refs))
    assert 'Critical lines to check (engine)' in sum_crit
    assert '[Full engine view and audit](#white-engine)' in sum_crit
    summary_text = summary_report([bundle])
    assert 'Critical lines to check (engine)' in summary_text
    # Ensure full report generation has no broken links or em dashes.
    generate([path])
    full_md = (path.parent / 'report.md').read_text(encoding='utf-8')
    assert '\u2014' not in full_md and '\u2014' not in summary_text
