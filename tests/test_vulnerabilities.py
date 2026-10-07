import hashlib
import json

import chess
import pytest

from repertoire_score.evaluate import forward
from repertoire_score.explorer import Explorer, DEFAULT_FILTERS
from repertoire_score.graph import key
from repertoire_score.vulnerabilities import analyze, candidates, rank_scope, reaches, representative_lines
from helpers import data, graph, position, setup
from helpers import cache_row


def fixture(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nf3 *')
    root, e4, e5, leaf = g.roots[0], position('e4'), position('e4 e5'), position('e4 e5 Nf3')
    evidence = {
        root: data(60, 0, 40, [('e2e4', 10, 0, 10), ('d2d4', 50, 0, 30)]),
        e4: data(60, 0, 40, [('e7e5', 60, 0, 20), ('c7c5', 0, 0, 20)]),
        e5: data(60, 0, 40, [('g1f3', 4, 0, 6), ('b1c3', 56, 0, 34)]),
        leaf: data(50, 0, 50),
    }
    m, o, raw, _, v, _ = setup(g, True, evidence)
    return g, m, o, raw, v, evidence


def test_opponent_uses_prepared_values_and_parent_rows_without_child_query(tmp_path):
    g, m, o, raw, v, e = fixture(tmp_path)
    rows = candidates(g, m, raw, v, e, True, 30)
    scope = rank_scope(rows, reaches(m, o, raw, {g.roots[0]: 1}), representative_lines(g, m, o, raw, {g.roots[0]: 1}))
    reply = scope['rankings']['opponent'][0]
    assert reply['move_san'] == 'c5'
    assert reply['branch_reach'] == pytest.approx(.2)
    assert reply['reference_score'] == pytest.approx(.4)  # .8 * prepared .5 + .2 * deviation 0
    assert reply['weighted_drag_pp'] == pytest.approx(8)
    prepared = next(r for r in scope['all_signed_rows'] if r['kind'] == 'opponent' and r['move_san'] == 'e5')
    assert prepared['move_score'] == .5  # not the .75 from that move row
    assert prepared['weighted_drag_pp'] == pytest.approx(-8)
    assert position('e4 c5') not in e  # deviation child statistics are never necessary


def test_own_deficit_forced_probability_and_same_table_alternative(tmp_path):
    g, m, o, raw, v, e = fixture(tmp_path)
    rows = candidates(g, m, raw, v, e, True, 30)
    scope = rank_scope(rows, reaches(m, o, raw, {g.roots[0]: 1}), {})
    e4, nf3 = scope['rankings']['own']
    assert nf3['move_san'] == 'Nf3'
    assert nf3['parent_reach'] == pytest.approx(.8)
    assert nf3['branch_probability'] == 1  # not historical 10%
    assert nf3['weighted_drag_pp'] == pytest.approx(8)
    assert e4['weighted_drag_pp'] == pytest.approx(20)
    assert nf3['alternative']['san'] == 'Nc3'
    assert nf3['alternative']['sample_count'] == 90
    assert nf3['alternative_opportunity_pp'] == pytest.approx(.8*100*(56/90-.4))


def test_chapter_weighted_multiple_entries_and_transposition_aggregation(tmp_path):
    g = graph(tmp_path, '1. Nf3 d5 2. g3 Nf6 3. Bg2 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    root, junction = g.roots[0], position('Nf3 d5 g3 Nf6')
    evidence = {position('Nf3'): data(10, 0, 0, [('d7d5', 10, 0, 0)]),
                position('Nf3 d5 g3'): data(10, 0, 0, [('g8f6', 10, 0, 0)]),
                position('g3'): data(10, 0, 0, [('g8f6', 10, 0, 0)]),
                position('g3 Nf6 Nf3'): data(10, 0, 0, [('d7d5', 10, 0, 0)]),
                position('Nf3 d5 g3 Nf6 Bg2'): data(3, 2, 5)}
    m, o, raw, _, v, _ = setup(g, True, evidence, {root: {'g1f3': .7, 'g2g3': .3}})
    for k, node in m.items():
        if node.mode == 'own':
            evidence[k] = data(60, 0, 40, [(b.move, 3, 0, 7) for b in node.branches])
    local = candidates(g, m, raw, v, evidence, True, 30)
    mass = reaches(m, o, raw, {root: 1})
    assert mass[junction] == pytest.approx(1)
    overall = rank_scope(local, mass, {})
    joined = [r for r in overall['rankings']['own'] if r['position'] == junction]
    assert len(joined) == 1
    assert joined[0]['weighted_drag_pp'] == pytest.approx(20)
    selected = [r for r in overall['all_signed_rows'] if r['position'] == root]
    assert sorted(r['weighted_drag_pp'] for r in selected) == pytest.approx([6, 14])
    entries = [position('Nf3'), position('g3'), junction]
    _, first = forward(m, o, raw, {root: 1}, stop_at=entries)
    assert first[junction][0] == 0
    weights = {k: float(x[0]) for k, x in first.items()}
    conditional = rank_scope(local, reaches(m, o, raw, weights), {}, .25)
    join = next(r for r in conditional['rankings']['own'] if r['position'] == junction)
    assert join['study_drag_pp_after_entry'] == pytest.approx(5)
    assert all(r['position'] != root for r in conditional['all_signed_rows'])
    # The second route bypasses the early entry and first enters at the shared descendant.
    _, late = forward(m, o, raw, {root: 1}, stop_at=[position('Nf3'), junction])
    assert late[position('Nf3')][0] == pytest.approx(.7)
    assert late[junction][0] == pytest.approx(.3)
    late_reach = reaches(m, o, raw, {k: float(w[0]) for k, w in late.items()})
    assert late_reach[junction] == pytest.approx(1)


def test_black_perspective_missing_and_sparse_evidence(tmp_path):
    g = graph(tmp_path, '1. e4 e5 *')
    root, e4, leaf = g.roots[0], position('e4'), position('e4 e5')
    e = {root: data(60, 0, 40, [('e2e4', 60, 0, 40)]),
         e4: data(50, 0, 50, [('e7e5', 8, 0, 2), ('c7c5', 42, 0, 48)]), leaf: data(70, 0, 30)}
    m, o, raw, _, v, _ = setup(g, False, e)
    own = next(r for r in candidates(g, m, raw, v, e, False, 30) if r['kind'] == 'own')
    assert own['move_score'] == .3
    assert own['move_database_score'] == .2
    assert own['local_drop_pp'] == pytest.approx(20)
    assert own['sparse']
    assert own['alternative']['san'] == 'c5'
    e[e4] = data(0, 0, 0)
    rows = candidates(g, m, raw, v, e, False, 30)
    scope = rank_scope(rows, reaches(m, o, raw, {root: 1}), {})
    assert len(scope['unresolved_rows']) == 1
    assert not scope['rankings']['own']


def saved_fixture(tmp_path):
    g, m, o, raw, v, e = fixture(tmp_path)
    root, entry = g.roots[0], position('e4 e5')
    cache = tmp_path/'cache'
    cache.mkdir()
    provenance = {k: cache_row(cache, k, d) for k, d in e.items() if m[k].mode != 'own'}
    report = dict(color='white', overall=dict(raw_empirical_score=.4, resolved_contribution=.4, unresolved_mass=0),
                  manifest=dict(input_path=str(tmp_path/'fixture.pgn'),
                                input_sha256=hashlib.sha256((tmp_path/'fixture.pgn').read_bytes()).hexdigest(),
                                configuration={}, root_weights={root: 1}, filters=DEFAULT_FILTERS,
                                evidence=provenance, sparse_threshold=30),
                  chapters=[dict(id='1', name='Test', entries=[dict(position=entry)],
                                 score=dict(entry_probability=.8, raw_empirical_score=.5))])
    path = tmp_path/'white.json'
    path.write_text(json.dumps(report))
    return path, cache, e, m


def test_fetches_only_missing_own_parents_then_runs_fully_offline(tmp_path, monkeypatch):
    path, cache, e, m = saved_fixture(tmp_path)
    original_get = Explorer.get
    network_calls = []
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')

    def guarded_get(self, k):
        if self.offline:
            return original_get(self, k)
        assert m[k].mode == 'own'
        network_calls.append(k)
        self.provenance[k] = cache_row(cache, k, e[k])
        return e[k]

    monkeypatch.setattr(Explorer, 'get', guarded_get)
    with pytest.raises(ValueError, match='own-parent tables missing'):
        analyze(path, cache)
    result = analyze(path, cache, fetch_missing=True)
    assert set(network_calls) == {k for k in m if m[k].mode == 'own'}
    assert result['manifest']['candidate_child_queries'] == 0
    network_calls.clear()
    replay = analyze(path, cache)
    assert not network_calls
    assert replay['overall'] == result['overall']
    chapter_own = replay['chapters'][0]['rankings']['own'][0]
    assert chapter_own['weighted_drag_pp'] == pytest.approx(10)
    assert chapter_own['study_drag_pp_after_entry'] == pytest.approx(8)
    for scope in [replay['overall'], *replay['chapters']]:
        for row in scope['all_signed_rows']:
            board = chess.Board()
            for token in row['line'].split():
                board.push_san(token.split('.')[-1])
            parent = chess.Board(row['position']+' 0 1')
            parent.push_uci(row['move'])
            assert key(parent) == key(board)
    # Prevent silently mixing refreshed evaluation evidence with an old report.
    cached_path = next(cache.glob('*.json'))
    content = json.loads(cached_path.read_text())
    content['retrieved_at'] = 'changed'
    cached_path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='Cached evidence changed since scoring'):
        analyze(path, cache)


def test_changed_source_rejected_before_loading_evidence(tmp_path):
    path, cache, _, _ = saved_fixture(tmp_path)
    (tmp_path/'fixture.pgn').write_text('1. d4 d5 *')
    with pytest.raises(ValueError, match='PGN changed since scoring'):
        analyze(path, cache)


def test_prepared_continuation_can_be_a_strength_despite_bad_historical_move_row(tmp_path):
    g, _, _, _, _, e = fixture(tmp_path)
    e[position('e4 e5 Nf3')] = data(80, 0, 20)
    m, o, raw, _, v, _ = setup(g, True, e)
    scope = rank_scope(candidates(g, m, raw, v, e, True, 30), reaches(m, o, raw, {g.roots[0]: 1}), {})
    nf3 = next(r for r in scope['strengths'] if r['move_san'] == 'Nf3')
    assert nf3['move_database_score'] == .4
    assert nf3['move_score'] == .8
    assert nf3['local_gain_pp'] == pytest.approx(20)
    assert not scope['rankings']['own']


def test_unknown_continuation_never_falls_back_to_historical_own_move_score(tmp_path):
    g, _, _, _, _, e = fixture(tmp_path)
    e[position('e4 e5 Nf3')] = data(0, 0, 0)
    m, o, raw, _, v, _ = setup(g, True, e)
    scope = rank_scope(candidates(g, m, raw, v, e, True, 30), reaches(m, o, raw, {g.roots[0]: 1}), {})
    assert not scope['rankings']['own'] and not scope['strengths']
    for row in scope['all_signed_rows']:
        if row['kind'] == 'own':
            assert row['move_database_score'] is not None
            assert row['move_score'] is None and row['local_drop_pp'] is None


def test_own_ranking_uses_direct_deficit_while_opponent_ranking_uses_weighted_drag():
    from repertoire_score.vulnerabilities import rankings
    rows = [dict(id='rare', kind='own', local_drop_pp=20., weighted_drag_pp=.02),
            dict(id='common', kind='own', local_drop_pp=5., weighted_drag_pp=4.),
            dict(id='reply-rare', kind='opponent', local_drop_pp=20., weighted_drag_pp=.02),
            dict(id='reply-common', kind='opponent', local_drop_pp=5., weighted_drag_pp=4.)]
    result, _ = rankings(rows)
    assert [r['id'] for r in result['own']] == ['rare', 'common']
    assert [r['id'] for r in result['opponent']] == ['reply-common', 'reply-rare']
