import json
from types import SimpleNamespace

import chess
import httpx
import pytest
from helpers import cache_row, data, position

from repertoire import score
from repertoire.evaluate import dominators
from repertoire.graph import parse
from repertoire.vulnerabilities import analyze

# Two chapters meet after 1.e4 e5 2.Nf3 Nc6 by different move orders, and 1...d5 ends at a theory leaf.
PGN = """[ChapterURL "https://lichess.org/study/test/open"]
[ChapterName "Open"]

1. e4 e5 (1... d5) 2. Nf3 Nc6 3. Bc4 *

[ChapterURL "https://lichess.org/study/test/knight"]
[ChapterName "Knight"]

1. e4 Nc6 2. Nf3 e5 3. Bc4 *
"""


def table_for(k, edges):
    """A consistent table: every recorded reply has its own results, plus one unprepared reply."""
    board = chess.Board(k + ' 0 1')
    if board.turn == chess.WHITE:
        return data(55, 10, 35)
    # Results differ from table to table, so the two move orders into the shared position score differently.
    shift = sum(i * ord(c) for i, c in enumerate(k)) % 11
    rows = [(move, 20 + 7 * i + shift, 4, 16 - 3 * i) for i, move in enumerate(sorted(edges))]
    spare = next(m.uci() for m in sorted(board.legal_moves, key=lambda m: m.uci()) if m.uci() not in edges)
    rows.append((spare, 9, 2, 9))
    return data(*(sum(r[i] for r in rows) for i in (1, 2, 3)), rows)


@pytest.fixture
def scored(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('no network'))
    pgn = tmp_path / 'white.pgn'
    pgn.write_text(PGN, encoding='utf-8')
    cache = tmp_path / 'cache'
    cache.mkdir()
    graph = parse(pgn)
    for k, node in graph.nodes.items():
        cache_row(cache, k, table_for(k, node.edges))
    # The theory leaf's own table counts every move order, unlike the 1...d5 row that leads to it.
    cache_row(cache, position('e4 d5'), data(70, 0, 30))
    args = SimpleNamespace(
        config=None,
        pgn=str(pgn),
        color='white',
        output=str(tmp_path / 'white'),
        command='run',
        cache=str(cache),
        offline=True,
        refresh=False,
        prior=[0.5] * 3,
        sparse_threshold=30,
        tolerance=1,
    )
    score.analyze(args)
    path = tmp_path / 'white.json'
    return path, cache, json.loads(path.read_text())


def test_edge_ledger_adds_up_to_the_delta(scored):
    path, cache, saved = scored
    result = analyze(path, cache)
    ledger = result['overall']['edge']
    assert ledger['status'] == 'resolved'
    assert ledger['delta_pp'] == pytest.approx(result['overall_delta_pp'], abs=1e-12)
    parts = ledger['decisions_pp'] + ledger['move_orders_pp'] + ledger['theory_leaves_pp'] + ledger['finished_games_pp']
    assert parts == pytest.approx(ledger['delta_pp'], abs=1e-12)
    assert ledger['move_orders_pp'] != pytest.approx(0, abs=1e-6)
    assert ledger['theory_leaves_pp'] != pytest.approx(0, abs=1e-6)
    rows = [r for r in result['overall']['all_signed_rows'] if r['kind'] == 'own']
    assert sum(r['edge_pp'] for r in rows) == pytest.approx(ledger['decisions_pp'], abs=1e-12)
    for row in rows:
        gain = 100 * (row['move_database_score'] - row['reference_score'])
        assert row['edge_pp'] == pytest.approx(row['branch_reach'] * gain, abs=1e-12)
    assert sum(g['edge_pp'] for g in ledger['by_move_number']) == pytest.approx(ledger['decisions_pp'], abs=1e-12)
    assert result['validation']['edge_ledger_reproduces_delta']


def test_transpositions_list_each_move_order_with_its_share(scored):
    path, cache, _ = scored
    ledger = analyze(path, cache)['overall']['edge']
    meeting = next(t for t in ledger['transpositions'] if t['position'] == position('e4 e5 Nf3 Nc6'))
    assert {a['move'] for a in meeting['arrivals']} == {'b8c6', 'e7e5'}
    effect = sum(a['reach'] * (meeting['database_score'] - a['score']) for a in meeting['arrivals'])
    assert 100 * effect == pytest.approx(meeting['effect_pp'], abs=1e-12)
    assert sum(t['effect_pp'] for t in ledger['transpositions']) == pytest.approx(ledger['move_orders_pp'], abs=1e-12)


def test_chapter_ledgers_add_up_to_the_entry_baseline_delta(scored):
    path, cache, saved = scored
    result = analyze(path, cache)
    deltas = {c['id']: c['entry_baseline']['difference_pp'] for c in saved['chapters']}
    for chapter in result['chapters']:
        ledger = chapter['edge']
        assert ledger['status'] == 'resolved'
        assert ledger['delta_pp'] == pytest.approx(deltas[chapter['id']], abs=1e-12)
        parts = sum(ledger[k] for k in ('decisions_pp', 'move_orders_pp', 'theory_leaves_pp', 'finished_games_pp'))
        assert parts == pytest.approx(ledger['delta_pp'], abs=1e-12)


def test_dominators_follow_every_route():
    # a -> b -> d and a -> c -> d: d is dominated by a only; e after d is dominated by d.
    model = {
        'a': SimpleNamespace(branches=[SimpleNamespace(target='b'), SimpleNamespace(target='c')]),
        'b': SimpleNamespace(branches=[SimpleNamespace(target='d')]),
        'c': SimpleNamespace(branches=[SimpleNamespace(target='d')]),
        'd': SimpleNamespace(branches=[SimpleNamespace(target='e'), SimpleNamespace(target='x')]),
        'e': SimpleNamespace(branches=[]),
        'x': SimpleNamespace(branches=[]),
    }
    sampled = {k: [(0.5, None)] * len(n.branches) for k, n in model.items()}
    sampled['d'] = [(1.0, None), (0.0, None)]
    order = ['x', 'e', 'd', 'c', 'b', 'a']
    idom = dominators(model, order, sampled, {'a': 1.0})
    assert idom == {'a': None, 'b': 'a', 'c': 'a', 'd': 'a', 'e': 'd'}


def test_effort_ranks_disjoint_pruning_lines_and_rare_moves(scored):
    path, cache, saved = scored
    result = analyze(path, cache)
    rows = {r['id']: r for r in result['overall']['all_signed_rows'] if r['kind'] == 'own'}
    effort = result['overall']['effort']
    root = next(r for r in rows.values() if r['position'] == position(''))
    assert root['decisions_dropped'] == len(rows)
    for row in rows.values():
        assert row['drop_value_pp'] == pytest.approx(row['branch_reach'] * row['local_gain_pp'])
        expected = row['drop_value_pp'] * (1 - row['branch_reach']) ** effort['recent_games']
        assert row['review_priority_pp'] == pytest.approx(expected)
    ranked = [rows[i] for i in effort['review']]
    assert ranked == sorted(ranked, key=lambda r: -r['review_priority_pp'])
    pruned = [rows[i] for i in effort['pruning']]
    assert all(r['decisions_dropped'] >= effort['prune_min_decisions'] for r in pruned)
    assert effort['decisions'] == len(rows)
    counts = {c['id']: c['decisions'] for c in effort['chapters']}
    for chapter in saved['chapters']:
        recorded = [r for r in rows.values() if chapter['id'] in r['chapter_attribution']['source_ids']]
        assert counts[chapter['id']] == len(recorded) > 0
