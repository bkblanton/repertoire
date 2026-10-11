"""Repetition loops: unrolled up to the third occurrence of a position, which is a draw."""

import json
import math
import re
from types import SimpleNamespace

import chess
import httpx
import numpy as np
import pytest
from helpers import cache_row, data, graph, position, setup
from test_consolidated import check_links

from repertoire import score as cli
from repertoire.board_cache import position_of
from repertoire.character import analyze as character
from repertoire.engine import analyze as engine
from repertoire.evaluate import COMPLETED, KNOWN, backward, forward
from repertoire.graph import loops, policy_transitions, resolve, topology, unroll
from repertoire.openings import analyze as openings
from repertoire.preparation import Evaluator, chess_facts
from repertoire.preparation import analyze as preparation
from repertoire.ratings import analyze as ratings
from repertoire.report.generate import generate
from repertoire.report_insights import analyze as report_insights
from repertoire.vulnerabilities import analyze as vulnerabilities

# White plays 1.Nf3 and 2.Ng1. After 2...Ng8 the game is back at the start.
SHUFFLE = '1. Nf3 Nf6 2. Ng1 *'


def shuffle_evidence(knight, other, back, stay):
    """Tables after 1.Nf3 (2...Nf6 or d5) and after 2.Ng1 (2...Ng8 or e5), as white/draw/black counts."""
    return {
        position('Nf3'): data(*(a + b for a, b in zip(knight, other)), [('g8f6', *knight), ('d7d5', *other)]),
        position('Nf3 Nf6 Ng1'): data(*(a + b for a, b in zip(back, stay)), [('f6g8', *back), ('e7e5', *stay)]),
    }


def owner_score(counts):
    return (counts[0] + counts[1] / 2) / sum(counts)


def test_loops_are_unrolled_to_the_third_occurrence(tmp_path):
    g = graph(tmp_path, SHUFFLE)
    start = g.roots[0]
    positional = policy_transitions(g, True, {})
    assert loops(positional) == [sorted([start, position('Nf3'), position('Nf3 Nf6'), position('Nf3 Nf6 Ng1')])]
    transitions = resolve(g, True, {})
    # Follow the loop moves only: start, Nf3, Nf6, Ng1, Ng8, twice, then the third start is a draw.
    path: list[str] = []
    node: str | None = start
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
        path.append(node)
        node = transitions[node][move][0]
    assert node is None
    assert [position_of(k) for k in path] == [start, position('Nf3'), position('Nf3 Nf6'), position('Nf3 Nf6 Ng1')] * 2
    assert path[0] == start and all('#' in k for k in path[1:])
    assert len(set(path)) == 8
    # Without a loop nothing changes.
    plain = policy_transitions(graph(tmp_path, '1. e4 e5 2. Nf3 *'), True, {})
    assert unroll(plain) is plain


def test_unrolled_moves_follow_the_threefold_rule_from_every_entry():
    # Two loops sharing b and c, entered from outside at e and from any loop position.
    positional = {
        'a': {'1': ('b', 1.0)},
        'b': {'2': ('c', None), '3': ('a', None)},
        'c': {'4': ('a', 1.0), '5': ('d', 1.0), '6': ('x', 1.0)},
        'd': {'7': ('b', None)},
        'e': {'8': ('c', None)},
        'x': {},
    }
    transitions = unroll(positional)
    walked = 0

    def walk(node, counts):
        nonlocal walked
        position = position_of(node)
        for move, (target, _) in positional[position].items():
            unrolled, _ = transitions[node][move]
            seen = counts.get(target, 0) + 1
            walked += 1
            if seen == 3:
                assert unrolled is None, (node, move)
                continue
            assert unrolled is not None and position_of(unrolled) == target, (node, move, unrolled)
            walk(unrolled, {**counts, target: seen})

    for start in positional:
        walk(start, {start: 1})
    assert walked > 100
    assert topology(transitions, list(positional))


def test_repetition_scores_a_draw_after_two_passes(tmp_path):
    g = graph(tmp_path, SHUFFLE)
    knight, other, back, stay = (30, 20, 10), (10, 10, 20), (20, 0, 20), (30, 10, 20)
    model, order, raw, _, values, _ = setup(g, True, shuffle_evidence(knight, other, back, stay))
    a, b = sum(knight) / 100, sum(back) / 100
    s1, s2 = owner_score(other), owner_score(stay)
    second = a * (b * 0.5 + (1 - b) * s2) + (1 - a) * s1
    first = a * (b * second + (1 - b) * s2) + (1 - a) * s1
    root = g.roots[0]
    assert values[root][KNOWN] == pytest.approx(first)
    stops, _ = forward(model, order, raw, {root: 1.0})
    assert sum(stops.values()) == pytest.approx(1.0)
    draws = {(k, j): p for (k, j), p in stops.items() if model[k].branches[j].kind == 'repetition'}
    assert list(draws.values()) == pytest.approx([(a * b) ** 2])
    (k, j), _ = next(iter(draws.items()))
    assert position_of(k) == position('Nf3 Nf6 Ng1') and model[k].branches[j].fixed_score == 0.5


def test_variance_adds_each_tables_influence_over_both_passes(tmp_path):
    g = graph(tmp_path, SHUFFLE)
    model, order, _, posterior, _, _ = setup(g, True, shuffle_evidence((8, 4, 4), (1, 1, 2), (6, 2, 4), (2, 1, 1)))
    rng, n = np.random.default_rng(7), 60000
    draws = {}
    for k, alpha in posterior.tables.items():
        theta = rng.dirichlet(alpha.ravel(), size=n).reshape(n, *alpha.shape)
        draws[k] = theta.sum(axis=2), (theta @ posterior.owner) / theta.sum(axis=2)
    # Every occurrence of a position uses the same draw of its table.
    sample = {}
    for k in order:
        node = model[k]
        if position_of(k) not in draws:
            sample[k] = posterior.sample[k]
            continue
        rows, scores = draws[position_of(k)]
        sample[k] = (
            [(1.0, scores[:, 0])]
            if node.mode == 'stop'
            else [
                (rows[:, j], b.fixed_score if b.fixed_score is not None else scores[:, j])
                for j, b in enumerate(node.branches)
            ]
        )
    root = g.roots[0]
    simulated = backward(model, order, sample, 30, n)[root][COMPLETED]
    assert posterior.values[root][COMPLETED] == pytest.approx(simulated.mean(), abs=0.003)
    variance = posterior.value_variance({root: 1.0})
    assert variance == pytest.approx(simulated.var(), rel=0.1)
    separate = sum(c * c * posterior.table_variance(m) for m, c in posterior.combined_influence({root: 1.0}).items())
    assert abs(separate - simulated.var()) > abs(variance - simulated.var())


def test_entry_inside_a_loop_is_evaluated_where_games_arrive(tmp_path):
    g = graph(tmp_path, SHUFFLE)
    evidence = shuffle_evidence((30, 20, 10), (10, 10, 20), (20, 0, 20), (30, 10, 20))
    entry, root = position('Nf3 Nf6'), g.roots[0]
    facts = chess_facts(g, True, evidence)
    fresh = Evaluator(g, True, evidence, facts).evaluate({entry: 1.0})
    arrived = Evaluator(g, True, evidence, facts, roots={root: 1.0}).evaluate({entry: 1.0})
    # Games reach 2.Ng1 having played the start and 1.Nf3 once: one pass fewer before the draw than a fresh start.
    model, order, raw, _, values, _ = setup(g, True, evidence)
    _, entries = forward(model, order, raw, {root: 1.0}, stop_at=[entry])
    (node, mass), *others = [(k, m) for k, m in entries.items() if m > 0]
    assert not others and node != entry and mass == pytest.approx(0.6)
    assert arrived[0] == pytest.approx(values[node][KNOWN])
    assert fresh[0] != pytest.approx(arrived[0])


VIENNA = (
    '[ChapterURL "https://lichess.org/study/test/main"]\n[ChapterName "Main"]\n\n1. e4 e5 2. Nc3 Nf6 3. f4 *\n\n'
    '[ChapterURL "https://lichess.org/study/test/sidelines"]\n[ChapterName "Sidelines"]\n\n'
    '1. e4 e5 2. Nc3 Nc6 (2... Qf6 3. Nd5 Qd8 4. Nc3) 3. g3 *'
)


def score_vienna(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A fully cached run must not use the network')

    monkeypatch.setattr(httpx.Client, 'request', forbidden)
    g = graph(tmp_path, VIENNA)
    cache = tmp_path / 'cache'
    cache.mkdir(exist_ok=True)
    for k, n in g.nodes.items():
        board = chess.Board(n.fen)
        if board.turn == chess.WHITE and not n.edges or not n.edges and board.turn == chess.BLACK:
            table = data(55, 10, 35)
        else:
            # Recorded moves, plus one unprepared legal move.
            moves = list(n.edges) + [next(m.uci() for m in board.legal_moves if m.uci() not in n.edges)]
            rows = [(m, 20 + 7 * i, 5, 15 + 3 * i) for i, m in enumerate(moves)]
            table = data(*(sum(r[i] for r in rows) for i in (1, 2, 3)), rows)
        cache_row(cache, k, table)
    cache_row(cache, position(''), data(50, 10, 40))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({}))
    args = SimpleNamespace(
        config=str(config),
        pgn=str(tmp_path / 'fixture.pgn'),
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
    cli.analyze(args)
    return tmp_path / 'white.json', cache


def test_every_stage_and_report_handle_a_repetition_loop(tmp_path, monkeypatch):
    path, cache = score_vienna(tmp_path, monkeypatch)
    report = json.loads(path.read_text())
    repetitions = [e for e in report['events'] if e['type'] == 'repetition' and e['probability'] > 0]
    assert len(repetitions) == 1
    draw = repetitions[0]
    assert draw['raw_score'] == 0.5 and draw['probability'] > 0
    assert draw['representative_path_san'] == ['e4', 'e5', 'Nc3', 'Qf6', 'Nd5', 'Qd8', 'Nc3']
    assert sum(e['probability'] for e in report['events']) == pytest.approx(1.0)
    assert all('#' not in e['parent_position'] for e in report['events'])
    sidelines = next(c for c in report['chapters'] if c['id'] == 'sidelines')
    assert position('e4 e5 Nc3 Qf6') in {e['position'] for e in sidelines['entries']}
    # Each stage reproduces the saved scores, so this also checks they agree on where games enter the loop.
    for name, analyze in [
        ('vulnerabilities', vulnerabilities),
        ('preparation', preparation),
        ('character', character),
        ('ratings', ratings),
        ('openings', openings),
        ('insights', report_insights),
        ('engine', engine),
    ]:
        result = analyze(path, cache)
        path.with_suffix(f'.{name}.json').write_text(json.dumps(result))
    moves = json.loads(path.with_suffix('.vulnerabilities.json').read_text())['overall']['all_signed_rows']
    second = [r for r in moves if r.get('node') and r['move'] == 'd8f6']
    assert len(second) == 1 and second[0]['position'] == position('e4 e5 Nc3')
    assert second[0]['line'].endswith('2...Qf6 3.Nd5 3...Qd8 4.Nc3 4...Qf6')
    repeat = next(r for r in moves if r['move'] == 'd5c3' and r['move_score'] == 0.5)
    assert repeat['score_basis'] == 'terminal result'
    preparation_scopes = json.loads(path.with_suffix('.preparation.json').read_text())['scopes']
    stops = next(s for s in preparation_scopes if s['id'] == 'overall')['stops']
    assert [s['reach'] for s in stops if s['type'] == 'repetition'] == pytest.approx([draw['probability']])
    assert math.isclose(sum(s['reach'] for s in stops), 1.0)
    assert all('#' not in r['target'] for r in moves if r['target'])
    generate([path])
    # History nodes stay inside the analyses: pages show and link positions only.
    for page in check_links(path.parent):
        assert not re.search(r'[ _](?:-|[a-h][36])#[0-2]+', page.read_text(encoding='utf-8')), page
