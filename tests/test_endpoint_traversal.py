import pytest
from helpers import data, graph, position, setup

from repertoire.character import scope_metrics
from repertoire.evaluate import KNOWN, forward
from repertoire.graph import reachable, resolve
from repertoire.openings import classify, name_flow
from repertoire.preparation import Evaluator, chess_facts, position_lines
from repertoire.routes import depth_distribution


def test_cached_endpoint_replies_merge_transposed_gaps_and_scores(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nc3 Nf6 3. g3 d5 4. exd5 *\n\n1. e4 e5 2. Nc3 d6 3. g3 *')
    early = position('e4 e5 Nc3 Nf6 g3')
    endpoint = position('e4 e5 Nc3 d6 g3')
    gap = position('e4 e5 Nc3 Nf6 g3 d6')
    e = {
        position('e4'): data(50, 0, 50, [('e7e5', 50, 0, 50)]),
        position('e4 e5 Nc3'): data(50, 0, 50, [('g8f6', 30, 0, 30), ('d7d6', 20, 0, 20)]),
        early: data(50, 0, 50, [('d7d5', 30, 0, 20), ('d7d6', 20, 0, 30)]),
        endpoint: data(35, 0, 65, [('g8f6', 30, 0, 45), ('f8e7', 5, 0, 20)]),
        position('e4 e5 Nc3 Nf6 g3 d5 exd5'): data(20, 0, 80, [('d8d5', 20, 0, 80)]),
    }
    m, order, raw, _, values, _ = setup(g, True, e)
    ev = Evaluator(g, True, e, chess_facts(g, True, e))
    root = {g.roots[0]: 1.0}
    result = scope_metrics(ev, root, position_lines(g))
    row = next(r for r in result['positions'] if r['position'] == gap)
    assert gap not in e and gap not in g.nodes
    assert row['reach'] == pytest.approx(0.6 * 0.5 + 0.4 * 0.75)
    assert row['database_score'] == pytest.approx(0.4)
    assert {r['parent_position'] for r in row['unprepared_origins']} == {early, endpoint}
    assert sum(r['reach'] for r in row['unprepared_origins']) == pytest.approx(row['reach'])
    gaps = {r['position']: r['reach'] for r in result['gap_coverage']['gaps']}
    assert gaps[gap] == pytest.approx(row['reach'])
    assert sum(gaps.values()) == pytest.approx(1.0)
    assert result['validation']['unanswered_reach_matches_first_gaps']
    assert values[g.roots[0]][KNOWN] == pytest.approx(0.32)
    assert ev.values[g.roots[0]][0] == pytest.approx(0.32)
    assert sum(forward(m, order, raw, root)[0].values()) == pytest.approx(1.0)
    opportunities = result['predictability']['positions']
    assert next(r for r in opportunities if r['position'] == endpoint)['reach'] == pytest.approx(0.4)


@pytest.mark.parametrize('color', [True, False])
def test_cached_endpoint_transposition_resumes_preparation_and_chapter_entry(tmp_path, color):
    if color:
        text = '1. Nf3 d5 2. g3 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *'
        endpoint = position('Nf3 d5 g3')
        target = position('Nf3 d5 g3 Nf6')
        final = position('Nf3 d5 g3 Nf6 Bg2')
        reply, last_reply = 'g8f6', 'c7c5'
        policy = {position(''): {'g1f3': 0.4, 'g2g3': 0.6}}
    else:
        text = '1. d4 d5 2. c4 e6 *\n\n1. Nf3 d5 2. d4 e6 3. c4 Nf6 *'
        endpoint = position('d4 d5 c4 e6')
        target = position('d4 d5 c4 e6 Nf3')
        final = position('d4 d5 c4 e6 Nf3 Nf6')
        reply, last_reply = 'g1f3', 'b1c3'
        policy = {}
    g = graph(tmp_path, text)
    e = {}
    for k, n in g.nodes.items():
        moves = list(n.edges)
        e[k] = data(50 * max(1, len(moves)), 0, 50 * max(1, len(moves)), [(move, 50, 0, 50) for move in moves])
    e[endpoint] = data(90, 0, 10, [(reply, 90, 0, 10)])
    e[final] = data(20, 0, 80, [(last_reply, 20, 0, 80)])
    assert not g.nodes[endpoint].edges
    t = resolve(g, color, policy)
    assert t[endpoint][reply] == (target, None)
    assert target in reachable(t, g.roots)
    m, order, raw, _, values, _ = setup(g, color, e, policy)
    ev = Evaluator(g, color, e, chess_facts(g, color, e), policy)
    roots = {g.roots[0]: 1.0}
    result = scope_metrics(ev, roots, position_lines(g))
    expected = 0.2 if color else 0.8
    assert values[g.roots[0]][KNOWN] == pytest.approx(expected)
    assert ev.values[g.roots[0]][0] == pytest.approx(expected)
    assert ev.reaches(roots)[target] == pytest.approx(1.0)
    _, entries = forward(m, order, raw, roots, stop_at={target})
    assert entries[target] == pytest.approx(1.0)
    assert result['gap_coverage']['gap_mass'] == pytest.approx(1.0)
    assert result['reuse']['expected_encounters_per_game'] == pytest.approx(3.0)
    assert result['validation']['unanswered_reach_matches_first_gaps']

    # An endpoint's cached opening label must follow its implicit reply edge.
    names = {endpoint: dict(name='Endpoint opening', eco='A00')}
    _, exact, labels, _ = classify(g, names, ev.facts, color)
    assert 'Endpoint opening' in labels[target]
    named_mass = 0.4 if color else 0.5
    assert name_flow(ev, roots, exact)[target]['Endpoint opening'] == pytest.approx(named_mass)

    # Missing endpoint reply counts cannot conceal a possible prepared reentry.
    missing = dict(e, **{endpoint: data(0, 0, 0)})
    missing_ev = Evaluator(g, color, missing, chess_facts(g, color, missing), policy)
    depth = depth_distribution(missing_ev, roots)
    assert depth['expected_moves'] is None
    assert depth['expected_bounds'] == pytest.approx([3.0 - named_mass, 3.0])
