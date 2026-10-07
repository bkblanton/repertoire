import chess
import pytest
from helpers import data, graph, position, setup

from repertoire.baseline import chapter_entry_baseline
from repertoire.evaluate import chapter_score, forward
from repertoire.graph import chapter_positions, infer_entries, resolve
from repertoire.model import Branch, ModelNode
from repertoire.score import chapter_entries, inspect_repertoire
from repertoire.transitions import chapter_transitions, hitting_bounds

VIENNA = """[ChapterURL "https://lichess.org/study/test/quiet"]

1. e4 e5 2. Nc3 Nf6 3. g3 Nc6 4. Bg2 Bc5 5. Nge2 *

[ChapterURL "https://lichess.org/study/test/late"]

1. e4 e5 2. Nc3 Nf6 3. g3 Bc5 4. Bg2 Nc6 5. Nge2 d6 6. h3 *

[ChapterURL "https://lichess.org/study/test/paulsen"]

1. e4 e5 2. Nc3 Nc6 3. g3 Nf6 *"""


def automatic(g, color=True):
    return {cid: found['positions'] for cid, found in infer_entries(g, color).items()}


def test_reach_counts_only_entry_boards_from_any_move_order(tmp_path):
    g = graph(tmp_path, VIENNA)
    early = position('e4 e5 Nc3 Nf6 g3 Nc6')
    late = position('e4 e5 Nc3 Nf6 g3 Bc5 Bg2 Nc6')
    # Paulsen hands 3...Nf6 to the quiet chapter by ending its line there.
    assert automatic(g) == {
        'quiet': [early],
        'late': [position('e4 e5 Nc3 Nf6 g3 Bc5')],
        'paulsen': [position('e4 e5 Nc3 Nc6')],
    }
    # The quiet chapter prepares the shared later board, but reaching it is not reaching the chapter.
    prepared = chapter_positions(g, 'quiet', [early])
    assert late in prepared and position('e4 e5 Nc3 Nf6 g3') not in prepared
    t = resolve(g, True, {})
    evidence = {}
    for k, node in g.nodes.items():
        if not t[k]:
            evidence[k] = data(80, 0, 20)
        elif not chess.Board(node.fen).turn:
            moves = [(move, 50, 0, 50) for move in t[k]]
            evidence[k] = data(50 * len(moves), 0, 50 * len(moves), moves)
    evidence[position('e4')] = data(50, 0, 50, [('e7e5', 40, 0, 40), ('c7c5', 10, 0, 10)])
    evidence[position('e4 e5 Nc3')] = data(50, 0, 50, [('g8f6', 25, 0, 25), ('b8c6', 25, 0, 25)])
    evidence[position('e4 e5 Nc3 Nf6 g3')] = data(
        50, 0, 50, [('b8c6', 10, 0, 15), ('f8c5', 25, 0, 25), ('d7d5', 15, 0, 10)]
    )
    evidence[position('e4 e5 Nc3 Nf6 g3 Bc5 Bg2')] = data(50, 0, 50, [('b8c6', 20, 0, 20), ('d7d6', 30, 0, 30)])
    evidence[early] = data(60, 0, 40)
    evidence[late] = data(20, 0, 80)
    m, o, r, s, v, p = setup(g, True, evidence)
    result = chapter_score(m, o, r, v, {g.roots[0]: 1}, [early], s)
    # Both move orders reach the entry board: 0.8 * 0.5 * 0.25 + 0.8 * 0.5. The later 3...Bc5 4.Bg2 Nc6
    # transposition into the quiet chapter's lines belongs to the late chapter alone.
    assert result['entry_probability'] == pytest.approx(0.5)
    assert result['first_entry_weights'] == {early: pytest.approx(1)}
    assert result['raw_empirical_score'] == pytest.approx(0.8)
    assert chapter_score(m, o, r, v, {g.roots[0]: 1}, automatic(g)['late'], s)['entry_probability'] == pytest.approx(
        0.2
    )
    baseline = chapter_entry_baseline([early], result, evidence, True, {})
    assert baseline['raw_score'] == pytest.approx(0.6)
    assert sum(forward(m, o, r, {g.roots[0]: 1})[0].values()) == pytest.approx(1)


def test_transposing_chapter_keeps_its_own_entry(tmp_path):
    # Like the Alekhine 1...Nf6 2.Nc3 e5: the transposition does not make the Vienna board an Alekhine entry.
    g = graph(tmp_path, '1. e4 e5 2. Nc3 Nf6 3. g3 *\n\n1. e4 Nf6 2. Nc3 e5 3. g3 *')
    assert automatic(g) == {'1': [position('e4 e5')], '2': [position('e4 Nf6')]}


def test_parent_chapter_keeps_the_branches_it_does_not_hand_off(tmp_path):
    g = graph(
        tmp_path,
        '1. e4 e5 2. Nc3 Nf6 3. g3 d5 (3... Nc6) (3... Bc5 4. Bg2) *\n\n1. e4 e5 2. Nc3 Nf6 3. g3 Nc6 4. Bg2 Bc5 *',
    )
    assert automatic(g) == {
        '1': sorted([position('e4 e5 Nc3 Nf6 g3 d5'), position('e4 e5 Nc3 Nf6 g3 Bc5')]),
        '2': [position('e4 e5 Nc3 Nf6 g3 Nc6')],
    }


def test_entries_follow_the_chapter_policy_and_unique_leaves(tmp_path):
    # Only the first own move is followed; a line ending on a board no other chapter reaches is an entry.
    g = graph(tmp_path, '1. e4 e5 (1... c5) *\n\n1. e4 (1. d4 d5) e6 *')
    assert automatic(g) == {'1': sorted([position('e4 e5'), position('e4 c5')]), '2': [position('e4 e6')]}
    # For Black, 1.d4 is the opponent's move, so the d4 line is followed too.
    assert automatic(g, False)['2'] == sorted([position('e4 e6'), position('d4')])


def test_configured_entries_override_and_are_validated(tmp_path):
    g = graph(tmp_path, VIENNA)
    inspection = inspect_repertoire(g, True)
    subtree = position('e4 e5 Nc3 Nf6 g3 Nc6 Bg2 Bc5')
    entries, status, prepared = chapter_entries(
        g, True, {'entries': {'quiet': [{'path': ['e4', 'e5', 'Nc3', 'Nf6', 'g3', 'Nc6', 'Bg2', 'Bc5']}]}}, inspection
    )
    assert entries['quiet'] == [subtree] and status['quiet'] == 'explicit configuration'
    assert entries['late'] == [position('e4 e5 Nc3 Nf6 g3 Bc5')] and status['late'].startswith('Automatic')
    assert prepared['quiet'] == [subtree, position('e4 e5 Nc3 Nf6 g3 Nc6 Bg2 Bc5 Nge2')]
    with pytest.raises(ValueError, match='not in repertoire'):
        chapter_entries(g, True, {'entries': {'paulsen': [{'path': ['d4']}]}}, inspection)
    with pytest.raises(ValueError, match='unknown chapter IDs'):
        chapter_entries(g, True, {'entries': {'missing': [{'path': ['e4']}]}}, inspection)
    with pytest.raises(ValueError, match='chapter_regions is no longer supported'):
        chapter_entries(g, True, {'chapter_regions': {}}, inspection)


def test_transition_direction_and_simultaneous_entry():
    # Both histories visit A and B, but their order differs. An intersection
    # calculation would incorrectly report 100% in both directions.
    m = {
        'root': ModelNode('own', [Branch(target='a1'), Branch(target='b1')]),
        'a1': ModelNode('own', [Branch(target='b2')]),
        'b1': ModelNode('own', [Branch(target='a2')]),
        'a2': ModelNode('stop', [Branch(kind='theory_leaf')]),
        'b2': ModelNode('stop', [Branch(kind='theory_leaf')]),
    }
    order = ['b2', 'a1', 'a2', 'b1', 'root']
    sampled = {k: [(1.0, None)] for k in m}
    sampled['root'] = [(0.5, None), (0.5, None)]
    chapters = [
        {'id': cid, 'name': cid, 'score': {'entry_probability': 1.0, 'first_entry_weights': weights}}
        for cid, weights in [('A', {'a1': 0.5, 'a2': 0.5}), ('B', {'b1': 0.5, 'b2': 0.5})]
    ]
    destinations = {'A': {'a1', 'a2'}, 'B': {'b1', 'b2'}}
    rows = chapter_transitions(m, order, sampled, chapters, destinations)
    assert all(row['conditional_probability'] == 0.5 for row in rows)
    destinations['B'].add('a2')
    row = next(row for row in chapter_transitions(m, order, sampled, chapters, destinations) if row['source_id'] == 'A')
    assert row['conditional_probability'] == 1.0


def test_transition_missing_distribution_remains_unresolved():
    m = {
        'a': ModelNode('stop', [Branch(kind='unresolved_distribution')], potential_targets=['b']),
        'b': ModelNode('stop', [Branch(kind='theory_leaf')]),
    }
    sampled = {'a': [(1.0, None)], 'b': [(1.0, 0.5)]}
    assert hitting_bounds(m, ['b', 'a'], sampled, {'b'})['a'] == (0.0, 1.0)
    assert hitting_bounds(m, ['b', 'a'], sampled, {'a'})['a'] == (1.0, 1.0)
