import hashlib
import json
import math

import chess
import httpx
import pytest

from repertoire_score.consolidated import generate, load
from repertoire_score.preparation import Evaluator, chess_facts
from repertoire_score.ratings import (Context, analyze, first_entries, mixture, move_context,
                                      reply_rating, response_rating, unavailable,
                                      comparison_fields, add_reply_differences, refresh_differences)
from test_model import data, graph, position
from test_chapter_policies import run_fixture


def rated(w, d, b, moves=()):
    result = data(w, d, b, [r[:4] for r in moves])
    for row, values in zip(result['moves'], moves):
        row['averageRating'] = values[4]
    return result


def test_reply_mean_count_weights_missing_ratings_and_residual_games():
    rows = rated(100, 0, 100, [('e7e5', 50, 0, 50, 2000), ('c7c5', 20, 0, 40, 1000),
                              ('e7e6', 10, 0, 10, None)])
    mean = response_rating(rows)
    assert mean['mean'] == 1625
    assert mean['known_coverage'] == .8 and mean['missing_coverage'] == pytest.approx(.2)
    assert mean['rated_observations'] == 160
    assert reply_rating(rows, 'e7e5')['mean'] == 2000
    assert reply_rating(rows, 'e7e6')['mean'] is None
    assert reply_rating(rows, 'd7d5')['mean'] is None
    for value in (None, 0, -1, True, float('nan'), float('inf'), '1800'):
        assert response_rating(rated(1, 0, 0, [('e7e5', 1, 0, 0, value)]))['mean'] is None
    assert response_rating(data(0, 0, 0))['mean'] is None
    assert response_rating(data(100, 0, 0))['missing_coverage'] == 1
    assert reply_rating(rated(0, 0, 0, [('e7e5', 0, 0, 0, 1900)]), 'e7e5')['mean'] is None


def test_reply_rating_difference_uses_same_player_parent_mean_and_discloses_coverage():
    from repertoire_score.consolidated import rating_difference
    table = rated(50, 0, 50, [('e7e5', 30, 0, 30, 1000), ('c7c5', 10, 0, 10, 2000),
                             ('e7e6', 10, 0, 10, None)])
    parent = response_rating(table)
    lower = comparison_fields(reply_rating(table, 'e7e5'), parent)
    higher = comparison_fields(reply_rating(table, 'c7c5'), parent)
    assert parent['mean'] == 1250
    assert lower['difference_vs_parent'] == -250
    assert higher['difference_vs_parent'] == 750
    assert lower['parent_known_coverage'] == .8
    assert rating_difference(lower) == '-250 (parent 80.0% rated)'
    assert rating_difference(higher) == '+750 (parent 80.0% rated)'
    assert .75 * lower['difference_vs_parent'] + .25 * higher['difference_vs_parent'] == 0
    missing = comparison_fields(reply_rating(table, 'e7e6'), parent)
    assert missing['difference_vs_parent'] is None and missing['comparison_coverage'] == 0
    assert rating_difference(missing) == 'unavailable'
    assert comparison_fields(reply_rating(table, 'e7e5'), unavailable())['difference_vs_parent'] is None
    assert rating_difference(response_rating(table)) == 'n/a'  # current-position response average is not a specific reply


@pytest.mark.parametrize('color,path,own_move,opponent_move', [
    (chess.WHITE, '1. e4 e5 2. Nf3 *', 'e2e4', 'e7e5'),
    (chess.BLACK, '1. e4 e5 2. Nf3 *', 'e7e5', 'e2e4')])
def test_move_maker_rule_for_both_colors_and_no_child_queries(tmp_path, color, path, own_move, opponent_move):
    g = graph(tmp_path, path)
    root, e4, e5, end = [position(p) for p in ('', 'e4', 'e4 e5', 'e4 e5 Nf3')]
    evidence = {root: rated(50, 0, 50, [('e2e4', 50, 0, 50, 900 if color else 2100)]),
                e4: rated(50, 0, 50, [('e7e5', 50, 0, 50, 2100 if color else 900)]),
                e5: rated(50, 0, 50, [('g1f3', 50, 0, 50, 900 if color else 2200)]),
                end: rated(50, 0, 50, [('b8c6', 50, 0, 50, 2300 if color else 900)])}
    e = Evaluator(g, color, evidence, chess_facts(g, color, evidence))
    context = Context(e, {root: 1})
    own_position = root if color else e4
    opponent_position = e4 if color else root
    assert context.local[own_position]['mean'] == (None if color else 2100)
    assert context.local[opponent_position]['mean'] == 2100
    assert move_context(evidence, color, own_position, own_move)['mean'] == (2100 if color else 2200)
    assert move_context(evidence, color, opponent_position, opponent_move)['mean'] == 2100
    if color:
        assert context.local[e5]['mean'] == 2100
        assert context.score_evidence()['mean'] == 2300
    else:
        assert context.score_evidence()['mean'] == 2200  # our turn at leaf: preceding White Nf3
    missing_child = 'c7c5' if color else 'd2d4'
    evidence[opponent_position]['moves'].append(dict(uci=missing_child, white=0, draws=0, black=1, averageRating=2400))
    assert move_context(evidence, color, opponent_position, missing_child)['mean'] == 2400
    assert len(evidence) == 4
    assert move_context(evidence, color, own_position, 'd2d4' if color else 'c7c5')['mean'] is None


def test_starting_board_local_context_keeps_opponent_response_average_in_the_report(tmp_path):
    from repertoire_score.consolidated import character_section
    from repertoire_score.ratings import attach
    from repertoire_score.consolidated import Chapters, opponent_rating
    g = graph(tmp_path, '1. e4 c5 *')
    root, e4, leaf = [position(p) for p in ('', 'e4', 'e4 c5')]
    evidence = {root: rated(100, 0, 0, [('e2e4', 75, 0, 0, 1600), ('d2d4', 25, 0, 0, 2000)]),
                e4: rated(100, 0, 0, [('c7c5', 100, 0, 0, 9999)]),
                leaf: rated(100, 0, 0, [('g1f3', 100, 0, 0, 1800)])}
    black = Context(Evaluator(g, chess.BLACK, evidence, chess_facts(g, chess.BLACK, evidence)), {root: 1}).inventory()
    assert black['positions'][root]['local']['mean'] == 1700
    assert 'continuation' not in black['positions'][root]
    assert 'score_evidence' not in black
    row = dict(position=root, line='(PGN root)', reach=1., effective_replies=2.,
               recorded_reply_observations=100, sparse=False, replies=[])
    scope = dict(id='overall', reuse=dict(reachable_distinct_decisions=0, curve=[], decisions=[]),
                 predictability=dict(effective_replies=2., recorded_reply_coverage=1.,
                                     sparse_recorded_opportunity_fraction=0., positions=[row]),
                 position_profiles={'all': dict(effective_pawn_structures=1., distributions={}, features={})})
    bundle = dict(report=dict(color='black', chapters=[], events=[]), character=dict(scopes=[scope]),
                  ratings=dict(scopes=[dict(id='overall', **black)]))
    attach(bundle)
    assert row['opponent_rating']['mean'] == 1700
    rendered = '\n'.join(character_section(scope, Chapters(bundle['report']), 1))
    assert '| [(PGN root)](https://lichess.org/analysis/standard/' in rendered and '| 1,700 |' in rendered
    white = Context(Evaluator(g, chess.WHITE, evidence, chess_facts(g, chess.WHITE, evidence)), {root: 1}).inventory()
    assert white['positions'][root]['local']['mean'] is None
    assert opponent_rating(white['positions'][root]['local']) == 'n/a (no preceding opponent move)'


def transposed_context(tmp_path, duplicate=False):
    text = '1. Nf3 d5 2. g3 Nf6 *\n\n1. g3 Nf6 2. Nf3 d5 *'
    g = graph(tmp_path, text + ('\n\n' + text if duplicate else ''))
    evidence = {}
    for k, node in g.nodes.items():
        rows = []
        board = chess.Board(node.fen)
        for move in node.edges:
            n = 10000 if move == 'd7d5' else 100
            value = 1000 if move == 'g8f6' else 2000
            if board.turn == chess.WHITE: value = 9999
            # The final opponent move differs across transposed arrivals.
            if k == position('Nf3 d5 g3'): value = 1000
            rows.append((move, n // 2, 0, n // 2, value))
        total = sum(r[1] + r[3] for r in rows) or 100
        evidence[k] = rated(total // 2, 0, total // 2, rows)
    root = g.roots[0]
    policy = {root: {'g1f3': .75, 'g2g3': .25}}
    e = Evaluator(g, chess.WHITE, evidence, chess_facts(g, chess.WHITE, evidence), policy)
    return e, root


def test_transposed_arrivals_use_policy_flow_not_game_counts_and_duplicates(tmp_path):
    e, root = transposed_context(tmp_path)
    final = position('Nf3 d5 g3 Nf6')
    original = Context(e, {root: 1})
    assert original.local[final]['mean'] == 1250  # .75 * 1000 + .25 * 2000
    assert original.score_evidence()['mean'] == 1250
    e2, root2 = transposed_context(tmp_path, duplicate=True)
    repeated = Context(e2, {root2: 1})
    assert repeated.local[final] == original.local[final]
    assert repeated.reach[final] == 1


def test_transposed_reply_differences_mix_paired_parent_comparisons_by_arrival(tmp_path):
    e, root = transposed_context(tmp_path)
    a, b = position('Nf3 d5 g3'), position('g3 Nf6 Nf3')
    e.evidence[a] = rated(100, 0, 100, [('g8f6', 50, 0, 50, 1000), ('e7e5', 50, 0, 50, 3000)])
    e.evidence[b] = rated(10000, 0, 10000, [('d7d5', 5000, 0, 5000, 2000), ('c7c5', 5000, 0, 5000, 1000)])
    e.facts = chess_facts(e.graph, chess.WHITE, e.evidence)
    ctx = Context(e, {root: 1})
    scope = dict(id='overall', moves={}, **ctx.inventory())
    result = dict(color='white', scopes=[scope], manifest={}, validation={})
    before = json.loads(json.dumps(result))
    add_reply_differences(result, e.evidence)
    final = position('Nf3 d5 g3 Nf6')
    local = scope['positions'][final]['local']
    assert local['mean'] == 1250
    assert local['parent_mean'] == 1875
    assert local['difference_vs_parent'] == -625  # .75*(-1000) + .25*(500), not count weights
    assert local['comparison_coverage'] == 1
    # Enrichment leaves all existing means, probabilities, and stopping contexts intact.
    for k, fields in before['scopes'][0]['positions'].items():
        for name, value in fields['local'].items():
            assert scope['positions'][k]['local'][name] == value
        assert scope['positions'][k].get('continuation') == fields.get('continuation')
    assert 'difference_vs_parent' not in scope


def test_transposed_reply_difference_partial_arrival_coverage(tmp_path):
    e, root = transposed_context(tmp_path)
    a, b = position('Nf3 d5 g3'), position('g3 Nf6 Nf3')
    e.evidence[a] = rated(100, 0, 100, [('g8f6', 50, 0, 50, 1000), ('e7e5', 50, 0, 50, 3000)])
    e.evidence[b] = rated(100, 0, 100, [('d7d5', 50, 0, 50, None), ('c7c5', 50, 0, 50, 1500)])
    e.facts = chess_facts(e.graph, chess.WHITE, e.evidence)
    scope = dict(id='overall', moves={}, **Context(e, {root: 1}).inventory())
    add_reply_differences(dict(color='white', scopes=[scope], manifest={}, validation={}), e.evidence)
    row = scope['positions'][position('Nf3 d5 g3 Nf6')]['local']
    assert row['difference_vs_parent'] == -1000
    assert row['comparison_coverage'] == .75 and row['known_coverage'] == .75


def test_multiple_first_entries_separate_initial_and_later_arrivals(tmp_path):
    e, root = transposed_context(tmp_path)
    early, final = position('Nf3 d5'), position('Nf3 d5 g3 Nf6')
    weights, initial = first_entries(e, {root: 1}, {early, final})
    assert weights == {early: .75, final: .25}
    assert initial[final]['mean'] == 2000  # earlier 1000 route has already entered
    ctx = Context(e, weights, initial)
    assert ctx.local[final]['mean'] == 1250  # combine later arrivals for local position
    assert ctx.score_evidence()['mean'] == 1250
    direct = mixture([(ctx.reach[k] * p, ctx.stop(k, move, kind))
                      for k in ctx.reach for move, p, kind, _, _ in e.stops[k]], 'enumeration')
    assert direct['weighted_rating_sum'] == ctx.score_evidence()['weighted_rating_sum']


def test_stopping_mixture_does_not_weight_every_visited_node(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nf3 Nc6 *')
    root, e4, e5, nf3, leaf = [position(p) for p in ('', 'e4', 'e4 e5', 'e4 e5 Nf3', 'e4 e5 Nf3 Nc6')]
    evidence = {root: rated(50, 0, 50, [('e2e4', 50, 0, 50, 9999)]),
                e4: rated(50, 0, 50, [('e7e5', 25, 0, 25, 1500), ('c7c5', 15, 0, 15, 2000)]),
                e5: rated(50, 0, 50, [('g1f3', 50, 0, 50, 9999)]),
                nf3: rated(50, 0, 50, [('b8c6', 50, 0, 50, 2400)]), leaf: data(50, 0, 50)}
    ctx = Context(Evaluator(g, chess.WHITE, evidence, chess_facts(g, chess.WHITE, evidence)), {root: 1})
    result = ctx.score_evidence()
    assert result['known_coverage'] == pytest.approx(.8)  # unnamed residual 20%
    assert result['mean'] == pytest.approx((.5 * 2400 + .3 * 2000) / .8)
    assert position('e4 c5') not in evidence
    assert ctx.inventory()['positions'][position('e4 c5')]['local']['mean'] == 2000
    assert result['known_coverage'] + result['missing_coverage'] == 1


def test_cache_only_ledger_preserves_scores_and_has_no_overall_mean(tmp_path, monkeypatch):
    from repertoire_score.preparation import analyze as preparation
    from repertoire_score.character import analyze as character
    from repertoire_score.vulnerabilities import analyze as vulnerabilities
    saved, cache = run_fixture(tmp_path, monkeypatch, common_entry=True, multiple_entries=True)
    path = tmp_path / 'white.json'
    for family, fn in [('preparation', preparation), ('character', character), ('vulnerabilities', vulnerabilities)]:
        path.with_suffix(f'.{family}.json').write_text(json.dumps(fn(path, cache)), encoding='utf-8')
    originals = {p: p.read_bytes() for p in [path, *tmp_path.glob('white.*.json'), tmp_path / 'fixture.pgn', *cache.glob('*.json')]}
    result = analyze(path, cache)
    overall = result['scopes'][0]
    assert 'score_evidence' not in overall and 'entry_baseline' not in overall
    assert overall['positions'][position('')]['local']['mean'] is None
    assert overall['positions'][position('')]['local']['reason'] == 'no_preceding_opponent_move'
    assert 'continuation' not in overall['positions'][position('')]
    assert not any(k in result for k in ('mean', 'known_coverage', 'opponent_rating'))
    assert result['manifest']['network_requests'] == 0
    assert result['validation']['first_entry_weights_reproduced']
    assert result['validation']['stopping_rating_moments_reproduced']
    assert all(p.read_bytes() == value for p, value in originals.items())
    for scope in result['scopes'][1:]:
        assert scope['score_evidence']['mean'] is None
        assert scope['score_evidence']['missing_coverage'] == 1
    path.with_suffix('.ratings.json').write_text(json.dumps(result), encoding='utf-8')
    refreshed = refresh_differences(path, cache)
    assert refreshed['manifest']['report_sha256'] == result['manifest']['report_sha256']
    assert refreshed['manifest']['cache_sha256'] == result['manifest']['cache_sha256']
    assert refreshed['validation']['reply_differences_use_cached_parents']
    source = tmp_path / 'fixture.pgn'
    source_bytes = source.read_bytes()
    try:
        source.write_bytes(source_bytes + b'\n{New source edits outside the saved snapshot}\n')
        snapshot = refresh_differences(path, cache)
        assert not snapshot['manifest']['reply_difference_source_current']
        assert snapshot['manifest']['input_sha256'] == result['manifest']['input_sha256']
        assert snapshot['scopes'] == refreshed['scopes']
    finally:
        source.write_bytes(source_bytes)
    provenance = next(iter(result['manifest']['evidence'].values()))
    cache_file = cache / (provenance['cache_key'] + '.json')
    cache_bytes = cache_file.read_bytes()
    try:
        modified = json.loads(cache_bytes)
        modified['data']['extra'] = 'a cache change with unchanged retrieval time'
        cache_file.write_text(json.dumps(modified), encoding='utf-8')
        with pytest.raises(ValueError, match='Cached rating evidence changed'):
            refresh_differences(path, cache)
    finally:
        cache_file.write_bytes(cache_bytes)
    assert all(p.read_bytes() == value for p, value in originals.items())
    generate([path])
    full = (tmp_path / 'report.md').read_text(encoding='utf-8')
    # Skip the contents list, which names the opponent-rating correlation section.
    overview = full.split('<a id="overview"></a>', 1)[1].split('## White repertoire')[0]
    color_totals = full.split('### Score and evidence limits')[1].split('### Uncertainty priorities')[0]
    assert 'opponent rating' not in overview.lower() + color_totals.lower()
    chapters = '\n'.join(p.read_text(encoding='utf-8') for p in (tmp_path / 'chapters').glob('W*.md'))
    assert 'Avg opponent rating' in full and 'entry-baseline opponent rating' in chapters
    assert '\u2014' not in full + chapters
    assert 'Rating Δ vs parent' in full
    changed = path.with_suffix('.character.json')
    changed.write_bytes(changed.read_bytes() + b'\n')
    before = (tmp_path / 'report.md').read_bytes()
    with pytest.raises(ValueError, match='supporting analysis changed'):
        generate([path])
    assert (tmp_path / 'report.md').read_bytes() == before
    assert 'ratings' in load([path], strict=False)[0]['unavailable']




def test_rated_alternative_chapters_keep_their_own_policy_and_entry_baseline(tmp_path, monkeypatch):
    from repertoire_score.preparation import analyze as preparation
    from repertoire_score.character import analyze as character
    from repertoire_score.vulnerabilities import analyze as vulnerabilities
    saved, cache = run_fixture(tmp_path, monkeypatch, common_entry=True)
    leaves = {position('e4 e6 d4 d5 e5 c5 c3'): 1400,
              position('e4 e6 d4 d5 Nd2 Nf6 e5'): 2400,
              position('e4 e6 d4 d5 Nd2 c5 exd5'): 2000}
    for path in cache.glob('*.json'):
        item = json.loads(path.read_bytes())
        fen = item['identity']['query']['fen']
        k = ' '.join(fen.split()[:4]); board = chess.Board(fen)
        table = item['data']
        for row in table['moves']:
            row['averageRating'] = 9999 if board.turn else 1800
        if k in leaves:
            table['moves'] = [dict(uci=next(iter(board.legal_moves)).uci(),
                                    white=table['white'], draws=table['draws'], black=table['black'],
                                    averageRating=leaves[k])]
        path.write_text(json.dumps(item), encoding='utf-8')
    path = tmp_path / 'white.json'
    original = path.read_bytes()
    for family, fn in [('preparation', preparation), ('character', character), ('vulnerabilities', vulnerabilities)]:
        path.with_suffix(f'.{family}.json').write_text(json.dumps(fn(path, cache)), encoding='utf-8')
    result = analyze(path, cache)
    scopes = {s['id']: s for s in result['scopes']}
    assert scopes['advance']['score_evidence']['mean'] == 1400
    assert scopes['tarrasch']['score_evidence']['mean'] == pytest.approx(2240)
    assert scopes['split']['score_evidence']['mean'] == pytest.approx(2240)
    for sid in ('advance', 'tarrasch', 'split'):
        assert scopes[sid]['entry_baseline']['mean'] == 1800
        assert scopes[sid]['entry_baseline']['known_coverage'] == 1
    assert path.read_bytes() == original

