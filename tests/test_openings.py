import hashlib
from pathlib import Path

import chess
import httpx
import pytest
from helpers import data, graph, position, run_fixture

from repertoire_score.openings import (
    analyze,
    chapter_sources,
    classify,
    cohort,
    entered_reach,
    first_entries,
    most_common_source,
    name_flow,
    named_regions,
    opening_identity,
)
from repertoire_score.preparation import Evaluator, Evaluators, chess_facts
from repertoire_score.report.links import Chapters
from repertoire_score.report.markdown import table
from repertoire_score.report.pages import opening_details_page
from repertoire_score.report.sections import openings_section
from repertoire_score.sharpness import recursive_wdl


def named(evidence, k, name, eco='A00'):
    evidence[k]['opening'] = dict(name=name, eco=eco)
    return opening_identity(evidence[k]['opening'])


def transposing(tmp_path):
    g = graph(tmp_path, '1. Nf3 d5 2. g3 Nf6 3. Bg2 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    e = {k: data(50, 0, 50, [(m, 50, 0, 50) for m in n.edges]) for k, n in g.nodes.items()}
    e[position('Nf3 d5')] = data(75, 0, 25, [('g2g3', 75, 0, 25)])
    leaf = position('Nf3 d5 g3 Nf6 Bg2')
    e[leaf] = data(10, 20, 70, [('c7c6', 10, 20, 70)])
    policy = {g.roots[0]: {'g1f3': 0.4, 'g2g3': 0.6}}
    ev = Evaluator(g, True, e, chess_facts(g, True, e), policy)
    return g, e, ev


def test_unnamed_transposition_unions_all_routes_and_exact_names_reset(tmp_path):
    g, e, ev = transposing(tmp_path)
    family = named(e, g.roots[0], 'Family')
    a = named(e, position('Nf3 d5'), 'Family: One')
    b = named(e, position('g3 Nf6'), 'Other')
    deeper = named(e, position('Nf3 d5 g3 Nf6 Bg2'), 'Family: One, Deeper')
    catalog, exact, labels, memberships = classify(g, e, ev.facts, True)
    shared = position('Nf3 d5 g3 Nf6')
    assert shared == position('g3 Nf6 Nf3 d5')
    assert labels[shared] == {a, b}
    assert labels[position('Nf3 d5 g3 Nf6 Bg2')] == {deeper}
    assert memberships[shared] == {family, a, b}
    assert memberships[position('Nf3 d5 g3 Nf6 Bg2')] == {family, a, deeper}
    assert b not in catalog[deeper]['parent_ids']
    assert set(exact) == {g.roots[0], position('Nf3 d5'), position('g3 Nf6'), position('Nf3 d5 g3 Nf6 Bg2')}
    # Position labels still merge unused alternative routes, without changing policy.
    ev = Evaluator(g, True, e, ev.facts, {g.roots[0]: 'g1f3'})
    assert classify(g, e, ev.facts, True)[2][shared] == {a, b}


def test_identical_names_with_multiple_eco_codes_form_one_category(tmp_path):
    g, e, ev = transposing(tmp_path)
    a = named(e, position('Nf3 d5'), 'Same opening', 'A00')
    b = named(e, position('g3 Nf6'), 'Same opening', 'A01')
    assert a == b
    catalog, _, labels, memberships = classify(g, e, ev.facts, True)
    assert len(catalog) == 1
    assert catalog[a]['eco_codes'] == ['A00', 'A01']
    assert labels[position('Nf3 d5 g3 Nf6')] == {a}
    region = {k for k, ids in memberships.items() if a in ids}
    _, reach, _ = first_entries(ev, {g.roots[0]: 1.0}, region)
    assert reach == pytest.approx(1.0)


def test_rare_opening_keeps_only_its_incoming_share_at_unnamed_transposition(tmp_path):
    g, e, ev = transposing(tmp_path)
    rare = named(e, position('Nf3 d5'), 'Rare')
    common = named(e, position('g3 Nf6'), 'Common')
    ev = Evaluator(g, True, e, ev.facts, {g.roots[0]: {'g1f3': 0.01, 'g2g3': 0.99}})
    roots = {g.roots[0]: 1.0}
    catalog, exact, _, potential_memberships = classify(g, e, ev.facts, True)
    shared = position('Nf3 d5 g3 Nf6')
    flows = name_flow(ev, roots, exact)
    assert flows[shared] == pytest.approx({rare: 0.01, common: 0.99})
    # This is the old failure: the structural union grabs the common route too.
    potential_region = {k for k, ids in potential_memberships.items() if rare in ids}
    assert first_entries(ev, roots, potential_region)[1] == pytest.approx(1.0)
    entries, total, _ = first_entries(ev, roots, named_regions(exact, catalog)[rare])
    assert total == pytest.approx(0.01)
    assert entered_reach(ev, entries)[shared] == pytest.approx(0.01)
    result = cohort(ev, entries, total, recursive_wdl(ev))
    assert result['entry_baseline']['raw_score'] == pytest.approx(0.75)
    assert result['expected_prepared_moves'] == pytest.approx(2.0)


def test_unclassified_routes_stay_unclassified_at_shared_unnamed_board(tmp_path):
    g, e, ev = transposing(tmp_path)
    rare = named(e, position('Nf3 d5'), 'Rare')
    _, exact, _, _ = classify(g, e, ev.facts, True)
    flows = name_flow(ev, {g.roots[0]: 1.0}, exact)
    assert flows[position('Nf3 d5 g3 Nf6')] == pytest.approx({rare: 0.4, None: 0.6})


def test_named_reset_changes_current_name_but_preserves_prior_origin_mass(tmp_path):
    g, e, ev = transposing(tmp_path)
    rare = named(e, position('Nf3 d5'), 'Rare')
    named(e, position('g3 Nf6'), 'Other')
    leaf = position('Nf3 d5 g3 Nf6 Bg2')
    common = named(e, leaf, 'Common')
    catalog, exact, _, _ = classify(g, e, ev.facts, True)
    roots = {g.roots[0]: 1.0}
    flows = name_flow(ev, roots, exact)
    assert flows[leaf] == pytest.approx({common: 1.0})
    entries, total, _ = first_entries(ev, roots, named_regions(exact, catalog)[rare])
    assert total == pytest.approx(0.4)
    assert entered_reach(ev, entries)[leaf] == pytest.approx(0.4)


def test_named_specific_variation_can_introduce_family_through_bypass(tmp_path):
    g, e, ev = transposing(tmp_path)
    family = named(e, position('Nf3 d5'), 'Family')
    named(e, position('g3 Nf6'), 'Other')
    deeper = named(e, position('Nf3 d5 g3 Nf6'), 'Family: Deeper')
    catalog, exact, _, _ = classify(g, e, ev.facts, True)
    roots = {g.roots[0]: 1.0}
    regions = named_regions(exact, catalog)
    entries, total, _ = first_entries(ev, roots, regions[family])
    assert total == pytest.approx(1.0)
    assert {r['position']: r['mass'] for r in entries} == pytest.approx(
        {position('Nf3 d5'): 0.4, position('Nf3 d5 g3 Nf6'): 0.6}
    )
    assert first_entries(ev, roots, regions[deeper])[1] == pytest.approx(1.0)


def test_unused_alternative_supplies_no_name_probability(tmp_path):
    g, e, ev = transposing(tmp_path)
    selected = named(e, position('Nf3 d5'), 'Selected')
    unused = named(e, position('g3 Nf6'), 'Unused')
    ev = Evaluator(g, True, e, ev.facts, {g.roots[0]: 'g1f3'})
    catalog, exact, _, _ = classify(g, e, ev.facts, True)
    roots = {g.roots[0]: 1.0}
    assert name_flow(ev, roots, exact)[position('Nf3 d5 g3 Nf6')] == {selected: 1.0}
    assert first_entries(ev, roots, named_regions(exact, catalog)[unused])[1] == 0.0


def test_first_entry_absorption_preserves_late_bypass_and_never_counts_twice(tmp_path):
    g, e, ev = transposing(tmp_path)
    roots = {g.roots[0]: 1.0}
    early, shared = position('Nf3 d5'), position('Nf3 d5 g3 Nf6')
    e[position('Nf3')]['moves'][0]['averageRating'] = 1200
    e[position('g3 Nf6 Nf3')]['moves'][0]['averageRating'] = 1900
    entries, total, missed = first_entries(ev, roots, {early, shared})
    assert total == pytest.approx(1.0) and missed == pytest.approx(0.0)
    assert {r['position']: r['mass'] for r in entries} == pytest.approx({early: 0.4, shared: 0.6})
    result = cohort(ev, entries, total, recursive_wdl(ev))
    assert result['repertoire_score'] == pytest.approx(0.2)
    assert result['entry_baseline']['raw_score'] == pytest.approx(0.6)
    assert result['difference_pp'] == pytest.approx(-40.0)
    assert result['outcomes']['sharpness'] == pytest.approx(44.0)
    assert result['expected_prepared_moves'] == pytest.approx(1.4)
    # The same gap reached from both entries is combined before squaring.
    assert result['gap_coverage']['equivalent_gap_reach'] == pytest.approx(1.0)
    assert result['gap_coverage']['distinct_gaps'] == 1
    for row in result['entries']:
        board = chess.Board(row['example']['root_fen'])
        for move in row['example']['path_uci']:
            board.push_uci(move)
        from repertoire_score.graph import key

        assert key(board) == row['position']
        assert row['example']['conditional_probability'] <= row['conditional_weight'] + 1e-10
    late = next(r for r in result['entries'] if r['position'] == shared)
    assert late['example']['path_uci'][0] == 'g2g3'
    assert late['opponent_rating']['mean'] == 1900
    assert next(r for r in result['entries'] if r['position'] == early)['opponent_rating']['mean'] == 1200
    assert 'opponent_rating' not in result  # no rating for an opening group


def test_unprepared_reply_uses_parent_evidence_and_inherited_name(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nf3 *')
    e4, leaf, gap = position('e4'), position('e4 e5 Nf3'), position('e4 c5')
    e = {e4: data(26, 0, 24, [('e7e5', 20, 0, 20), ('c7c5', 6, 0, 4)]), leaf: data(50, 0, 50)}
    e[e4]['moves'][0]['averageRating'] = 1000
    e[e4]['moves'][1]['averageRating'] = 2000
    identity = named(e, e4, "King's Pawn")
    ev = Evaluator(g, True, e, chess_facts(g, True, e))
    assert classify(g, e, ev.facts, True)[2][gap] == {identity}
    entries, total, missed = first_entries(ev, {g.roots[0]: 1.0}, {gap})
    assert total == pytest.approx(0.2) and missed == pytest.approx(0.8)
    result = cohort(ev, entries, total, recursive_wdl(ev))
    assert result['repertoire_score'] == result['entry_baseline']['raw_score'] == pytest.approx(0.6)
    assert result['expected_prepared_moves'] == 0.0
    assert result['gap_coverage']['equivalent_gap_reach'] == 1.0
    assert result['entries'][0]['games_source'] == 'parent move rows'
    assert result['entries'][0]['opponent_rating']['mean'] == 2000
    assert result['entries'][0]['opponent_rating']['difference_vs_parent'] == pytest.approx(800)
    assert gap not in e


def test_black_score_and_unknown_entry_evidence_are_not_filled_in(tmp_path):
    g = graph(tmp_path, '1. e4 c5 2. Nf3 *')
    root, sicilian, leaf = g.roots[0], position('e4 c5'), position('e4 c5 Nf3')
    e = {
        root: data(50, 0, 50, [('e2e4', 50, 0, 50)]),
        sicilian: data(50, 0, 50, [('g1f3', 50, 0, 50)]),
        leaf: data(10, 20, 70),
    }
    ev = Evaluator(g, False, e, chess_facts(g, False, e))
    entries, total, _ = first_entries(ev, {root: 1.0}, {leaf})
    result = cohort(ev, entries, total, recursive_wdl(ev))
    assert result['repertoire_score'] == pytest.approx(0.8)
    assert result['outcomes']['sharpness'] == pytest.approx(44.0)
    # A forced own-move entry does not need local outcome counts for its score.
    entries, total, _ = first_entries(ev, {root: 1.0}, {position('e4')})
    result = cohort(ev, entries, total, recursive_wdl(ev))
    assert result['repertoire_score'] == pytest.approx(0.8)
    assert result['entry_baseline']['raw_score'] is None
    assert result['entry_baseline']['unresolved_mass'] == 1.0
    assert result['difference_pp'] is None


def test_cache_only_analysis_preserves_scores_sources_and_validates_staleness(tmp_path, monkeypatch):
    saved, cache = run_fixture(tmp_path, monkeypatch, common_entry=True)
    path = tmp_path / 'white.json'
    originals = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [path, Path(saved['manifest']['input_path']), *cache.glob('*.json')]
    }
    calls = []
    from repertoire_score.explorer import Explorer

    original = Explorer.get

    def tracked(self, k):
        calls.append(k)
        return original(self, k)

    monkeypatch.setattr(Explorer, 'get', tracked)
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **kw: pytest.fail('No network allowed'))
    result = analyze(path, cache)
    assert result['manifest']['network_requests'] == 0
    assert result['validation']['saved_score_reproduced']
    assert result['openings'] == [] and result['coverage']['ever_classified_probability'] == 0.0
    g = graph(tmp_path, Path(saved['manifest']['input_path']).read_text())
    assert set(calls) == set(g.nodes)
    for p, digest in originals.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    Path(saved['manifest']['input_path']).write_text('changed')
    with pytest.raises(ValueError, match='PGN changed'):
        analyze(path, cache)


def test_opening_tables_show_spread_baselines_and_entry_details(tmp_path):
    from helpers import check_score_tables

    g, e, ev = transposing(tmp_path)
    identity = named(e, position('Nf3 d5'), 'Family: One')
    catalog, exact, labels, memberships = classify(g, e, ev.facts, True)
    region = {k for k, ids in memberships.items() if identity in ids}
    entries, total, _ = first_entries(ev, {g.roots[0]: 1.0}, region)
    row = dict(**catalog[identity], reach=total, chapter_ids=['1'], **cohort(ev, entries, total, recursive_wdl(ev)))
    bundle = dict(
        report=dict(color='white', chapters=g.chapters, manifest=dict(sparse_threshold=30)),
        openings=dict(
            openings=[row],
            catalog=list(catalog.values()),
            coverage=dict(named_repertoire_positions=1, ever_classified_probability=1.0),
            positions={k: dict(exact_name=exact.get(k), current_ids=sorted(ids)) for k, ids in labels.items()},
        ),
    )
    refs = Chapters(bundle['report'])
    table_text = '\n'.join(openings_section(bundle, refs))
    details = '\n'.join(opening_details_page(bundle, refs))
    text = table_text + '\n' + details
    # Per-opening evidence moved to its own page; the report keeps the table and links there.
    assert 'First-entry example' not in table_text and '](@openings/white)' in table_text
    assert details.startswith('# White openings') and '<a id="white-opening-1"></a>' in details
    assert 'Family: One' in text and 'First-entry example' in text
    assert '| Entry baseline | Repertoire score | Delta | Score spread |' in text
    assert ' cp' not in text and '| Score CP |' not in text
    assert 'white-opening-1' in text and '[W1]' in text
    assert 'no aggregate opponent rating' in text
    assert '\u2014' not in text
    check_score_tables(text)


def test_unrelated_classification_is_not_a_parent_and_null_has_no_label(tmp_path):
    g = graph(tmp_path, '1. e4 c5 2. d4 cxd4 3. Nf3 *')
    e = {k: data(50, 0, 50) for k in g.nodes}
    old = named(e, position('e4 c5 d4'), 'Smith-Morra')
    new = named(e, position('e4 c5 d4 cxd4'), 'Open Sicilian')
    result = classify(g, e, chess_facts(g, True, e), True)
    assert result[2][g.roots[0]] == set()
    assert result[2][position('e4 c5 d4 cxd4 Nf3')] == {new}
    assert old not in result[0][new]['parent_ids']


@pytest.mark.parametrize('value', [dict(name='', eco='A00'), dict(name='Opening'), 'Opening'])
def test_malformed_cached_names_are_rejected(value):
    with pytest.raises(ValueError, match='Invalid cached opening'):
        opening_identity(value)


@pytest.mark.parametrize(
    'weights,expected',
    [
        ({'Rare': 0.01, 'Common': 0.99}, dict(id='Common', share=0.99)),
        ({'Named': 0.2, None: 0.8}, dict(id=None, share=0.8)),
        ({'B': 0.5, 'A': 0.5}, dict(id='A', share=0.5)),
        ({None: 0.5, 'Named': 0.5}, dict(id='Named', share=0.5)),
        ({'Only': 0.02}, dict(id='Only', share=1.0)),
        ({}, None),
    ],
)
def test_most_common_source_uses_current_partition_and_deterministic_ties(weights, expected):
    assert most_common_source(weights) == expected


def test_chapter_sources_keep_alternative_policy_and_first_entry_context(tmp_path):
    g, e, ev = transposing(tmp_path)
    a = named(e, position('Nf3 d5'), 'First route')
    b = named(e, position('g3 Nf6'), 'Alternative route')
    shared = position('Nf3 d5 g3 Nf6')
    _, exact, _, _ = classify(g, e, ev.facts, True)
    roots = {g.roots[0]: 1.0}
    saved = dict(
        color='white',
        manifest=dict(root_weights=roots, configuration=dict(policy=ev.policy), sparse_threshold=30),
        chapters=[],
    )
    for chapter in g.chapters:
        own = Evaluator(g, True, e, ev.facts, ev.policy, chapter=chapter['id'])
        value = own.evaluate({shared: 1.0})
        saved['chapters'].append(
            dict(
                id=chapter['id'],
                entries=[dict(position=shared)],
                score=dict(first_entry_weights={shared: 1.0}, resolved_contribution=value[0], unresolved_mass=value[1]),
            )
        )
    sources = {
        r['id']: r
        for r in chapter_sources(Evaluators(g, True, e, ev.facts, ev.policy), saved, exact, name_flow(ev, roots, exact))
    }
    assert sources['overall']['positions'][shared] == dict(id=b, share=0.6)
    for cid, expected in [('1', a), ('2', b)]:
        assert sources[cid]['positions'][shared] == dict(id=expected, share=1.0)
        assert sources[cid]['entry_sources'][shared] == dict(id=expected, share=1.0)


def test_first_entry_name_flow_absorbs_before_later_transposition(tmp_path):
    g, e, ev = transposing(tmp_path)
    a = named(e, position('Nf3 d5'), 'Early')
    b = named(e, position('g3 Nf6'), 'Bypass')
    _, exact, _, _ = classify(g, e, ev.facts, True)
    early, shared = position('Nf3 d5'), position('Nf3 d5 g3 Nf6')
    flow = name_flow(ev, {g.roots[0]: 1.0}, exact, stop_at={early, shared})
    assert flow[early] == {a: 0.4}
    assert flow[shared] == {b: 0.6}


def test_opening_source_move_uses_parent_flow_and_named_child_resets(tmp_path):
    g, _, _ = transposing(tmp_path)
    parent = position('Nf3 d5 g3')
    shared = position('Nf3 d5 g3 Nf6')
    data = dict(
        positions={shared: dict(exact_name=None)},
        openings=[dict(id='First route'), dict(id='Common')],
        source_scopes=[
            dict(
                id='overall',
                positions={parent: dict(id='First route', share=1.0), shared: dict(id='Common', share=0.6)},
                entry_sources={},
            ),
            dict(
                id='1',
                positions={shared: dict(id='First route', share=1.0)},
                entry_sources={shared: dict(id='Entry route', share=0.75)},
            ),
        ],
    )
    refs = Chapters(dict(color='white', chapters=g.chapters), data)
    assert refs.opening_source(dict(position=shared)) == '[Common](#white-opening-2) (60.00%)'
    # A sole source needs no share; partial sources keep theirs.
    assert refs.opening_source(dict(position=parent, move='g8f6')) == '[First route](#white-opening-1)'
    assert refs.for_scope('1').opening_source(dict(position=shared)) == '[First route](#white-opening-1)'
    assert refs.for_scope('1').opening_source(dict(position=shared, _opening_entry=True)) == 'Entry route (75.00%)'
    data['positions'][shared]['exact_name'] = 'Named child'
    assert refs.opening_source(dict(position=parent, move='g8f6')) == 'Named child'
    assert refs.opening_source(dict(position='invalid', move='g8f6')) == 'unavailable'


def test_source_cells_add_opening_column_without_duplicates():
    from repertoire_score.report.markdown import SourceCell

    source = SourceCell('[W1](#white-chapter-1)', 'Vienna (90.00%)')
    rendered = table(['Line', 'Chapter source', 'Reach'], [['e4', source, '10.00%']])
    assert rendered[0] == '| Line | Chapter source | Most common opening source | Reach |'
    assert rendered[2] == '| e4 | [W1](#white-chapter-1) | Vienna (90.00%) | 10.00% |'
    explicit = table(['Line', 'Chapter source', 'Most common opening source'], [['e4', source, 'Already given']])
    assert explicit[0].count('Most common opening source') == 1
    assert 'Most common opening source' not in table(['Opening', 'Chapters'], [['Vienna', source]])[0]
