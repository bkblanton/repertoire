"""Cached opening names, transposition inheritance, and first-entry cohorts."""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import chess
import numpy as np

from .explorer import Explorer, counts
from .gaps import distribution as gap_distribution
from .graph import key, parse
from .model import score
from .preparation import Evaluator, chess_facts
from .ratings import (comparison_fields, comparison_mixture, first_entries as rating_entries,
                      mixture as rating_mixture, reply_rating, response_rating)
from .sharpness import recursive_wdl, stopping_wdl, summarize


def opening_identity(value):
    if not value:
        return None
    if (not isinstance(value, dict) or not isinstance(value.get('name'), str)
            or not value['name'].strip() or not isinstance(value.get('eco'), str)):
        raise ValueError('Invalid cached opening name')
    return value['name']


def classify(graph, evidence, facts, color):
    """Potential labels for annotation, independent of modeled probability.

    A structural union must never be used as a probability region: another
    route into an unnamed board has not thereby played every inherited opening.
    The monotone worklist also handles cycles in unused recorded variations.
    """
    exact, catalog = {}, {}
    edges = {k: set(n.edges.values()) for k, n in graph.nodes.items()}
    for k, node in graph.nodes.items():
        value = evidence.get(k, {}).get('opening')
        identity = opening_identity(value)
        if identity:
            exact[k] = identity
            opening = catalog.setdefault(identity, dict(id=identity, name=value['name'], eco_codes=set()))
            opening['eco_codes'].add(value['eco'])
        if node.edges and facts[k]['turn'] != color and facts[k]['outcome'] is None:
            edges[k].update(facts[k]['possible_targets'])
            # Only the parent's cached move rows are needed for unprepared boards.
            edges[k].update(target for target, _ in facts[k]['after'].values())
    labels = {k: set() for k in sorted(set(edges) | {t for targets in edges.values() for t in targets})}
    pending = deque()
    for k, identity in exact.items():
        labels[k].add(identity)
        pending.append(k)
    while pending:
        k = pending.popleft()
        for target in edges.get(k, ()):
            if target in exact:
                continue
            added = labels[k] - labels[target]
            if added:
                labels[target].update(added)
                pending.append(target)
    # Only cached names with a literal family/variation prefix are parents.
    # An unrelated earlier name is a transition, not a hierarchy relationship.
    for identity, opening in catalog.items():
        opening['eco_codes'] = sorted(opening['eco_codes'])
        opening['eco'] = ', '.join(opening['eco_codes'])
        opening['parent_ids'] = sorted(other for other, parent in catalog.items()
            if opening['name'].startswith(parent['name'] + ': ')
            or opening['name'].startswith(parent['name'] + ', '))
    memberships = {k: ids | {p for identity in ids for p in catalog[identity]['parent_ids']}
                   for k, ids in labels.items()}
    return catalog, exact, labels, memberships


def named_regions(exact, catalog):
    """First activation occurs at an exact name or a named specific descendant.

    Inheriting a name along an unnamed route cannot introduce a new category:
    that route already passed its named entry. Absorbing at named entries is
    therefore equivalent to route-aware first activation, without expanding
    the score graph into histories of previously visited opening categories.
    """
    regions = defaultdict(set)
    for k, identity in exact.items():
        for category in [identity, *catalog[identity]['parent_ids']]:
            regions[category].add(k)
    return regions


def name_flow(evaluator, roots, exact, *, initial=None, stop_at=()):
    """Preserve the last exact name separately on each incoming probability flow."""
    evaluator.evaluate(roots)
    flows = defaultdict(lambda: defaultdict(float))
    stop_at = set(stop_at)
    for k, weight in roots.items():
        if weight:
            incoming = {exact[k]: 1.} if k in exact else (initial or {}).get(k, {None: 1.})
            if not math.isclose(sum(incoming.values()), 1., abs_tol=1e-10):
                raise AssertionError('Initial opening name weights are not normalized')
            for identity, share in incoming.items():
                flows[k][identity] += weight * share
    for k in reversed(evaluator.values):
        if k in stop_at:
            continue
        incoming = list(flows[k].items())
        for identity, mass in incoming:
            for _, p, target in evaluator.edges[k]:
                flows[target][exact.get(target, identity)] += mass * p
            for move, p, kind, _, _ in evaluator.stops[k]:
                if kind == 'deviation':
                    target = evaluator.facts[k]['after'][move][0]
                    flows[target][exact.get(target, identity)] += mass * p
    if not stop_at:
        reach = evaluator.reaches(roots)
        for k, mass in reach.items():
            if not math.isclose(sum(flows[k].values()), mass, abs_tol=1e-10):
                raise AssertionError('Opening name flow did not reproduce canonical position reach')
    return {k: dict(values) for k, values in flows.items() if sum(values.values()) > 0}


def most_common_source(weights):
    """A source is a last-name partition, not overlapping historical families."""
    total = sum(weights.values())
    if not total:
        return None
    identity, mass = min(weights.items(), key=lambda item: (-item[1], item[0] is None, item[0] or ''))
    return dict(id=identity, share=mass / total)


def chapter_sources(graph, saved, evidence, facts, exact, overall_flows):
    scopes = [dict(id='overall', positions={k: most_common_source(v) for k, v in overall_flows.items()}, entry_sources={})]
    manifest = saved['manifest']
    for chapter in saved['chapters']:
        starts = {k: w for k, w in chapter['score'].get('first_entry_weights', {}).items() if w}
        if not starts and len(chapter['entries']) == 1:
            starts = {chapter['entries'][0]['position']: 1.}
        scope = dict(id=chapter['id'], positions={}, entry_sources={})
        if not starts:
            scope['status'] = 'unresolved entry weights'
            scopes.append(scope)
            continue
        evaluator = Evaluator(graph, saved['color'] == 'white', evidence, facts,
            manifest['configuration'].get('policy', {}), chapter=chapter['id'], sparse=manifest['sparse_threshold'])
        entry_flow = name_flow(evaluator, manifest['root_weights'], exact, stop_at=starts)
        initial = {}
        for k in starts:
            incoming = entry_flow.get(k, {})
            total = sum(incoming.values())
            if not total:
                raise AssertionError('Positive chapter entry lacks name flow under its comparison policy')
            initial[k] = {identity: mass / total for identity, mass in incoming.items()}
            scope['entry_sources'][k] = most_common_source(incoming)
        value = evaluator.evaluate(starts)
        if (not math.isclose(value[0], chapter['score']['resolved_contribution'], abs_tol=1e-10)
                or not math.isclose(value[1], chapter['score']['unresolved_mass'], abs_tol=1e-10)):
            raise AssertionError('Opening sources did not reproduce the chapter comparison score')
        flow = name_flow(evaluator, starts, exact, initial=initial)
        scope['positions'] = {k: most_common_source(v) for k, v in flow.items()}
        scope['status'] = 'resolved'
        scopes.append(scope)
    return scopes


def entered_reach(evaluator, entries):
    """Joint probability of reaching each board after first entering this opening.

    Keep this historical origin even after a later named board changes the
    current classification. Incoming first-entry cohorts are disjoint.
    """
    starts, reached = defaultdict(float), defaultdict(float)
    for entry in entries:
        if entry['parent'] is None:
            starts[entry['position']] += entry['mass']
        else:
            reached[entry['position']] += entry['mass']
    if starts:
        regular = evaluator.reaches(starts)
        for k, mass in regular.items():
            reached[k] += mass
            for move, p, kind, _, _ in evaluator.stops[k]:
                if kind == 'deviation':
                    target = evaluator.facts[k]['after'][move][0]
                    reached[target] += mass * p
    return {k: mass for k, mass in reached.items() if mass > 0}


def first_entries(evaluator, roots, region):
    """Absorb on first membership, including cached unprepared reply boards."""
    evaluator.evaluate(roots)
    incoming = defaultdict(float, roots)
    best = {k: (w, k, ()) for k, w in roots.items() if w}
    entries, missed = [], 0.
    for k in reversed(evaluator.values):
        mass = incoming[k]
        if not mass:
            continue
        if k in region:
            entries.append(dict(position=k, mass=mass, parent=None, move=None, sample=None,
                                fixed=None, witness=best[k]))
            continue
        probability, root, path = best[k]
        for move, p, target in evaluator.edges[k]:
            incoming[target] += mass * p
            candidate = probability * p, root, (*path, move)
            if target not in best or candidate[0] > best[target][0]:
                best[target] = candidate
        for move, p, kind, sample, fixed in evaluator.stops[k]:
            target = evaluator.facts[k]['after'][move][0] if kind == 'deviation' else None
            if target in region:
                entries.append(dict(position=target, mass=mass * p, parent=k, move=move,
                                    sample=sample, fixed=fixed,
                                    witness=(probability * p, root, (*path, move))))
            else:
                missed += mass * p
    total = sum(r['mass'] for r in entries)
    if not math.isclose(total + missed, sum(roots.values()), abs_tol=1e-10):
        raise AssertionError('Opening first-entry probability is not conserved')
    _, initial = rating_entries(evaluator, roots, {r['position'] for r in entries if r['parent'] is None})
    for entry in entries:
        k, parent = entry['position'], entry['parent']
        if parent is not None:
            context = reply_rating(evaluator.evidence.get(parent), entry['move'])
            context.update(comparison_fields(context, response_rating(evaluator.evidence.get(parent))))
        else:
            context = initial[k]
            if evaluator.facts[k]['turn'] == evaluator.color:
                paired = []
                for origin in context.get('origins', []):
                    data = evaluator.evidence.get(origin['parent_position'])
                    reply = reply_rating(data, origin['move'])
                    paired.append((origin['weight'], comparison_fields(reply, response_rating(data))))
                context.update(comparison_mixture(paired))
                if k == key(chess.Board()):
                    context['reason'] = 'no_preceding_opponent_move'
        entry['opponent_rating'] = context
    return entries, total, missed


def example(evaluator, witness):
    probability, root, moves = witness
    board = chess.Board(evaluator.graph.nodes[root].fen)
    text = []
    for move in moves:
        text.append(f'{board.fullmove_number}{"." if board.turn else "..."}{board.san(chess.Move.from_uci(move))}')
        board.push_uci(move)
    return dict(root_fen=evaluator.graph.nodes[root].fen, path_uci=list(moves),
                line=' '.join(text) or '(PGN root)', root_probability=probability,
                position=key(board))


def cohort(evaluator, entries, total, wdl):
    """Mix continuations and entry baselines with the same first-arrival weights."""
    outcomes = np.zeros(4)
    baseline_known = baseline_unknown = depth = regular_mass = 0.
    regular_starts, direct_gaps = defaultdict(float), defaultdict(float)
    direct_terminal = 0.
    rows = {}
    for entry in entries:
        k, weight = entry['position'], entry['mass'] / total
        is_reply = entry['parent'] is not None
        if is_reply:
            sample, fixed = entry['sample'], entry['fixed']
            continuation = stopping_wdl(sample, evaluator.color, fixed)
            base = fixed if fixed is not None else score(sample, evaluator.color)
            if fixed is None:
                direct_gaps[k] += weight
            else:
                direct_terminal += weight
        else:
            data = evaluator.evidence.get(k)
            sample = counts(data) if data is not None else [0, 0, 0]
            fixed = evaluator.facts[k]['outcome']
            base = fixed if fixed is not None else score(sample, evaluator.color)
            continuation = wdl[k]
            depth += weight * evaluator.values[k][2]
            regular_starts[k] += weight
            regular_mass += weight
        outcomes += weight * continuation
        if base is None:
            baseline_unknown += weight
        else:
            baseline_known += weight * base
        route = example(evaluator, entry['witness'])
        if route['position'] != k:
            raise AssertionError('Opening entry witness reaches the wrong board')
        row = rows.setdefault(k, dict(position=k, conditional_weight=0., entry_probability=0.,
            example=route, games=0, games_source='parent move rows' if is_reply else 'position',
            baseline_contribution=0., baseline_unresolved_weight=0.,
            repertoire_contribution=0., unresolved_weight=0., outcomes_vector=np.zeros(4), origins=[], chapter_ids=set(), rating_parts=[]))
        row['chapter_ids'].update(evaluator.graph.nodes[entry['parent'] if is_reply else k].chapters)
        row['conditional_weight'] += weight
        row['entry_probability'] += entry['mass']
        row['games'] += sum(sample) if fixed is None else 0
        row['baseline_contribution'] += weight * (base or 0.)
        row['baseline_unresolved_weight'] += weight * (base is None)
        row['repertoire_contribution'] += weight * (continuation[0] + continuation[1] / 2)
        row['unresolved_weight'] += weight * continuation[3]
        row['outcomes_vector'] += weight * continuation
        row['origins'].append(dict(parent=entry['parent'], move=entry['move'], conditional_weight=weight,
                                  counts_white_draw_black=sample, fixed_outcome=fixed))
        row['rating_parts'].append((weight, entry['opponent_rating']))
        if route['root_probability'] > row['example']['root_probability']:
            row['example'] = route
    summary = summarize(outcomes)
    repertoire = summary['resolved_score'] if summary['unresolved_probability'] == 0 else None
    baseline = baseline_known if baseline_unknown == 0 else None
    # Mix canonical gap distributions before taking their Euclidean norm.
    gaps = dict(direct_gaps)
    terminal, unknown = direct_terminal, 0.
    if regular_mass:
        normalized = {k: w / regular_mass for k, w in regular_starts.items()}
        result = gap_distribution(evaluator, normalized)
        for gap in result['gaps']:
            k = gap['position']
            gaps[k] = gaps.get(k, 0.) + regular_mass * gap['reach']
        terminal += regular_mass * result['terminal_mass']
        unknown += regular_mass * result['unresolved_mass']
    gap_mass = sum(gaps.values())
    if not math.isclose(gap_mass + terminal + unknown, 1., abs_tol=1e-9):
        raise AssertionError('Opening gap probability is not conserved')
    collision = sum(p * p for p in gaps.values())
    largest = max(gaps.values(), default=0.)
    bounds = [math.sqrt(collision), math.sqrt(min(1., collision + 2 * largest * unknown + unknown ** 2))]
    for row in rows.values():
        weight = row['conditional_weight']
        row['chapter_ids'] = [c['id'] for c in evaluator.graph.chapters if c['id'] in row['chapter_ids']]
        row['baseline_score'] = row['baseline_contribution'] / weight if row['baseline_unresolved_weight'] == 0 else None
        row['repertoire_score'] = row['repertoire_contribution'] / weight if row['unresolved_weight'] == 0 else None
        row['outcomes'] = summarize(row.pop('outcomes_vector') / weight)
        row['example']['conditional_probability'] = row['example']['root_probability'] / total
        parts = row.pop('rating_parts')
        context = rating_mixture(parts, 'first-entry incoming opponent moves')
        if all(r['basis'] == 'current opponent response rows' for _, r in parts):
            context['basis'] = 'current opponent response rows'
        else:
            context.update(comparison_mixture([(p, {field: r.get(field, 0. if 'coverage' in field else None)
                for field in ('parent_mean', 'parent_known_coverage', 'difference_vs_parent', 'comparison_coverage')})
                for p, r in parts]))
        if any(r.get('reason') == 'no_preceding_opponent_move' for _, r in parts):
            context['reason'] = 'no_preceding_opponent_move'
        row['opponent_rating'] = context
    return dict(repertoire_score=repertoire, outcomes=summary,
        entry_baseline=dict(raw_score=baseline, unresolved_mass=baseline_unknown,
                            conditional_bounds=[baseline_known, baseline_known + baseline_unknown]),
        difference_pp=100 * (repertoire - baseline) if repertoire is not None and baseline is not None else None,
        expected_prepared_moves=float(depth),
        gap_coverage=dict(equivalent_gap_reach=bounds[0] if unknown == 0 else None,
                          equivalent_gap_reach_bounds=bounds, unresolved_mass=unknown,
                          distinct_gaps=len(gaps), gap_mass=gap_mass, terminal_mass=terminal),
        entries=sorted(rows.values(), key=lambda r: (-r['conditional_weight'], r['position'])),
        validation=dict(first_entry_weights_sum=float(sum(r['conditional_weight'] for r in rows.values())),
                        gap_probability_conserved=True, canonical_gaps_merged_before_squaring=True))


def analyze(path, cache='.cache/explorer'):
    path = Path(path)
    source_bytes = path.read_bytes()
    saved = json.loads(source_bytes)
    manifest = saved['manifest']
    source = Path(manifest['input_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('PGN changed since scoring; regenerate scores first')
    graph = parse(source, manifest['configuration'].get('exclude', []))
    color = saved['color'] == 'white'
    explorer = Explorer(cache, manifest['filters'], offline=True)
    evidence, missing = {}, []
    try:
        for k in graph.nodes:
            try:
                evidence[k] = explorer.get(k)
            except ValueError as exc:
                if not str(exc).startswith('Offline cache miss:'):
                    raise
                missing.append(k)
            if k in manifest['evidence'] and explorer.provenance.get(k) != manifest['evidence'][k]:
                raise ValueError('Saved score evidence changed; regenerate scores first')
    finally:
        explorer.close()
    facts = chess_facts(graph, color, evidence)
    evaluator = Evaluator(graph, color, evidence, facts, manifest['configuration'].get('policy', {}),
                          sparse=manifest['sparse_threshold'])
    roots = manifest['root_weights']
    value = evaluator.evaluate(roots)
    for actual, expected in [(value[0], saved['overall']['resolved_contribution']),
                             (value[1], saved['overall']['unresolved_mass'])]:
        if not math.isclose(actual, expected, abs_tol=1e-10):
            raise AssertionError('Opening analysis did not reproduce the saved score')
    wdl, reach = recursive_wdl(evaluator), evaluator.reaches(roots)
    catalog, exact, potential, _ = classify(graph, evidence, facts, color)
    regions = named_regions(exact, catalog)
    flows = name_flow(evaluator, roots, exact)
    positions = {}
    for k in sorted(set(potential) | set(flows)):
        incoming = flows.get(k, {})
        mass = sum(incoming.values())
        active = {identity: weight for identity, weight in incoming.items() if identity is not None}
        membership = defaultdict(float)
        for identity, weight in active.items():
            for category in [identity, *catalog[identity]['parent_ids']]:
                membership[category] += weight
        positions[k] = dict(exact_name=exact.get(k), cached_opening=evidence.get(k, {}).get('opening'),
            potential_ids=sorted(potential.get(k, ())),
            current_ids=[exact[k]] if k in exact else sorted(active), membership_ids=sorted(membership),
            reach=mass, current_name_reach=active,
            current_name_weights={identity: weight / mass for identity, weight in active.items()} if mass else {},
            unclassified_reach=incoming.get(None, 0.), membership_reach=dict(membership),
            opening_reach_contributions={}, opening_reach_fractions={})
    rows, unreachable = [], []
    for identity, opening in catalog.items():
        entries, total, missed = first_entries(evaluator, roots, regions[identity])
        if not total:
            unreachable.append(identity)
            continue
        for k, mass in entered_reach(evaluator, entries).items():
            board = positions[k]
            if mass > board['reach'] + 1e-10 or mass > total + 1e-10:
                raise AssertionError('Opening origin exceeds actual board or opening reach')
            board['opening_reach_contributions'][identity] = mass
            board['opening_reach_fractions'][identity] = min(1., mass / board['reach'])
        chapter_ids = set()
        for k in regions[identity]:
            if reach.get(k, 0.) > 0:
                chapter_ids.update(graph.nodes[k].chapters)
        for entry in entries:
            if entry['parent']:
                chapter_ids.update(graph.nodes[entry['parent']].chapters)
        rows.append(dict(**opening, reach=total, not_reached_probability=missed,
                         chapter_ids=[c['id'] for c in graph.chapters if c['id'] in chapter_ids],
                         **cohort(evaluator, entries, total, wdl)))
    _, classified_reach, _ = first_entries(evaluator, roots, set(exact))
    sources = chapter_sources(graph, saved, evidence, facts, exact, flows)
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('PGN changed during opening analysis')
    return dict(color=saved['color'], openings=sorted(rows, key=lambda r: (-r['reach'], r['name'], r['eco'])),
        catalog=sorted(catalog.values(), key=lambda r: (r['name'], r['eco'])),
        positions=positions, source_scopes=sources,
        coverage=dict(ever_classified_probability=classified_reach, unreachable_opening_ids=sorted(unreachable),
                      named_repertoire_positions=len(exact), inherited_positions=sum(bool(p['current_ids']) and not p['exact_name'] for p in positions.values()),
                      multi_name_positions=sum(len(p['current_ids']) > 1 for p in positions.values())),
        manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
            report_path=str(path.resolve()), report_sha256=hashlib.sha256(source_bytes).hexdigest(),
            input_path=str(source), input_sha256=manifest['input_sha256'], filters=manifest['filters'],
            schema_version=2, cache_only=True, network_requests=0, evidence=explorer.provenance,
            source_schema_version=1,
            uncached_positions=missing, source_pgn_unchanged=True, policy_basis='overall selected repertoire policy',
            naming_rule='Exact cached names replace the current name on every arriving route. Unnamed boards preserve each incoming name and its probability share under the selected policy. Structural potential labels never add probability. Unprepared replies inherit parent name flows without child queries.',
            parent_rule='Only existing cached names that match at colon or comma boundaries are broader parents. Earlier unrelated labels are not parents.',
            reach_rule='First arrival at a cached exact name or a known more specific named descendant. Inheritance preserves existing route probability and never introduces another opening. Each modeled game counts once per opening; rows overlap. Per-board origin contributions retain the mass that previously entered each opening, even after a later name reset.',
            score_rule='Recursive repertoire WDL at regular entries; parent move-row WDL at unprepared entries. Baselines, depth and gap distributions use the same normalized first-entry weights.'),
        validation=dict(saved_score_reproduced=True, no_unprepared_child_queries=True,
                        first_entry_probability_conserved=True, name_flow_conserved=True,
                        origin_contributions_bounded_by_board_reach=True, chapter_source_scores_reproduced=True,
                        source_pgn_unchanged=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--cache', default='.cache/explorer')
    args = parser.parse_args()
    for path in args.reports:
        result = analyze(path, args.cache)
        path.with_suffix('.openings.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f"{result['color']}: {len(result['openings'])} reached openings; network requests: 0", flush=True)
    from .render import update_report_outputs
    update_report_outputs(args.reports[-1])


if __name__ == '__main__':
    main()
