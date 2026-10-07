"""Rank repertoire vulnerabilities using parent move tables and saved model evidence."""

import json

import numpy as np

from .attribution import enrich
from .board_cache import fen_number, move_text, route_line, san
from .context import DEFAULT_CACHE, AnalysisContext, stage_main
from .evaluate import KNOWN, UNKNOWN, backward, best_routes, can_enter, forward, reaches
from .explorer import Explorer, add_token_option, apply_token_file, collect, counts
from .graph import resolve, topology
from .model import empirical, prepare, score
from .status import Status


def representative_lines(graph, model, order, sampled, roots, prefixes=None):
    """One legal policy route for labels, as (move text, full-move number at the position); probabilities always
    use all routes. Numbers count along the route itself, since transposed routes can differ in length.
    `prefixes` continues earlier lines to the roots."""
    paths = {}
    for k, (_, root, moves) in best_routes(model, order, sampled, roots).items():
        prefix, number = (prefixes or {}).get(root, ('', fen_number(graph.nodes[root].fen)))
        text, position, number = route_line(root, number, moves)
        if position != k:
            raise AssertionError('Representative route does not reach its position')
        paths[k] = (prefix + ' ' + text).strip(), number
    return paths


def candidates(graph, model, sampled, values, evidence, color, sparse_threshold):
    """Local signed score changes. This pure function cannot request child evidence."""
    result = []
    for k, node in model.items():
        if node.mode == 'stop':
            continue
        number = fen_number(graph.nodes[k].fen)
        parent_data = evidence[k]
        parent_n = sum(counts(parent_data))
        rows = {r['uci']: r for r in parent_data['moves']}
        parent_score = score(counts(parent_data), color)
        for b, (p, empirical_score) in zip(node.branches, sampled[k]):
            if not b.move or p <= 0:
                continue
            row = rows.get(b.move)
            n = sum(counts(row)) if row else 0
            common = dict(
                id=f'{k}|{b.move}',
                position=k,
                move=b.move,
                move_san=san(k, b.move),
                move_label=move_text(k, number, b.move),
                branch_probability=float(p),
                sample_count=n,
                parent_sample_count=parent_n,
                counts_white_draw_black=counts(row) if row else [0, 0, 0],
                sparse=n < sparse_threshold,
                chapters=sorted(graph.nodes[k].chapters),
                target=b.target,
            )
            if node.mode == 'opponent':
                before = float(values[k][KNOWN]) if values[k][UNKNOWN] == 0 else None
                if b.target is not None:
                    after = float(values[b.target][KNOWN]) if values[b.target][UNKNOWN] == 0 else None
                    basis = 'prepared continuation'
                else:
                    after = empirical_score
                    basis = 'terminal result' if b.fixed_score is not None else 'parent move row'
                common.update(
                    kind='opponent',
                    reference_score=before,
                    move_score=after,
                    score_basis=basis,
                    prepared=b.target is not None,
                    reference_basis='repertoire value before opponent reply',
                    alternative=None,
                )
            else:
                move_database_score = score(counts(row), color) if row else None
                after = (
                    float(values[b.target][KNOWN]) if b.target is not None and values[b.target][UNKNOWN] == 0 else None
                )
                alternatives = [r for r in rows.values() if r['uci'] != b.move and sum(counts(r)) > 0]
                alternatives.sort(key=lambda r: (-score(counts(r), color), -sum(counts(r)), r['uci']))
                alternative = None
                if (
                    move_database_score is not None
                    and alternatives
                    and score(counts(alternatives[0]), color) > move_database_score
                ):
                    a = alternatives[0]
                    alternative = dict(
                        move=a['uci'],
                        san=san(k, a['uci']),
                        score=score(counts(a), color),
                        sample_count=sum(counts(a)),
                        sparse=sum(counts(a)) < sparse_threshold,
                        gap_pp=100 * (score(counts(a), color) - move_database_score),
                        reference_basis='selected move database score; historical screen only',
                    )
                common.update(
                    kind='own',
                    reference_score=parent_score,
                    move_score=after,
                    move_database_score=move_database_score,
                    score_basis='prepared repertoire continuation',
                    prepared=True,
                    parent_sparse=parent_n < sparse_threshold,
                    continuation_endpoint_sparse=(
                        b.target is not None
                        and model[b.target].mode == 'stop'
                        and model[b.target].branches[0].fixed_score is None
                        and model[b.target].sample < sparse_threshold
                    ),
                    reference_basis='ordinary database score at parent',
                    alternative=alternative,
                )
            before, after = common['reference_score'], common['move_score']
            common['local_drop_pp'] = None if before is None or after is None else 100 * (before - after)
            if common['kind'] == 'own':
                common['local_gain_pp'] = None if common['local_drop_pp'] is None else -common['local_drop_pp']
            result.append(common)
    return result


def rankings(rows):
    result = {}
    for kind in ('opponent', 'own'):
        field = 'weighted_drag_pp' if kind == 'opponent' else 'local_drop_pp'
        result[kind] = sorted(
            (r for r in rows if r['kind'] == kind and r.get(field) is not None and r[field] > 1e-12),
            key=lambda r: (-r[field], r['id']),
        )
    strengths = sorted(
        (r for r in rows if r['kind'] == 'own' and r.get('local_gain_pp') is not None and r['local_gain_pp'] > 1e-12),
        key=lambda r: (-r['local_gain_pp'], r['id']),
    )
    return result, strengths


def rank_scope(local, reach, lines, entry_probability=1.0):
    rows = []
    for candidate in local:
        k = candidate['position']
        if reach[k] <= 0:
            continue
        text, number = lines.get(k, ('', None))
        label = candidate['move_label'] if number is None else move_text(k, number, candidate['move'])
        row = dict(candidate, parent_reach=reach[k], move_label=label, line=(text + ' ' + label).strip())
        row['branch_reach'] = reach[k] * row['branch_probability']
        row['weighted_drag_pp'] = None if row['local_drop_pp'] is None else row['branch_reach'] * row['local_drop_pp']
        row['study_drag_pp_after_entry'] = (
            None
            if entry_probability is None or row['weighted_drag_pp'] is None
            else entry_probability * row['weighted_drag_pp']
        )
        row['alternative_opportunity_pp'] = (
            row['branch_reach'] * row['alternative']['gap_pp'] if row['alternative'] else None
        )
        rows.append(row)
    ranked, strengths = rankings(rows)
    return dict(
        rankings=ranked,
        strengths=strengths,
        evaluated_rows=len(rows),
        unresolved_rows=[r for r in rows if r['weighted_drag_pp'] is None],
        all_signed_rows=rows,
    )


def analyze(path, cache=DEFAULT_CACHE, fetch_missing=False):
    analysis = AnalysisContext(path)
    graph, color, saved, manifest = analysis.graph, analysis.color, analysis.saved, analysis.manifest
    transitions = resolve(graph, color, analysis.policy)
    roots = manifest['root_weights']
    analysis_roots = [*roots, *[e['position'] for c in saved['chapters'] for e in c['entries']]]
    order = topology(transitions, analysis_roots)
    contexts = {'{}': {'transitions': transitions, 'order': order}}
    for chapter in saved['chapters']:
        overrides = chapter.get('policy_overrides', {})
        identity = json.dumps(overrides, sort_keys=True)
        if identity not in contexts:
            ct = resolve(graph, color, dict(analysis.policy, **overrides))
            contexts[identity] = {'transitions': ct, 'order': topology(ct, analysis_roots)}
    # Saved evaluation evidence must be cached and identical to the original run.
    evidence = analysis.read_evidence(cache, list(manifest['evidence']), required=True)
    for context in contexts.values():
        context['model'] = prepare(graph, context['transitions'], context['order'], color, evidence)
        context['sampled'] = empirical(context['model'], color)
        context['values'] = backward(
            context['model'], context['order'], context['sampled'], manifest['sparse_threshold']
        )
    model, sampled, values = (contexts['{}'][k] for k in ('model', 'sampled', 'values'))
    overall = sum(w * values[k] for k, w in roots.items())
    if not np.allclose(
        overall[[KNOWN, UNKNOWN]],
        [saved['overall']['resolved_contribution'], saved['overall']['unresolved_mass']],
        atol=1e-12,
        rtol=0,
    ):
        raise AssertionError('Reconstructed model differs from saved scores')
    own_positions = sorted({k for context in contexts.values() for k, n in context['model'].items() if n.mode == 'own'})
    analysis.read_evidence(cache, own_positions)
    missing = list(analysis.missing)
    if missing and not fetch_missing:
        raise ValueError(f'{len(missing)} own-parent tables missing; use --fetch-missing to cache parents only')
    if missing:
        online = Explorer(cache, manifest['filters'])
        try:
            evidence.update(collect(online, missing, f'{saved["color"]} own-move parents'))
            analysis.provenance.update(online.provenance)
            analysis.missing.clear()
        finally:
            online.close()
    for context in contexts.values():
        context['local'] = candidates(
            graph,
            context['model'],
            context['sampled'],
            context['values'],
            evidence,
            color,
            manifest['sparse_threshold'],
        )
        context['lines'] = representative_lines(graph, context['model'], context['order'], context['sampled'], roots)
    local = contexts['{}']['local']
    reach = reaches(model, order, sampled, roots)
    lines = representative_lines(graph, model, order, sampled, roots)
    overall_scope = rank_scope(local, reach, lines)
    chapters = []
    for chapter in saved['chapters']:
        context = contexts[json.dumps(chapter.get('policy_overrides', {}), sort_keys=True)]
        chapter_model, chapter_order, chapter_sampled, chapter_values = (
            context[k] for k in ('model', 'order', 'sampled', 'values')
        )
        entries = [e['position'] for e in chapter['entries']]
        _, entry_mass = forward(
            chapter_model,
            chapter_order,
            chapter_sampled,
            roots,
            stop_at=entries,
            entering=can_enter(chapter_model, chapter_order, entries),
        )
        probability = sum(float(w) for w in entry_mass.values())
        weights = {k: float(v) / probability for k, v in entry_mass.items() if v > 0} if probability else {}
        reported_probability = chapter['score'].get('entry_probability')
        if reported_probability is not None and not np.isclose(probability, reported_probability, atol=1e-12, rtol=0):
            raise AssertionError('Chapter first-entry probability changed')
        # A single disconnected entry has a meaningful conditional report but no study mass.
        if not weights and len(entries) == 1:
            weights = {entries[0]: 1.0}
        if chapter['score'].get('status') == Status.UNRESOLVED_ENTRY_WEIGHTS:
            weights = {}
        if weights:
            conditional = sum(w * chapter_values[k] for k, w in weights.items())
            expected = chapter['score'].get('raw_empirical_score')
            if expected is not None and not np.isclose(conditional[KNOWN], expected, atol=1e-12, rtol=0):
                raise AssertionError('Chapter conditional score changed')
            scope_lines = representative_lines(
                graph, chapter_model, chapter_order, chapter_sampled, weights, context['lines']
            )
            scope = rank_scope(
                context['local'],
                reaches(chapter_model, chapter_order, chapter_sampled, weights),
                scope_lines,
                reported_probability,
            )
        else:
            scope = dict(rankings={'opponent': [], 'own': []}, all_signed_rows=[], unresolved_rows=[], evaluated_rows=0)
        chapters.append(
            dict(
                id=chapter['id'],
                name=chapter['name'],
                url=chapter.get('url'),
                policy_overrides=chapter.get('policy_overrides', {}),
                overall_policy_entry_probability=chapter['score'].get(
                    'overall_policy_entry_probability', reported_probability
                ),
                entry_probability=reported_probability,
                first_entry_weights=weights,
                repertoire_score=chapter['score'].get('raw_empirical_score'),
                entry_baseline_score=chapter.get('entry_baseline', {}).get('raw_score'),
                delta_vs_entry_baseline_pp=chapter.get('entry_baseline', {}).get('difference_pp'),
                expected_prepared_depth=chapter['score'].get('prepared_depth', {}).get('expected_moves'),
                status=Status.EVALUATED if weights else Status.UNRESOLVED_ENTRY_WEIGHTS,
                **scope,
            )
        )
    # The signed deviations from each opponent mean must cancel, including residual stops.
    max_balance_error = 0.0
    for context in contexts.values():
        chapter_model, chapter_sampled, chapter_values = (context[k] for k in ('model', 'sampled', 'values'))
        for k, node in chapter_model.items():
            if node.mode != 'opponent' or chapter_values[k][UNKNOWN] != 0:
                continue
            balance = sum(
                p * (chapter_values[k][KNOWN] - (chapter_values[b.target][KNOWN] if b.target else s))
                for b, (p, s) in zip(node.branches, chapter_sampled[k])
                if p > 0
            )
            max_balance_error = max(max_balance_error, abs(float(balance)))
    if max_balance_error > 1e-10:
        raise AssertionError('Opponent signed deviations do not balance')
    used = {k: analysis.provenance[k] for k in evidence}
    baseline = saved.get('starting_position_reference', {}).get('owner_score')
    total_score = saved['overall']['raw_empirical_score']
    result = dict(
        color=saved['color'],
        overall_score=total_score,
        starting_baseline_score=baseline,
        overall_delta_pp=None if baseline is None or total_score is None else 100 * (total_score - baseline),
        overall=overall_scope,
        chapters=chapters,
        manifest=analysis.companion_manifest(
            evidence=used,
            cache_only=not missing,
            network_requests=len(missing),
            parent_tables_fetched=len(missing),
            candidate_child_queries=0,
            sparse_threshold=manifest['sparse_threshold'],
        ),
        validation=dict(
            saved_scores_reproduced=True,
            chapter_scores_reproduced=True,
            probability_conservation=True,
            max_opponent_balance_error=max_balance_error,
            own_decision_positions=len(own_positions),
            own_parent_tables_complete=True,
        ),
    )
    result['manifest']['own_score_basis'] = 'prepared repertoire continuation'
    return enrich(result, graph)


def main():
    stage_main('vulnerabilities', analyze, __doc__, configure, options)


def configure(parser):
    parser.add_argument(
        '--fetch-missing',
        action='store_true',
        help='Fetch only missing own decision parent tables; default is cache-only',
    )
    add_token_option(parser)


def options(parser, args):
    apply_token_file(parser, args)
    return dict(fetch_missing=args.fetch_missing)


if __name__ == '__main__':
    main()
