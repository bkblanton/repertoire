"""Rank repertoire vulnerabilities using parent move tables and saved model evidence."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import chess
import numpy as np

from .evaluate import KNOWN, UNKNOWN, backward, forward
from .explorer import Explorer, counts
from .graph import parse, resolve, topology
from .model import empirical, prepare, score
from .attribution import enrich, chapter_text, ATTRIBUTION_NOTE, vulnerability_summary


def reaches(model, order, sampled, roots):
    """Incoming probability at each canonical position, aggregating transpositions."""
    mass = dict.fromkeys(order, 0.0)
    for k, weight in roots.items():
        mass[k] += weight
    stopped = 0.0
    for k in reversed(order):
        for b, (p, _) in zip(model[k].branches, sampled[k]):
            flow = mass[k] * p
            if b.target is None:
                stopped += flow
            else:
                mass[b.target] += flow
    if not np.isclose(stopped, sum(roots.values()), atol=1e-10):
        raise AssertionError('Vulnerability probability conservation failed')
    return mass


def move_label(board, uci):
    return f'{board.fullmove_number}{"." if board.turn else "..."}{board.san(chess.Move.from_uci(uci))}'


def representative_lines(graph, model, order, sampled, roots, prefixes=None):
    """One legal policy route for labels; probabilities always use all routes."""
    paths, best = {}, dict.fromkeys(order, -1.0)
    for k, weight in roots.items():
        if weight <= 0:
            continue
        best[k] = weight
        paths[k] = (prefixes or {}).get(k, '')
    for k in reversed(order):
        if k not in paths:
            continue
        for b, (p, _) in zip(model[k].branches, sampled[k]):
            if b.target is not None and p > 0 and best[k]*p > best[b.target]:
                best[b.target] = best[k]*p
                paths[b.target] = (paths[k]+' '+move_label(chess.Board(graph.nodes[k].fen), b.move)).strip()
    return paths


def candidates(graph, model, sampled, values, evidence, color, sparse_threshold):
    """Local signed score changes. This pure function cannot request child evidence."""
    result = []
    for k, node in model.items():
        if node.mode == 'stop':
            continue
        board = chess.Board(graph.nodes[k].fen)
        parent_data = evidence[k]
        parent_n = sum(counts(parent_data))
        rows = {r['uci']: r for r in parent_data['moves']}
        parent_score = score(counts(parent_data), color)
        for j, (b, (p, empirical_score)) in enumerate(zip(node.branches, sampled[k])):
            if not b.move or p <= 0:
                continue
            row = rows.get(b.move)
            n = sum(counts(row)) if row else 0
            common = dict(id=f'{k}|{b.move}', position=k, move=b.move,
                          move_san=board.san(chess.Move.from_uci(b.move)),
                          move_label=move_label(board, b.move), branch_probability=float(p),
                          sample_count=n, parent_sample_count=parent_n,
                          counts_white_draw_black=counts(row) if row else [0, 0, 0],
                          sparse=n < sparse_threshold, chapters=sorted(graph.nodes[k].chapters),
                          target=b.target)
            if node.mode == 'opponent':
                before = float(values[k][KNOWN, 0]) if values[k][UNKNOWN, 0] == 0 else None
                if b.target is not None:
                    after = float(values[b.target][KNOWN, 0]) if values[b.target][UNKNOWN, 0] == 0 else None
                    basis = 'prepared continuation'
                else:
                    after = empirical_score
                    basis = 'terminal result' if b.fixed_score is not None else 'parent move row'
                common.update(kind='opponent', reference_score=before, move_score=after,
                              score_basis=basis, prepared=b.target is not None,
                              reference_basis='repertoire value before opponent reply',
                              alternative=None)
            else:
                move_database_score = score(counts(row), color) if row else None
                after = (float(values[b.target][KNOWN, 0])
                         if b.target is not None and values[b.target][UNKNOWN, 0] == 0 else None)
                alternatives = [r for r in rows.values() if r['uci'] != b.move and sum(counts(r)) > 0]
                alternatives.sort(key=lambda r: (-score(counts(r), color), -sum(counts(r)), r['uci']))
                alternative = None
                if move_database_score is not None and alternatives and score(counts(alternatives[0]), color) > move_database_score:
                    a = alternatives[0]
                    alternative = dict(move=a['uci'], san=board.san(chess.Move.from_uci(a['uci'])),
                                       score=score(counts(a), color), sample_count=sum(counts(a)),
                                       sparse=sum(counts(a)) < sparse_threshold,
                                       gap_pp=100*(score(counts(a), color)-move_database_score),
                                       reference_basis='selected move database score; historical screen only')
                common.update(kind='own', reference_score=parent_score, move_score=after,
                              move_database_score=move_database_score,
                              score_basis='prepared repertoire continuation', prepared=True,
                              parent_sparse=parent_n < sparse_threshold,
                              continuation_endpoint_sparse=(b.target is not None and model[b.target].mode == 'stop'
                                  and model[b.target].branches[0].fixed_score is None and model[b.target].sample < sparse_threshold),
                              reference_basis='ordinary database score at parent', alternative=alternative)
            before, after = common['reference_score'], common['move_score']
            common['local_drop_pp'] = None if before is None or after is None else 100*(before-after)
            if common['kind'] == 'own':
                common['local_gain_pp'] = None if common['local_drop_pp'] is None else -common['local_drop_pp']
            result.append(common)
    return result


def rankings(rows):
    result = {}
    for kind in ('opponent', 'own'):
        field = 'weighted_drag_pp' if kind == 'opponent' else 'local_drop_pp'
        result[kind] = sorted((r for r in rows if r['kind'] == kind and r.get(field) is not None
                               and r[field] > 1e-12), key=lambda r: (-r[field], r['id']))
    strengths = sorted((r for r in rows if r['kind'] == 'own' and r.get('local_gain_pp') is not None
                        and r['local_gain_pp'] > 1e-12), key=lambda r: (-r['local_gain_pp'], r['id']))
    return result, strengths


def rank_scope(local, reach, lines, entry_probability=1.0):
    rows = []
    for candidate in local:
        k = candidate['position']
        if reach[k] <= 0:
            continue
        row = dict(candidate, parent_reach=reach[k],
                   line=(lines.get(k, '')+' '+candidate['move_label']).strip())
        row['branch_reach'] = reach[k]*row['branch_probability']
        row['weighted_drag_pp'] = None if row['local_drop_pp'] is None else row['branch_reach']*row['local_drop_pp']
        row['study_drag_pp_after_entry'] = (None if entry_probability is None or row['weighted_drag_pp'] is None
                                           else entry_probability*row['weighted_drag_pp'])
        row['alternative_opportunity_pp'] = (row['branch_reach']*row['alternative']['gap_pp']
                                              if row['alternative'] else None)
        rows.append(row)
    ranked, strengths = rankings(rows)
    return dict(rankings=ranked, strengths=strengths, evaluated_rows=len(rows),
                unresolved_rows=[r for r in rows if r['weighted_drag_pp'] is None],
                all_signed_rows=rows)


def analyze(path, cache='.cache/explorer', fetch_missing=False):
    path = Path(path)
    saved = json.loads(path.read_text(encoding='utf-8'))
    manifest = saved['manifest']
    source = Path(manifest['input_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError(f'PGN differs from saved scores; reanalyze first: {source}')
    color = saved['color'] == 'white'
    graph = parse(source, manifest['configuration'].get('exclude', []))
    transitions = resolve(graph, color, manifest['configuration'].get('policy', {}))
    roots = manifest['root_weights']
    analysis_roots = [*roots, *[e['position'] for c in saved['chapters'] for e in c['entries']]]
    order = topology(transitions, analysis_roots)
    contexts = {'{}': {'transitions': transitions, 'order': order}}
    for chapter in saved['chapters']:
        overrides = chapter.get('policy_overrides', {})
        identity = json.dumps(overrides, sort_keys=True)
        if identity not in contexts:
            ct = resolve(graph, color, dict(manifest['configuration'].get('policy', {}), **overrides))
            contexts[identity] = {'transitions': ct, 'order': topology(ct, analysis_roots)}
    evidence = {}
    # Saved evaluation evidence must be cached and identical to the original run.
    explorer = Explorer(cache, manifest['filters'], offline=True)
    try:
        for k, original in manifest['evidence'].items():
            evidence[k] = explorer.get(k)
            if explorer.provenance[k] != original:
                raise ValueError(f'Cached evaluation evidence changed; reanalyze scores first: {k}')
        for context in contexts.values():
            context['model'] = prepare(graph, context['transitions'], context['order'], color, evidence)
            context['sampled'] = empirical(context['model'], color)
            context['values'] = backward(context['model'], context['order'], context['sampled'], manifest['sparse_threshold'])
        model, sampled, values = (contexts['{}'][k] for k in ('model', 'sampled', 'values'))
        overall = sum(w*values[k] for k, w in roots.items())
        if not np.allclose(overall[[KNOWN, UNKNOWN], 0],
                           [saved['overall']['resolved_contribution'], saved['overall']['unresolved_mass']],
                           atol=1e-12, rtol=0):
            raise AssertionError('Reconstructed model differs from saved scores')
        missing = []
        own_positions = sorted({k for context in contexts.values() for k,n in context['model'].items() if n.mode == 'own'})
        for k in own_positions:
            if k not in evidence:
                try:
                    evidence[k] = explorer.get(k)
                except ValueError as exc:
                    if not str(exc).startswith('Offline cache miss:'):
                        raise
                    missing.append(k)
        provenance = dict(explorer.provenance)
    finally:
        explorer.close()
    if missing and not fetch_missing:
        raise ValueError(f'{len(missing)} own-parent tables missing; use --fetch-missing to cache parents only')
    if missing:
        online = Explorer(cache, manifest['filters'], delay=2.2)
        try:
            for i, k in enumerate(missing, 1):
                evidence[k] = online.get(k)
                if i % 10 == 0 or i == len(missing):
                    print(f'{saved["color"]}: cached {i}/{len(missing)} missing parent tables', flush=True)
            provenance.update(online.provenance)
        finally:
            online.close()
    for context in contexts.values():
        context['local'] = candidates(graph, context['model'], context['sampled'], context['values'], evidence, color, manifest['sparse_threshold'])
        context['lines'] = representative_lines(graph, context['model'], context['order'], context['sampled'], roots)
    local = contexts['{}']['local']
    reach = reaches(model, order, sampled, roots)
    lines = representative_lines(graph, model, order, sampled, roots)
    overall_scope = rank_scope(local, reach, lines)
    chapters = []
    for chapter in saved['chapters']:
        context = contexts[json.dumps(chapter.get('policy_overrides', {}), sort_keys=True)]
        cm, co, cs, cv = (context[k] for k in ('model', 'order', 'sampled', 'values'))
        entries = [e['position'] for e in chapter['entries']]
        _, entry_mass = forward(cm, co, cs, roots, stop_at=entries)
        probability = sum(float(w[0]) for w in entry_mass.values())
        weights = {k: float(v[0])/probability for k, v in entry_mass.items() if v[0] > 0} if probability else {}
        reported_probability = chapter['score'].get('entry_probability')
        if reported_probability is not None and not np.isclose(probability, reported_probability, atol=1e-12, rtol=0):
            raise AssertionError('Chapter first-entry probability changed')
        # A single disconnected entry has a meaningful conditional report but no study mass.
        if not weights and len(entries) == 1:
            weights = {entries[0]: 1.0}
        if chapter['score'].get('status') == 'unresolved_first_entry_weights':
            weights = {}
        if weights:
            conditional = sum(w*cv[k] for k, w in weights.items())
            expected = chapter['score'].get('raw_empirical_score')
            if expected is not None and not np.isclose(conditional[KNOWN, 0], expected, atol=1e-12, rtol=0):
                raise AssertionError('Chapter conditional score changed')
            scope_lines = representative_lines(graph, cm, co, cs, weights, context['lines'])
            scope = rank_scope(context['local'], reaches(cm, co, cs, weights), scope_lines, reported_probability)
        else:
            scope = dict(rankings={'opponent': [], 'own': []}, all_signed_rows=[], unresolved_rows=[], evaluated_rows=0)
        chapters.append(dict(id=chapter['id'], name=chapter['name'], url=chapter.get('url'),
                             policy_overrides=chapter.get('policy_overrides', {}),
                             overall_policy_entry_probability=chapter['score'].get('overall_policy_entry_probability', reported_probability),
                             entry_probability=reported_probability, first_entry_weights=weights,
                             repertoire_score=chapter['score'].get('raw_empirical_score'),
                             entry_baseline_score=chapter.get('entry_baseline', {}).get('raw_score'),
                             delta_vs_entry_baseline_pp=chapter.get('entry_baseline', {}).get('difference_pp'),
                             expected_prepared_depth=chapter['score'].get('prepared_depth', {}).get('expected_moves'),
                             status='evaluated' if weights else 'unresolved_entry_weights', **scope))
    # The signed deviations from each opponent mean must cancel, including residual stops.
    max_balance_error = 0.0
    for context in contexts.values():
        cm, cs, cv = (context[k] for k in ('model', 'sampled', 'values'))
        for k, node in cm.items():
            if node.mode != 'opponent' or cv[k][UNKNOWN, 0] != 0:
                continue
            balance = sum(p*(cv[k][KNOWN, 0]-(cv[b.target][KNOWN, 0] if b.target else s))
                          for b, (p, s) in zip(node.branches, cs[k]) if p > 0)
            max_balance_error = max(max_balance_error, abs(float(balance)))
    if max_balance_error > 1e-10:
        raise AssertionError('Opponent signed deviations do not balance')
    used = {k: provenance[k] for k in evidence}
    baseline = saved.get('starting_position_reference', {}).get('owner_score')
    total_score = saved['overall']['raw_empirical_score']
    result = dict(color=saved['color'], overall_score=total_score, starting_baseline_score=baseline,
                overall_delta_pp=None if baseline is None or total_score is None else 100*(total_score-baseline),
                overall=overall_scope, chapters=chapters,
                manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
                              report_path=str(path.resolve()), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              input_path=str(source), input_sha256=manifest['input_sha256'],
                              filters=manifest['filters'], evidence=used, parent_tables_fetched=len(missing),
                              candidate_child_queries=0, sparse_threshold=manifest['sparse_threshold']),
                validation=dict(saved_scores_reproduced=True, chapter_scores_reproduced=True,
                                probability_conservation=True, max_opponent_balance_error=max_balance_error,
                                own_decision_positions=len(own_positions),
                                own_parent_tables_complete=True))
    result['manifest']['own_score_basis'] = 'prepared repertoire continuation'
    return enrich(result, graph)


def refresh_saved(path):
    """Update own comparisons from a matching saved character continuation model.

    This does not parse a newer PGN, query the cache, or change scoring evidence.
    """
    path = Path(path); saved_bytes = path.read_bytes(); saved = json.loads(saved_bytes)
    companion = path.with_suffix('.vulnerabilities.json')
    result = json.loads(companion.read_bytes())
    character_path = path.with_suffix('.character.json')
    character_bytes = character_path.read_bytes(); character = json.loads(character_bytes)
    for data in (result, character):
        if data['color'] != saved['color'] or any(data['manifest'].get(k) != v for k, v in (
                ('report_sha256', hashlib.sha256(saved_bytes).hexdigest()),
                ('input_sha256', saved['manifest']['input_sha256']), ('filters', saved['manifest']['filters']))):
            raise ValueError('Saved continuation analysis belongs to a different score snapshot')
    positions = {s['id']: {r['position']: r for r in s.get('positions', [])} for s in character['scopes']}
    for sid, scope in [('overall', result['overall']), *[(s['id'], s) for s in result['chapters']]]:
        for row in scope['all_signed_rows']:
            if row['kind'] != 'own': continue
            target = positions.get(sid, {}).get(row['target'])
            if target is None:
                raise ValueError('Saved continuation position missing from character analysis')
            row.setdefault('move_database_score', row['move_score'])
            row['move_score'] = target['repertoire_score']
            row['score_basis'] = 'prepared repertoire continuation'
            row['continuation_endpoint_sparse'] = (target.get('kind') == 'theory_leaf' and target.get('games') is not None
                and target['games'] < result['manifest']['sparse_threshold'])
            row['parent_sparse'] = row['parent_sample_count'] < result['manifest']['sparse_threshold']
            before, after = row['reference_score'], row['move_score']
            row['local_drop_pp'] = None if before is None or after is None else 100 * (before - after)
            row['local_gain_pp'] = None if row['local_drop_pp'] is None else -row['local_drop_pp']
            row['weighted_drag_pp'] = None if row['local_drop_pp'] is None else row['branch_reach'] * row['local_drop_pp']
            probability = 1. if sid == 'overall' else scope['entry_probability']
            row['study_drag_pp_after_entry'] = (None if probability is None or row['weighted_drag_pp'] is None
                                               else probability * row['weighted_drag_pp'])
            if row.get('alternative'):
                row['alternative']['reference_basis'] = 'selected move database score; historical screen only'
        scope['rankings'], scope['strengths'] = rankings(scope['all_signed_rows'])
        scope['unresolved_rows'] = [r for r in scope['all_signed_rows'] if r['weighted_drag_pp'] is None]
    result['manifest']['own_score_basis'] = 'prepared repertoire continuation'
    result['manifest']['continuation_source_sha256'] = hashlib.sha256(character_bytes).hexdigest()
    result['manifest']['comparison_updated_at'] = datetime.now(timezone.utc).isoformat()
    result['validation']['own_scores_from_prepared_continuations'] = True
    return result


def pct(value):
    return 'unresolved' if value is None else f'{100*value:.2f}%'


def delta(value):
    return 'unresolved' if value is None else f'{value:+.3f} pp'


def table(rows, limit, chapter=False, own=False, catalog=()):
    if not rows:
        return ['No positive drag identified among resolved, reachable moves.', '']
    before_label = 'Parent database score' if own else 'Before reply repertoire score'
    after_label = 'Repertoire continuation score' if own else 'After reply score'
    heading = f'| # | Representative line | Chapter source / context | Reach of move | {before_label} | {after_label} | Drop (pp) | Weighted drag (pp) |'
    divider = '|---:|---|---|---:|---:|---:|---:|---:|'
    if chapter:
        heading += ' Root drag under chapter policy (pp) |'
        divider += '---:|'
    heading += ' Games for move |'
    divider += '---:|'
    if own:
        heading += ' Highest-scoring observed alternative (screen only) |'
        divider += '---|'
    text = [heading, divider]
    for i, row in enumerate(rows[:limit], 1):
        value = f"| {i} | `{row['line']}` | {chapter_text(row,catalog)} | {pct(row['branch_reach'])} | {pct(row['reference_score'])} | {pct(row['move_score'])} | {row['local_drop_pp']:.3f} | {row['weighted_drag_pp']:.4f} |"
        if chapter:
            v = row['study_drag_pp_after_entry']
            value += ' '+('unresolved' if v is None else f'{v:.4f}')+' |'
        value += f" {row['sample_count']:,}{' (sparse)' if row['sparse'] else ''} |"
        if own:
            a = row['alternative']
            value += (' none observed |' if not a else
                      f" {a['san']}: {pct(a['score'])}, n={a['sample_count']:,}{' (sparse)' if a['sparse'] else ''}; {chapter_text(a,catalog)} |")
        text.append(value)
    return text+['']


def opponent_tables(rows, limit, chapter=False, catalog=()):
    text = []
    for prepared, label in ((False, 'Unprepared'), (True, 'Prepared')):
        text += [f'### {label} opponent replies', '']
        text += table([r for r in rows if r['prepared'] == prepared], limit, chapter=chapter, catalog=catalog)
    return text


def markdown(report, top=20, chapter_top=5):
    color = report['color'].title()
    filters = report['manifest']['filters']
    catalog = report.get('chapter_catalog',[])
    text = [f'# {color} repertoire vulnerabilities', '',
            f"Repertoire score: **{pct(report['overall_score'])}**. "
            f"{color} starting-position baseline: **{pct(report['starting_baseline_score'])}**. "
            f"Difference: **{delta(report['overall_delta_pp'])}**.", '',
            f"Lichess population: speeds `{filters['speeds']}`; rating groups `{filters['ratings']}`; "
            f"dates `{filters['since']}` to `{filters['until']}`.", '',
            'Opponent replies rank by weighted score deficit; our moves rank by parent database score minus repertoire continuation score. '
            'Rows can overlap, so do not sum them or interpret drag as a guaranteed improvement. Definitions and evidence checks follow the chapter tables.', '',
            ATTRIBUTION_NOTE, '', '## Overall study', '']
    methods = [
            'Rankings identify empirical pressure points. Positive weighted drag means a score deficit relative to the stated local reference. '
            'It is a screening measure, not a promised improvement, an engine judgment, or proof that a move causes worse results.', '',
            'Opponent reply drag = probability of reaching the parent x reply probability x (parent repertoire score - score after reply). '
            'Prepared replies use the full merged continuation; unprepared replies use the cached parent move row. '
            'Our move drag = ordinary parent database score - repertoire continuation score after our move. Own rankings use this direct difference; the reach-weighted diagnostic is retained separately. '
            'Our move popularity is not a weight. The two benchmarks differ, so their rankings are separate.', '',
            'All scores use the repertoire owner\'s perspective. Weighted drag is in percentage points of the overall score, '
            'or of the conditional chapter score in chapter tables. Chapter root drag is weighted again by entry probability under the chapter comparison policy. '
            'These values overlap across depths and chapters and must not be added. They do not decompose the overall baseline delta.', '',
            'Reach aggregates all modeled transposition routes. The displayed line is one representative route, not its exclusive frequency. '
            'Chapter rankings start at first entry, use the same weighted entry mixture and chapter comparison policy as the score, and retain compatible continuations from other chapters. '
            'A move that enters a chapter is shown in the overall or upstream chapter ranking, not charged again after entry.', '',
            'Own-move alternatives are the highest observed scores in the same parent table, with their game counts. '
            'Selecting the largest observed score exaggerates noisy small samples; it is not a recommendation to replace the move. '
            f"Counts below {report['manifest']['sparse_threshold']} are flagged. Opponent counts measure reply frequency; prepared continuation scores can rely on other downstream evidence.", '',
            'Unobserved moves have zero empirical frequency. Missing score evidence stays unresolved and is excluded from positive rankings. '
            'No score intervals are inferred from these rankings. Non-move residual stopping buckets are included in model validation but are not ranked as chess moves.', '']
    text += opponent_tables(report['overall']['rankings']['opponent'], top, catalog=catalog)
    text += ['### Our selected moves', '']+table(report['overall']['rankings']['own'], top, own=True, catalog=catalog)
    for chapter in report['chapters']:
        text += [f"## {chapter['name']}", '', f"Chapter entry probability: **{pct(chapter['entry_probability'])}**. Rankings are conditional on first entry.", '']
        if chapter.get('policy_overrides'):
            text += ['Alternative chapter comparison: this chapter\'s first own moves take precedence, with the overall policy elsewhere. '
                     f"Region reach under the overall policy: **{pct(chapter.get('overall_policy_entry_probability'))}**. "
                     'Root drag uses the chapter comparison policy and is not an impact on the currently selected overall repertoire.', '']
        text += [f"Repertoire score: {pct(chapter['repertoire_score'])}; weighted entry baseline: {pct(chapter['entry_baseline_score'])}; "
                 f"difference: {delta(chapter['delta_vs_entry_baseline_pp'])}. "
                 'Expected prepared depth: '+('unresolved' if chapter['expected_prepared_depth'] is None
                                               else f"{chapter['expected_prepared_depth']:.2f} own moves")+'.', '']
        if chapter['status'] != 'evaluated':
            text += ['Entry weights are unresolved; no conditional ranking is available.', '']
            continue
        text += opponent_tables(chapter['rankings']['opponent'], chapter_top, chapter=True, catalog=catalog)
        text += ['### Our selected moves', '']+table(chapter['rankings']['own'], chapter_top, chapter=True, own=True, catalog=catalog)
        text += [f"Unresolved reachable move comparisons: {len(chapter['unresolved_rows'])}.", '']
    text += ['## How to read the rankings', '', *methods, '## Evidence and checks', '',
             'All candidate results come from parent-position response tables. Candidate child requests: **0**. '
             'Evaluation evidence is checked against the saved report cache provenance, and source PGN hashes must match. '
             'Additional own-parent tables may have newer retrieval dates; exact timestamps and cache keys are in JSON. '
             'The saved overall and chapter scores were reproduced, probability mass was conserved, and signed opponent deviations balanced around each parent mean.', '',
             f"Own decision parent tables available: {report['validation']['own_decision_positions']}. "
             f"Unresolved overall move comparisons: {len(report['overall']['unresolved_rows'])}. "
             'Full signed comparisons, exact FENs, first-entry weights and all positive rankings are retained in the companion JSON.', '']
    return '\n'.join(text)


def write_outputs(report, path, refresh_rating_provenance=False):
    path = Path(path)
    companion = path.with_suffix('.vulnerabilities.json')
    rating_path = path.with_suffix('.ratings.json'); rating = None
    if refresh_rating_provenance and rating_path.exists():
        rating = json.loads(rating_path.read_bytes()); m = rating['manifest']
        if rating['color'] != report['color'] or any(m.get(k) != report['manifest'][k]
                for k in ('report_sha256', 'input_sha256', 'filters')):
            raise ValueError('Rating ledger differs from saved comparison snapshot')
        for family, digest in m['supporting_sha256'].items():
            if hashlib.sha256(path.with_suffix(f'.{family}.json').read_bytes()).hexdigest() != digest:
                raise ValueError('Rating supporting analysis changed before comparison refresh')
    companion.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if rating is not None:
        # Only own score comparisons changed; all move identities and rating
        # sources remain the same, so keep the verified cached rating contexts.
        rating['manifest']['supporting_sha256']['vulnerabilities'] = hashlib.sha256(companion.read_bytes()).hexdigest()
        rating['manifest']['comparison_provenance_updated_at'] = datetime.now(timezone.utc).isoformat()
        rating_path.write_text(json.dumps(rating, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', help='Saved score JSON files, with unchanged source PGNs')
    parser.add_argument('--cache', default='.cache/explorer')
    refresh_mode = parser.add_mutually_exclusive_group()
    refresh_mode.add_argument('--fetch-missing', action='store_true', help='Fetch only missing own decision parent tables; default is cache-only')
    refresh_mode.add_argument('--refresh-saved', action='store_true', help='Update own comparisons from matching saved character scores without parsing PGNs or fetching data')
    args = parser.parse_args()
    for path in args.reports:
        result = refresh_saved(path) if args.refresh_saved else analyze(path, args.cache, args.fetch_missing)
        write_outputs(result, path, refresh_rating_provenance=args.refresh_saved)
        print(f"Generated {result['color']} vulnerabilities: overall and {len(result['chapters'])} chapters", flush=True)
    from .render import update_report_outputs
    update_report_outputs(args.reports[-1])


if __name__ == '__main__':
    main()
