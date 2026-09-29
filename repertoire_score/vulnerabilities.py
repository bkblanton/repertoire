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
                after = score(counts(row), color) if row else None
                alternatives = [r for r in rows.values() if r['uci'] != b.move and sum(counts(r)) > 0]
                alternatives.sort(key=lambda r: (-score(counts(r), color), -sum(counts(r)), r['uci']))
                alternative = None
                if after is not None and alternatives and score(counts(alternatives[0]), color) > after:
                    a = alternatives[0]
                    alternative = dict(move=a['uci'], san=board.san(chess.Move.from_uci(a['uci'])),
                                       score=score(counts(a), color), sample_count=sum(counts(a)),
                                       sparse=sum(counts(a)) < sparse_threshold,
                                       gap_pp=100*(score(counts(a), color)-after))
                common.update(kind='own', reference_score=parent_score, move_score=after,
                              score_basis='parent move row', prepared=True,
                              reference_basis='ordinary database score at parent', alternative=alternative)
            before, after = common['reference_score'], common['move_score']
            common['local_drop_pp'] = None if before is None or after is None else 100*(before-after)
            result.append(common)
    return result


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
    ranked = {}
    for kind in ('opponent', 'own'):
        ranked[kind] = sorted((r for r in rows if r['kind'] == kind and r['weighted_drag_pp'] is not None
                               and r['weighted_drag_pp'] > 1e-12),
                              key=lambda r: (-r['weighted_drag_pp'], r['id']))
    return dict(rankings=ranked, evaluated_rows=len(rows),
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
    order = topology(transitions, [*roots, *[e['position'] for c in saved['chapters'] for e in c['entries']]])
    evidence = {}
    # Saved evaluation evidence must be cached and identical to the original run.
    explorer = Explorer(cache, manifest['filters'], offline=True)
    try:
        for k, original in manifest['evidence'].items():
            evidence[k] = explorer.get(k)
            if explorer.provenance[k] != original:
                raise ValueError(f'Cached evaluation evidence changed; reanalyze scores first: {k}')
        model = prepare(graph, transitions, order, color, evidence)
        sampled = empirical(model, color)
        values = backward(model, order, sampled, manifest['sparse_threshold'])
        overall = sum(w*values[k] for k, w in roots.items())
        if not np.allclose(overall[[KNOWN, UNKNOWN], 0],
                           [saved['overall']['resolved_contribution'], saved['overall']['unresolved_mass']],
                           atol=1e-12, rtol=0):
            raise AssertionError('Reconstructed model differs from saved scores')
        missing = []
        for k in order:
            if model[k].mode == 'own' and k not in evidence:
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
    local = candidates(graph, model, sampled, values, evidence, color, manifest['sparse_threshold'])
    reach = reaches(model, order, sampled, roots)
    lines = representative_lines(graph, model, order, sampled, roots)
    overall_scope = rank_scope(local, reach, lines)
    chapters = []
    for chapter in saved['chapters']:
        entries = [e['position'] for e in chapter['entries']]
        _, entry_mass = forward(model, order, sampled, roots, stop_at=entries)
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
            conditional = sum(w*values[k] for k, w in weights.items())
            expected = chapter['score'].get('raw_empirical_score')
            if expected is not None and not np.isclose(conditional[KNOWN, 0], expected, atol=1e-12, rtol=0):
                raise AssertionError('Chapter conditional score changed')
            scope_lines = representative_lines(graph, model, order, sampled, weights, lines)
            scope = rank_scope(local, reaches(model, order, sampled, weights), scope_lines, reported_probability)
        else:
            scope = dict(rankings={'opponent': [], 'own': []}, all_signed_rows=[], unresolved_rows=[], evaluated_rows=0)
        chapters.append(dict(id=chapter['id'], name=chapter['name'], url=chapter.get('url'),
                             entry_probability=reported_probability, first_entry_weights=weights,
                             repertoire_score=chapter['score'].get('raw_empirical_score'),
                             entry_baseline_score=chapter.get('entry_baseline', {}).get('raw_score'),
                             delta_vs_entry_baseline_pp=chapter.get('entry_baseline', {}).get('difference_pp'),
                             expected_prepared_depth=chapter['score'].get('prepared_depth', {}).get('expected_moves'),
                             status='evaluated' if weights else 'unresolved_entry_weights', **scope))
    # The signed deviations from each opponent mean must cancel, including residual stops.
    max_balance_error = 0.0
    for k, node in model.items():
        if node.mode != 'opponent' or values[k][UNKNOWN, 0] != 0:
            continue
        balance = sum(p*(values[k][KNOWN, 0]-(values[b.target][KNOWN, 0] if b.target else s))
                      for b, (p, s) in zip(node.branches, sampled[k]) if p > 0)
        max_balance_error = max(max_balance_error, abs(float(balance)))
    if max_balance_error > 1e-10:
        raise AssertionError('Opponent signed deviations do not balance')
    used = {k: provenance[k] for k in evidence}
    baseline = saved.get('starting_position_reference', {}).get('owner_score')
    total_score = saved['overall']['raw_empirical_score']
    return dict(color=saved['color'], overall_score=total_score, starting_baseline_score=baseline,
                overall_delta_pp=None if baseline is None or total_score is None else 100*(total_score-baseline),
                overall=overall_scope, chapters=chapters,
                manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
                              report_path=str(path.resolve()), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              input_path=str(source), input_sha256=manifest['input_sha256'],
                              filters=manifest['filters'], evidence=used, parent_tables_fetched=len(missing),
                              candidate_child_queries=0, sparse_threshold=manifest['sparse_threshold']),
                validation=dict(saved_scores_reproduced=True, chapter_scores_reproduced=True,
                                probability_conservation=True, max_opponent_balance_error=max_balance_error,
                                own_decision_positions=sum(n.mode == 'own' for n in model.values()),
                                own_parent_tables_complete=True))


def pct(value):
    return 'unresolved' if value is None else f'{100*value:.2f}%'


def delta(value):
    return 'unresolved' if value is None else f'{value:+.3f} pp'


def table(rows, limit, chapter=False, own=False):
    if not rows:
        return ['No positive drag identified among resolved, reachable moves.', '']
    before_label = 'Parent database score' if own else 'Before reply repertoire score'
    after_label = 'Selected move database score' if own else 'After reply score'
    heading = f'| # | Representative line | Reach of move | {before_label} | {after_label} | Drop (pp) | Weighted drag (pp) |'
    divider = '|---:|---|---:|---:|---:|---:|---:|'
    if chapter:
        heading += ' Study drag after entry (pp) |'
        divider += '---:|'
    heading += ' Games for move |'
    divider += '---:|'
    if own:
        heading += ' Highest-scoring observed alternative (screen only) |'
        divider += '---|'
    text = [heading, divider]
    for i, row in enumerate(rows[:limit], 1):
        value = f"| {i} | `{row['line']}` | {pct(row['branch_reach'])} | {pct(row['reference_score'])} | {pct(row['move_score'])} | {row['local_drop_pp']:.3f} | {row['weighted_drag_pp']:.4f} |"
        if chapter:
            v = row['study_drag_pp_after_entry']
            value += ' '+('unresolved' if v is None else f'{v:.4f}')+' |'
        value += f" {row['sample_count']:,}{' (sparse)' if row['sparse'] else ''} |"
        if own:
            a = row['alternative']
            value += (' none observed |' if not a else
                      f" {a['san']}: {pct(a['score'])}, n={a['sample_count']:,}{' (sparse)' if a['sparse'] else ''} |")
        text.append(value)
    return text+['']


def opponent_tables(rows, limit, chapter=False):
    text = []
    for prepared, label in ((False, 'Unprepared'), (True, 'Prepared')):
        text += [f'### {label} opponent replies', '']
        text += table([r for r in rows if r['prepared'] == prepared], limit, chapter=chapter)
    return text


def markdown(report, top=20, chapter_top=5):
    color = report['color'].title()
    filters = report['manifest']['filters']
    text = [f'# {color} repertoire vulnerabilities', '',
            f"Repertoire score: **{pct(report['overall_score'])}**. "
            f"{color} starting-position baseline: **{pct(report['starting_baseline_score'])}**. "
            f"Difference: **{delta(report['overall_delta_pp'])}**.", '',
            f"Lichess population: speeds `{filters['speeds']}`; rating groups `{filters['ratings']}`; "
            f"dates `{filters['since']}` to `{filters['until']}`.", '',
            'Ranked by weighted score deficit. Opponent replies and our moves use different references and are ranked separately. '
            'Rows can overlap, so do not sum them or interpret drag as a guaranteed improvement. Definitions and evidence checks follow the chapter tables.', '',
            '## Overall study', '']
    methods = [
            'Rankings identify empirical pressure points. Positive weighted drag means a score deficit relative to the stated local reference. '
            'It is a screening measure, not a promised improvement, an engine judgment, or proof that a move causes worse results.', '',
            'Opponent reply drag = probability of reaching the parent x reply probability x (parent repertoire score - score after reply). '
            'Prepared replies use the full merged continuation; unprepared replies use the cached parent move row. '
            'Our move drag = probability of reaching the parent x selected policy weight x (ordinary parent database score - selected move-row score). '
            'Our move popularity is not a weight. The two benchmarks differ, so their rankings are separate.', '',
            'All scores use the repertoire owner\'s perspective. Weighted drag is in percentage points of the overall score, '
            'or of the conditional chapter score in chapter tables. Chapter study drag is weighted again by chapter entry probability. '
            'These values overlap across depths and chapters and must not be added. They do not decompose the overall baseline delta.', '',
            'Reach aggregates all modeled transposition routes. The displayed line is one representative route, not its exclusive frequency. '
            'Chapter rankings start at first entry, use the same weighted entry mixture as the score, and follow the complete merged repertoire thereafter. '
            'A move that enters a chapter is shown in the overall or upstream chapter ranking, not charged again after entry.', '',
            'Own-move alternatives are the highest observed scores in the same parent table, with their game counts. '
            'Selecting the largest observed score exaggerates noisy small samples; it is not a recommendation to replace the move. '
            f"Counts below {report['manifest']['sparse_threshold']} are flagged. Opponent counts measure reply frequency; prepared continuation scores can rely on other downstream evidence.", '',
            'Unobserved moves have zero empirical frequency. Missing score evidence stays unresolved and is excluded from positive rankings. '
            'No score intervals are inferred from these rankings. Non-move residual stopping buckets are included in model validation but are not ranked as chess moves.', '']
    text += opponent_tables(report['overall']['rankings']['opponent'], top)
    text += ['### Our selected moves', '']+table(report['overall']['rankings']['own'], top, own=True)
    for chapter in report['chapters']:
        text += [f"## {chapter['name']}", '', f"Chapter entry probability: **{pct(chapter['entry_probability'])}**. Rankings are conditional on first entry.", '']
        text += [f"Repertoire score: {pct(chapter['repertoire_score'])}; weighted entry baseline: {pct(chapter['entry_baseline_score'])}; "
                 f"difference: {delta(chapter['delta_vs_entry_baseline_pp'])}. "
                 'Expected prepared depth: '+('unresolved' if chapter['expected_prepared_depth'] is None
                                               else f"{chapter['expected_prepared_depth']:.2f} own moves")+'.', '']
        if chapter['status'] != 'evaluated':
            text += ['Entry weights are unresolved; no conditional ranking is available.', '']
            continue
        text += opponent_tables(chapter['rankings']['opponent'], chapter_top, chapter=True)
        text += ['### Our selected moves', '']+table(chapter['rankings']['own'], chapter_top, chapter=True, own=True)
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


def write_outputs(report, path, top=20, chapter_top=5):
    path = Path(path)
    path.with_suffix('.vulnerabilities.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    path.with_suffix('.vulnerabilities.md').write_text(markdown(report, top, chapter_top), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', help='Saved score JSON files, with unchanged source PGNs')
    parser.add_argument('--cache', default='.cache/explorer')
    parser.add_argument('--fetch-missing', action='store_true', help='Fetch only missing own decision parent tables; default is cache-only')
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--chapter-top', type=int, default=5)
    parser.add_argument('--summary', help='Combined summary Markdown; defaults to first report directory/vulnerabilities.md')
    args = parser.parse_args()
    if args.top < 1 or args.chapter_top < 1:
        parser.error('Ranking lengths must be positive')
    results = []
    for path in args.reports:
        result = analyze(path, args.cache, args.fetch_missing)
        write_outputs(result, path, args.top, args.chapter_top)
        results.append((result, Path(path)))
        print(f"Generated {result['color']} vulnerabilities: overall and {len(result['chapters'])} chapters", flush=True)
    summary = ['# Repertoire vulnerability summary', '',
               'Largest weighted deficits, separately for opponent replies and our moves. '
               'The benchmarks differ and the rows overlap, so do not sum them or interpret drag as an achievable improvement. '
               'See each full report for definitions, sample caveats, all chapters and exact evidence.', '']
    for report, path in results:
        summary += [f"## {report['color'].title()}", '',
                    f"Repertoire score: {pct(report['overall_score'])}; {report['color']} starting baseline: "
                    f"{pct(report['starting_baseline_score'])}; difference: {delta(report['overall_delta_pp'])}.", '',
                    f"[Full overall and chapter report]({path.with_suffix('.vulnerabilities.md').name})", '']
        summary += opponent_tables(report['overall']['rankings']['opponent'], min(args.top, 10))
        summary += ['### Our selected moves', '']+table(report['overall']['rankings']['own'], min(args.top, 10), own=True)
    target = Path(args.summary) if args.summary else Path(args.reports[0]).parent/'vulnerabilities.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(summary), encoding='utf-8')


if __name__ == '__main__':
    main()
