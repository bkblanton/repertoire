"""Cache-only preparation reuse, reply predictability and boundary board profiles."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import chess

from .explorer import Explorer, counts
from .graph import key, parse
from .model import score
from .preparation import Evaluator, chess_facts, position_lines, stopping_rows
from .attribution import enrich, chapter_text, ATTRIBUTION_NOTE
from .gaps import distribution as gap_distribution


DEFAULT_GAMES = (10, 50, 100, 500)
FEATURE_LABELS = {f'{side}_{feature}': f'{owner}: {label}'
                  for side,owner in [('own','Our side'),('opponent','Opponent')]
                  for feature,label in [('isolated','at least one isolated pawn'),
                                        ('doubled','at least one doubled pawn file'),
                                        ('passed','at least one passed pawn'),
                                        ('isolated_d_pawn','isolated d-pawn'),
                                        ('bishop_pair','bishop pair')]}


def entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def exposure(probability, games):
    """Stable probability of at least one encounter in independent games."""
    if not 0 <= probability <= 1 or games < 0:
        raise ValueError('Invalid encounter probability or game count')
    if games == 0: return 0.0
    if probability == 1: return 1.0
    return -math.expm1(games * math.log1p(-probability))


def reuse_metrics(decisions, games=DEFAULT_GAMES):
    if len({(r['position'], r['move']) for r in decisions}) != len(decisions):
        raise ValueError('Duplicate canonical decisions would inflate reuse')
    encounters = sum(r['reach'] for r in decisions)
    curve = []
    for n in games:
        distinct = sum(exposure(r['reach'], n) for r in decisions)
        total = n * encounters
        curve.append(dict(games=n, expected_encounters=total, expected_distinct_decisions=distinct,
                          expected_unseen_decisions=len(decisions)-distinct,
                          expected_repeat_encounters=total-distinct,
                          fraction_of_reachable_decisions_seen=distinct/len(decisions) if decisions else None))
    return dict(reachable_distinct_decisions=len(decisions), expected_encounters_per_game=encounters,
                curve=curve, decisions=sorted(decisions, key=lambda r: (-r['reach'], r['position'], r['move'])))


def predictability_metrics(evaluator, reach, lines):
    rows = []
    opportunities = recorded_opportunities = bits = sparse_opportunities = 0.0
    for k, mass in reach.items():
        node = evaluator.graph.nodes[k]
        if mass <= 0 or not node.edges or evaluator.facts[k]['turn'] == evaluator.color or evaluator.facts[k]['outcome'] is not None:
            continue
        opportunities += mass
        data = evaluator.evidence[k]
        total = sum(counts(data))
        observed = [(r['uci'], sum(counts(r))) for r in data['moves'] if sum(counts(r))]
        recorded = sum(n for _, n in observed)
        coverage = recorded/total if total else 0.0
        h = entropy(n/recorded for _, n in observed) if recorded else None
        board = chess.Board(node.fen)
        replies = sorted([dict(move=m, san=board.san(chess.Move.from_uci(m)), observations=n,
                               probability_given_recorded_reply=n/recorded) for m,n in observed],
                         key=lambda r: (-r['observations'], r['move']))
        weighted = mass * coverage
        recorded_opportunities += weighted
        bits += weighted * (h or 0)
        sparse_opportunities += weighted * (recorded < evaluator.sparse)
        rows.append(dict(position=k, line=lines[k], reach=mass, observations=total,
                         recorded_reply_observations=recorded, recorded_fraction=coverage,
                         entropy_bits=h, effective_replies=2**h if h is not None else None,
                         entropy_contribution_bits=weighted*(h or 0),
                         sparse=recorded < evaluator.sparse, replies=replies))
    mean = bits/recorded_opportunities if recorded_opportunities else None
    return dict(expected_opponent_opportunities=opportunities,
                expected_recorded_reply_opportunities=recorded_opportunities,
                recorded_reply_coverage=recorded_opportunities/opportunities if opportunities else None,
                mean_entropy_bits=mean, effective_replies=2**mean if mean is not None else None,
                expected_reply_information_bits=bits,
                sparse_recorded_opportunity_fraction=sparse_opportunities/recorded_opportunities if recorded_opportunities else None,
                positions=sorted(rows, key=lambda r: (-r['entropy_contribution_bits'], -r['reach'], r['position'])))


def pawn_properties(board, color):
    pawns = list(board.pieces(chess.PAWN, color))
    files = {chess.square_file(s) for s in pawns}
    isolated = [s for s in pawns if not ({chess.square_file(s)-1, chess.square_file(s)+1} & files)]
    doubled = len(files) < len(pawns)
    enemies = list(board.pieces(chess.PAWN, not color))
    passed = [s for s in pawns if not any(abs(chess.square_file(e)-chess.square_file(s)) <= 1
              and (chess.square_rank(e)-chess.square_rank(s))*(1 if color else -1) > 0 for e in enemies)]
    return dict(isolated=bool(isolated), doubled=doubled, passed=bool(passed),
                isolated_d_pawn=any(chess.square_file(s) == 3 for s in isolated))


def board_fingerprint(position, color):
    board = chess.Board(position + ' 0 1')
    own, opponent = pawn_properties(board, color), pawn_properties(board, not color)
    queens = (bool(board.pieces(chess.QUEEN, color)), bool(board.pieces(chess.QUEEN, not color)))
    def wing(c):
        square = board.king(c)
        if square is None: return 'missing'
        file = chess.square_file(square)
        return 'queenside' if file <= 2 else 'kingside' if file >= 5 else 'central'
    wings = (wing(color), wing(not color))
    king_relation = 'opposite wings' if set(wings) == {'queenside', 'kingside'} else 'same wing' if wings[0] == wings[1] and wings[0] in ('queenside', 'kingside') else 'at least one central king'
    def skeleton(c):
        return ' '.join(chess.square_name(s) for s in sorted(board.pieces(chess.PAWN, c))) or 'none'
    def material(c):
        return ' '.join(f'{chess.piece_symbol(t).upper()}{len(board.pieces(t,c))}' for t in (chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT,chess.PAWN))
    categories = dict(queens={(True,True):'both queens', (True,False):'own queen only',
                             (False,True):'opponent queen only', (False,False):'no queens'}[queens],
                      king_placement=king_relation, total_pawns=str(len(board.pieces(chess.PAWN,True))+len(board.pieces(chess.PAWN,False))),
                      total_rooks=str(len(board.pieces(chess.ROOK,True))+len(board.pieces(chess.ROOK,False))),
                      material=f'Own {material(color)} / Opponent {material(not color)}',
                      pawn_structure=f'White: {skeleton(True)}; Black: {skeleton(False)}')
    features = {f'{side}_{feature}': value for side, props in [('own',own),('opponent',opponent)] for feature,value in props.items()}
    features.update(own_bishop_pair=len(board.pieces(chess.BISHOP,color)) >= 2,
                    opponent_bishop_pair=len(board.pieces(chess.BISHOP,not color)) >= 2)
    return categories, features


def position_profile(stops, color):
    mass = sum(r['reach'] for r in stops)
    categories, features, examples = defaultdict(lambda: defaultdict(float)), defaultdict(float), {}
    for row in stops:
        cats, flags = board_fingerprint(row['position'], color)
        for name, value in cats.items():
            categories[name][value] += row['reach']
            previous = examples.get((name,value))
            if previous is None or row['reach'] > previous['reach']:
                examples[name,value] = row
        for name, flag in flags.items():
            features[name] += row['reach']*flag
    distributions = {name: [dict(value=value, probability=p/mass, example_line=examples[name,value]['line'],
                                  example_position=examples[name,value]['position'])
                            for value,p in sorted(values.items(), key=lambda item: (-item[1],item[0]))]
                     for name,values in categories.items()} if mass else {}
    return dict(scope_mass=mass, distributions=distributions,
                features={name:p/mass for name,p in features.items()} if mass else {},
                distinct_pawn_structures=len(categories['pawn_structure']),
                effective_pawn_structures=2**entropy(p/mass for p in categories['pawn_structure'].values()) if mass else None)


def position_reach_rows(evaluator, starts, reach, lines):
    """Canonical reached boards, including the first unprepared opponent reply."""
    paths, boards, best = {}, {}, {}
    for k, weight in starts.items():
        if weight <= 0: continue
        paths[k] = '' if lines[k] == '(PGN root)' else lines[k]
        boards[k] = chess.Board(evaluator.graph.nodes[k].fen)
        best[k] = weight

    def after_move(k, move):
        board = boards[k]
        label = f'{board.fullmove_number}{"." if board.turn else "..."}{board.san(chess.Move.from_uci(move))}'
        after = board.copy()
        after.push_uci(move)
        return (paths[k] + ' ' + label).strip(), after

    for k in reversed(evaluator.values):
        if k not in paths: continue
        for move, probability, target in evaluator.edges[k]:
            route_mass = best[k] * probability
            if probability <= 0 or route_mass <= best.get(target, -1): continue
            path, after = after_move(k, move)
            assert key(after) == target
            paths[target], boards[target], best[target] = path, after, route_mass
    rows = {}
    for k, mass in reach.items():
        if mass <= 0: continue
        if evaluator.facts[k]['outcome'] is not None:
            kind = 'terminal'
        elif any(s[2] == 'unresolved_distribution' for s in evaluator.stops[k]):
            kind = 'unresolved_distribution'
        elif not evaluator.graph.nodes[k].edges:
            kind = 'theory_leaf'
        else:
            kind = 'own_move' if evaluator.facts[k]['turn'] == evaluator.color else 'opponent_reply'
        data = evaluator.evidence.get(k)
        sample = counts(data) if data is not None else None
        value = evaluator.values[k]
        rows[k] = dict(position=k, line=paths[k] or '(PGN root)', reach=mass,
                       to_move='white' if evaluator.facts[k]['turn'] else 'black', kind=kind,
                       is_starting_position=k == key(chess.Board()),
                       repertoire_score=float(value[0]) if value[1] == 0 else None,
                       database_score=score(sample, evaluator.color) if sample is not None else None,
                       games=sum(sample) if sample is not None else None,
                       games_source='position' if sample is not None else 'unavailable',
                       counts_white_draw_black=sample)
    # The cached parent table supplies both the reply and its probability. No
    # evidence for the reached child board, or for its next moves, is requested.
    deviation_best = {}
    for k, mass in reach.items():
        if mass <= 0: continue
        for move, probability, kind, sample, fixed in evaluator.stops[k]:
            if kind != 'deviation' or probability <= 0: continue
            path, after = after_move(k, move)
            target = key(after)
            assert target not in evaluator.graph.nodes
            row = rows.setdefault(target, dict(position=target, line=path, reach=0.,
                to_move='white' if after.turn else 'black',
                kind='terminal' if fixed is not None else 'unprepared_reply',
                is_starting_position=target == key(chess.Board()), unprepared_origins=[],
                repertoire_score=None, database_score=0., games=0, games_source='parent_move_rows',
                counts_white_draw_black=[0,0,0]))
            branch_reach = mass * probability
            database_score = fixed if fixed is not None else score(sample, evaluator.color)
            assert database_score is not None  # a positive reply probability has observations
            row['reach'] += branch_reach
            row['database_score'] += branch_reach * database_score
            row['games'] += sum(sample)
            row['counts_white_draw_black'] = [a+b for a,b in zip(row['counts_white_draw_black'],sample)]
            row['unprepared_origins'].append(dict(parent_position=k, move=move, reach=branch_reach,
                database_score=database_score, games=sum(sample), counts_white_draw_black=sample))
            route_mass = best[k] * probability
            if route_mass > deviation_best.get(target, -1):
                row['line'] = path
                deviation_best[target] = route_mass
    for row in rows.values():
        if row.get('unprepared_origins'):
            row['database_score'] /= row['reach']
        if row['reach'] > 1 + 1e-10:
            raise AssertionError('Canonical position revisited in an acyclic graph')
        row['reach'] = min(1., row['reach'])
    return sorted(rows.values(), key=lambda r: (-r['reach'], len(r['line'].split()), r['line'], r['position']))


def scope_metrics(evaluator, starts, lines, games=DEFAULT_GAMES, entry_probability=1.0):
    reach = evaluator.reaches(starts)
    value = evaluator.evaluate(starts)
    decisions = []
    for k, mass in reach.items():
        if mass <= 0 or evaluator.facts[k]['turn'] != evaluator.color: continue
        board = chess.Board(evaluator.graph.nodes[k].fen)
        for move,p,_ in evaluator.edges[k]:
            probability = mass*p
            if probability > 1+1e-10: raise AssertionError('Decision revisited in an acyclic graph')
            decisions.append(dict(position=k, move=move, san=board.san(chess.Move.from_uci(move)),
                                  line=lines[k], reach=min(1.0,probability)))
    reuse = reuse_metrics(decisions, games)
    if not math.isclose(reuse['expected_encounters_per_game'], value[2], abs_tol=1e-10):
        raise AssertionError('Own-decision reach does not reproduce prepared depth')
    stops = stopping_rows(evaluator, starts, None, lines)
    profiles = {'all': position_profile(stops, evaluator.color)}
    for kind in sorted({r['type'] for r in stops}):
        profiles[kind] = position_profile([r for r in stops if r['type'] == kind], evaluator.color)
    unknown = sum(r['reach'] for r in stops if r['type'] == 'unresolved_distribution')
    return dict(positions=position_reach_rows(evaluator,starts,reach,lines),
                gap_coverage=gap_distribution(evaluator, starts, entry_probability),
                reuse=reuse, predictability=predictability_metrics(evaluator,reach,lines),
                position_profiles=profiles, stopping_outcomes=stops,
                unresolved_opponent_distribution_mass=unknown,
                validation=dict(resolved_score=float(value[0]), unresolved_score_mass=float(value[1]),
                                stopping_mass=sum(r['reach'] for r in stops), depth_reproduced=True))


def analyze(path, cache='.cache/explorer', games=DEFAULT_GAMES):
    path = Path(path)
    source_bytes = path.read_bytes()
    saved = json.loads(source_bytes)
    manifest = saved['manifest']
    source = Path(manifest['input_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('PGN changed since scoring; regenerate scores first')
    graph = parse(source, manifest['configuration'].get('exclude', []))
    color = saved['color'] == 'white'
    evidence, missing = {}, []
    explorer = Explorer(cache,manifest['filters'],offline=True)
    try:
        for k in graph.nodes:
            try: evidence[k] = explorer.get(k)
            except ValueError as exc:
                if not str(exc).startswith('Offline cache miss:'): raise
                missing.append(k)
            if k in manifest['evidence'] and explorer.provenance.get(k) != manifest['evidence'][k]:
                raise ValueError('Saved score evidence changed; regenerate scores first')
    finally:
        explorer.close()
    facts = chess_facts(graph,color,evidence)
    lines = position_lines(graph)
    scopes = [dict(id='overall', name='Overall repertoire', starts=manifest['root_weights'],
                   policy_basis='overall policy', chapter=None, entry_probability=1.0,
                   overall_policy_entry_probability=1.0, expected=saved['overall'])]
    for c in saved['chapters']:
        weights = c['score'].get('first_entry_weights', {})
        if not weights and len(c['entries']) == 1: weights = {c['entries'][0]['position']:1.0}
        scopes.append(dict(id=c['id'],name=c['name'],starts={k:w for k,w in weights.items() if w},
                           policy_basis=c.get('policy_basis','overall policy'),chapter=c['id'],
                           entry_probability=c['score'].get('entry_probability'),
                           overall_policy_entry_probability=c['score'].get('overall_policy_entry_probability'),
                           expected=c['score']))
    for scope in scopes:
        expected = scope.pop('expected')
        if not scope['starts']:
            scope['status'] = 'unavailable: unresolved entry weights'
            continue
        evaluator = Evaluator(graph,color,evidence,facts,manifest['configuration'].get('policy',{}),
                              chapter=scope['chapter'],sparse=manifest['sparse_threshold'])
        scope.update(scope_metrics(evaluator,scope['starts'],lines,games,scope['entry_probability']))
        validation = scope['validation']
        for actual, desired in [(validation['resolved_score'],expected['resolved_contribution']),
                                (validation['unresolved_score_mass'],expected['unresolved_mass'])]:
            if not math.isclose(actual,desired,abs_tol=1e-10):
                raise AssertionError('Character report did not reproduce saved scope score')
        depth = expected.get('prepared_depth',{}).get('expected_moves')
        if depth is not None and not math.isclose(depth,scope['reuse']['expected_encounters_per_game'],abs_tol=1e-10):
            raise AssertionError('Character report did not reproduce saved prepared depth')
        scope['status'] = 'partial: unknown opponent distribution' if scope['unresolved_opponent_distribution_mass'] else 'resolved'
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('PGN changed during character analysis')
    result = dict(color=saved['color'],scopes=scopes,manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
                report_path=str(path.resolve()),report_sha256=hashlib.sha256(source_bytes).hexdigest(),
                input_path=str(source),input_sha256=manifest['input_sha256'],filters=manifest['filters'],
                cache_only=True,network_requests=0,evidence=explorer.provenance,uncached_positions=missing,
                games=list(games),source_pgn_unchanged=True, gap_schema_version=1,
                gap_definition='First unanswered own-turn board under the selected policy, including cached opponent replies after prepared endpoints and all exact transpositions. Aggregate first-exit mass by board before sqrt(sum(p**2)); unknown distributions remain bounded.',
                weighted_gap_definition='Chapter entry probability multiplied by conditional equivalent gap reach. Entry weights and comparison policy match the chapter score. Overlapping chapters and shared gap boards make these weights non-additive.'))
    return enrich(result, graph)


def pct(value):
    return 'unavailable' if value is None else f'{100*value:.2f}%'


def num(value):
    return 'unavailable' if value is None else f'{value:.2f}'


def render(result, top=8):
    catalog = result.get('chapter_catalog',[])
    text = [f'# {result["color"].title()} repertoire character', '',
        'Expected reuse, opponent reply predictability and positions at the end of preparation. All estimates use cached Lichess population data with the saved repertoire policy, not personal game forecasts.', '',
        'Overall figures start at the repertoire roots. Chapter figures start at the saved weighted first-entry positions and include subsequent shared continuations. An N-game chapter curve means N games entering that chapter under its comparison policy. Alternative chapters remain separate comparisons; overlapping chapters must not be added.', '',
        ATTRIBUTION_NOTE, '', '## Chapter comparison', '',
        '| Chapter | Entry probability | Overall-policy entry | Distinct decisions | Own decisions / game | Effective replies | Effective pawn structures |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in result['scopes']:
        if 'reuse' not in s: continue
        label=s['name']+(' (alternative)' if s['policy_basis'] != 'overall policy' else '')
        text.append(f"| {label} | {pct(s['entry_probability'])} | {pct(s['overall_policy_entry_probability'])} | {s['reuse']['reachable_distinct_decisions']} | {num(s['reuse']['expected_encounters_per_game'])} | {num(s['predictability']['effective_replies'])} | {num(s['position_profiles']['all']['effective_pawn_structures'])} |")
    for s in result['scopes']:
        text += ['',f'## {s["name"]}', '', f"Policy: {s['policy_basis']}. Status: {s['status']}.", '']
        if 'reuse' not in s: continue
        r,p,profile=s['reuse'],s['predictability'],s['position_profiles']['all']
        text += ['### Expected reuse', '',
                 f"{r['reachable_distinct_decisions']} distinct reachable own decisions; {num(r['expected_encounters_per_game'])} expected encounters per game. Exact transpositions share one decision.", '',
                 '| Games | Distinct decisions encountered | Still unseen | Share encountered | Repeat encounters |', '|---:|---:|---:|---:|---:|']
        for row in r['curve']:
            text.append(f"| {row['games']} | {num(row['expected_distinct_decisions'])} | {num(row['expected_unseen_decisions'])} | {pct(row['fraction_of_reachable_decisions_seen'])} | {num(row['expected_repeat_encounters'])} |")
        text += ['', 'Most frequently revisited decisions (per game in this scope):', '',
                 '| Position reached by | Your move | Source chapters | Encounter probability |', '|---|---|---|---:|']
        for row in r['decisions'][:top if s['id']=='overall' else 3]:
            text.append(f"| {row['line']} | {row['san']} | {chapter_text(row,catalog)} | {pct(row['reach'])} |")
        text += ['', '### Opponent reply predictability', '',
                 f"**{num(p['effective_replies'])} effective replies** per recorded opponent decision; {num(p['mean_entropy_bits'])} bits of average reply entropy. Expected reply information across preparation: {num(p['expected_reply_information_bits'])} bits per game.", '',
                 f"Recorded replies cover {pct(p['recorded_reply_coverage'])} of reached opponent opportunities. Sparse parent samples account for {pct(p['sparse_recorded_opportunity_fraction'])} of recorded opportunities.", '',
                 'Positions contributing the most reply information (reach times recorded fraction times entropy):', '',
                 '| Position reached by | Source chapters | Reach | Effective replies | Most common reply | Reply share | Observations | Bits / game |', '|---|---|---:|---:|---|---:|---:|---:|']
        for row in p['positions'][:top if s['id']=='overall' else 3]:
            common=row['replies'][0] if row['replies'] else {}
            common_source = chapter_text(common,catalog) if common else 'None'
            text.append(f"| {row['line']} | {chapter_text(row,catalog)} | {pct(row['reach'])} | {num(row['effective_replies'])} | {common.get('san','unavailable')}<br>{common_source} | {pct(common.get('probability_given_recorded_reply'))} | {row['recorded_reply_observations']:,}{' (sparse)' if row['sparse'] else ''} | {num(row['entropy_contribution_bits'])} |")
        text += ['', '### Position profile when preparation ends', '',
                 'Includes prepared endpoints and the board after an unprepared opponent reply. These describe the boundary of preparation, not the eventual middlegame. Shares below use all stopping mass; features can overlap.', '',
                 '| Boundary type | Share |', '|---|---:|']
        for kind,sub in s['position_profiles'].items():
            if kind != 'all': text.append(f"| {kind.replace('_',' ')} | {pct(sub['scope_mass'])} |")
        text += ['', '| Position feature | Share |', '|---|---:|']
        for name in ('queens','king_placement'):
            for row in profile['distributions'].get(name,[]):
                text.append(f"| {row['value'].capitalize()} | {pct(row['probability'])} |")
        for name,value in profile['features'].items():
            text.append(f"| {FEATURE_LABELS[name]} | {pct(value)} |")
        text += ['', '| Pawns remaining on the board | Share |', '|---:|---:|']
        for row in profile['distributions'].get('total_pawns',[]):
            text.append(f"| {row['value']} | {pct(row['probability'])} |")
        text += ['', f"{profile['distinct_pawn_structures']} observed boundary pawn skeletons; **{num(profile['effective_pawn_structures'])} effective skeletons** after weighting their frequencies.", '',
                 '| Pawn skeleton | Share | Example route | Chapter source / context |', '|---|---:|---|---|']
        for row in profile['distributions'].get('pawn_structure',[])[:top if s['id']=='overall' else 3]:
            text.append(f"| {row['value']} | {pct(row['probability'])} | {row['example_line']} | {chapter_text({'chapter_attribution':row.get('example_chapter_attribution')},catalog)} |")
    text += ['', '## Definitions and limits', '',
        '- For decision probability p, expected encounters over N independent games are N*p; probability of seeing it at least once is 1-(1-p)^N. Summing gives distinct decisions encountered. Repeats equal total encounters minus distinct encounters. Only selected decisions with positive empirical reach count. This is exposure, not memory retention.',
        '- Reply entropy is -sum(p*log2(p)) over observed named replies at each active opponent decision. Effective replies are 2 to the power of the reach-and-recorded-fraction-weighted mean entropy, a geometric rather than arithmetic average of local effective reply counts. No move popularity is applied to our forced choices. PGN leaves are excluded because preparation has ended.',
        '- Unrecorded continuation mass is excluded from named-reply entropy and disclosed as missing coverage. Zero observations are unavailable, never perfect predictability. With unknown opponent distributions, reuse describes only the known prefix. Sample flags reuse the scoring report threshold; no threshold filters the metrics.',
        '- A pawn skeleton is the exact set of White and Black pawn squares, ignoring other pieces and move order. Equivalent plans with different pawn squares remain separate. Effective skeletons equal 2 to the power of skeleton entropy.',
        '- King wings mean current king files a-c or f-h; d-e are central. This is not a castling-history claim. Isolated pawns lack friendly pawns on adjacent files; doubled means more than one on a file; passed means no enemy pawn ahead on the same or adjacent files; isolated d-pawn means an isolated pawn currently on file d. Bishop pair means at least two bishops.',
        '- Board profiles are also separated by stopping type in JSON, which retains every decision, reply distribution, stopping position, material/pawn-count distribution and pawn skeleton. Representative routes label positions; reach combines all modeled transpositions and is not a historical full-line frequency.',
        '- These are descriptive empirical estimates, without sampling confidence intervals. Filters combine blitz, rapid and classical and all selected rating buckets. No new API requests are made; PGNs and existing score reports must match their saved evidence.', '']
    return '\n'.join(text)


def summary(results):
    text=['# Repertoire character overview', '', 'Reuse, predictability and boundary position profiles from cached data.', '',
          '| Color | Reachable own decisions | Own decisions / game | Effective opponent replies | Effective pawn skeletons | Details |', '|---|---:|---:|---:|---:|---|']
    for result in results:
        s=result['scopes'][0]
        text.append(f"| {result['color'].title()} | {s['reuse']['reachable_distinct_decisions']} | {num(s['reuse']['expected_encounters_per_game'])} | {num(s['predictability']['effective_replies'])} | {num(s['position_profiles']['all']['effective_pawn_structures'])} | [Overall and chapters]({result['report_filename']}) |")
    text += ['', '## Expected exposure', '', '| Color | Games | Distinct decisions encountered | Still unseen | Share encountered | Repeat encounters |', '|---|---:|---:|---:|---:|---:|']
    for result in results:
        for r in result['scopes'][0]['reuse']['curve']:
            text.append(f"| {result['color'].title()} | {r['games']} | {num(r['expected_distinct_decisions'])} | {num(r['expected_unseen_decisions'])} | {pct(r['fraction_of_reachable_decisions_seen'])} | {num(r['expected_repeat_encounters'])} |")
    text += ['', 'An encounter is a selected own position/move decision. Exact transpositions share one decision. Exposure assumes independent games under the current empirical policy. Effective replies summarize branching while preparation remains active; effective skeletons describe frequency-weighted diversity where it ends. Lower values are not inherently better. Full definitions, evidence coverage and chapter comparisons are in the linked reports.', '']
    text += ['## Positions where preparation ends', '',
             '| Color | Both queens | All 16 pawns | Our isolated pawns | Our doubled pawns | Our bishop pair | Unprepared deviation |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for result in results:
        profiles=result['scopes'][0]['position_profiles']; profile=profiles['all']
        def probability(name,value):
            return next((r['probability'] for r in profile['distributions'][name] if r['value']==value),0)
        text.append(f"| {result['color'].title()} | {pct(probability('queens','both queens'))} | {pct(probability('total_pawns','16'))} | {pct(profile['features']['own_isolated'])} | {pct(profile['features']['own_doubled'])} | {pct(profile['features']['own_bishop_pair'])} | {pct(profiles.get('deviation',{}).get('scope_mass',0))} |")
    text += ['', 'Isolated/doubled percentages mean at least one such pawn/file, not the number of pawns. These are preparation boundaries, often early in the opening; they do not predict eventual middlegame structures.', '',
             '## Chapter reply predictability', '',
             'The three lowest and highest effective reply counts for each color, among chapters using the overall policy. Figures are conditional on chapter entry, with overlapping chapters retained separately. A low count means replies concentrate on fewer moves, not that the position is easier or better.', '',
             '| Color | Reply concentration | Chapter | Entry probability | Effective replies | Own decisions after entry |',
             '|---|---|---|---:|---:|---:|']
    for result in results:
        candidates=sorted([s for s in result['scopes'][1:] if s.get('policy_basis')=='overall policy'
                           and s.get('predictability',{}).get('effective_replies') is not None],
                          key=lambda s:s['predictability']['effective_replies'])
        selections = [('Most concentrated', candidates[:3]), ('Most spread out', candidates[-3:][::-1])]
        for label,selected in selections:
            for s in selected:
                text.append(f"| {result['color'].title()} | {label} | {s['name']} | {pct(s['entry_probability'])} | {num(s['predictability']['effective_replies'])} | {num(s['reuse']['expected_encounters_per_game'])} |")
    text += ['', 'The full reports retain alternative chapters, all reply samples and coverage, the remaining learning-curve points and detailed board profiles. All computations use the cache; no PGNs are modified.', '']
    return '\n'.join(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+',type=Path)
    parser.add_argument('--cache',default='.cache/explorer')
    parser.add_argument('--games',nargs='+',type=int,default=list(DEFAULT_GAMES))
    args=parser.parse_args()
    if any(n <= 0 for n in args.games): parser.error('Games must be positive')
    for path in args.reports:
        result=analyze(path,args.cache,sorted(set(args.games)))
        path.with_suffix('.character.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(f"{result['color']}: character report generated for {len(result['scopes'])} scopes; no network requests",flush=True)
    from .render import update_report_outputs
    update_report_outputs(args.reports[-1])


if __name__ == '__main__':
    main()
