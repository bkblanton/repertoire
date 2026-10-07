"""Cache-only preparation reuse, reply predictability and boundary board profiles."""
from collections import defaultdict
from functools import lru_cache
import math

import chess
import numpy as np

from .status import Status
from .board_cache import STARTING_POSITION, children, fen_number, route_line, san, turn
from .context import DEFAULT_CACHE, AnalysisContext, stage_main
from .explorer import counts
from .model import score
from .preparation import Evaluators, chess_facts, position_lines, stopping_rows
from .attribution import enrich
from .gaps import distribution as gap_distribution
from .sharpness import recursive_wdl, scope_outcomes, stopping_wdl, summarize as summarize_outcomes


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
        if mass <= 0 or evaluator.facts[k]['turn'] == evaluator.color or evaluator.facts[k]['outcome'] is not None:
            continue
        opportunities += mass
        data = evaluator.evidence[k]
        total = sum(counts(data))
        observed = [(r['uci'], sum(counts(r))) for r in data['moves'] if sum(counts(r))]
        recorded = sum(n for _, n in observed)
        coverage = recorded/total if total else 0.0
        h = entropy(n/recorded for _, n in observed) if recorded else None
        replies = sorted([dict(move=m, san=san(k, m), observations=n,
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
    categories, features = _board_fingerprint(position, color)
    return dict(categories), dict(features)


@lru_cache(maxsize=32768)
def _board_fingerprint(position, color):
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
    return tuple(categories.items()), tuple(features.items())


def position_profile(stops, color):
    mass = sum(r['reach'] for r in stops)
    categories, features, examples = defaultdict(lambda: defaultdict(float)), defaultdict(float), {}
    for row in stops:
        cats, flags = board_fingerprint(row['position'], color)
        for name, value in cats.items():
            categories[name][value] += row['reach']
            previous = examples.get((name,value))
            # The most likely example; exact ties go to the first line, so the choice is independent of traversal order.
            if previous is None or (-row['reach'], row['line'], row['position']) < (-previous['reach'], previous['line'], previous['position']):
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


def position_reach_rows(evaluator, starts, reach, lines, wdl_values=None):
    """Canonical reached boards, including the first unprepared opponent reply."""
    if wdl_values is None:
        wdl_values = recursive_wdl(evaluator)
    routes = evaluator.routes(starts, replies=True)

    def line(k):
        # Each route keeps its own full-move number: transposed routes can differ in length.
        _, root, moves = routes[k]
        text, position, _ = route_line(root, fen_number(evaluator.graph.nodes[root].fen), moves)
        if position != k:
            raise AssertionError('Repertoire route does not reach its board')
        prefix = '' if lines[root] == '(PGN root)' else lines[root]
        return (prefix + ' ' + text).strip() or '(PGN root)'

    rows = {}
    for k, mass in reach.items():
        if mass <= 0: continue
        if evaluator.facts[k]['outcome'] is not None:
            kind = 'terminal'
        elif any(s.kind == 'unresolved_distribution' for s in evaluator.stops[k]):
            kind = 'unresolved_distribution'
        elif evaluator.facts[k]['turn'] == evaluator.color and not evaluator.graph.nodes[k].edges:
            kind = 'theory_leaf'
        else:
            kind = 'own_move' if evaluator.facts[k]['turn'] == evaluator.color else 'opponent_reply'
        data = evaluator.evidence.get(k)
        sample = counts(data) if data is not None else None
        value = evaluator.values[k]
        rows[k] = dict(position=k, line=line(k), reach=mass,
                       to_move='white' if evaluator.facts[k]['turn'] else 'black', kind=kind,
                       is_starting_position=k == STARTING_POSITION,
                       repertoire_score=float(value[0]) if value[1] == 0 else None,
                       outcomes=summarize_outcomes(wdl_values[k]),
                       database_score=score(sample, evaluator.color) if sample is not None else None,
                       games=sum(sample) if sample is not None else None,
                       games_source='position' if sample is not None else 'unavailable',
                       counts_white_draw_black=sample)
    # The cached parent table supplies both the reply and its probability. No
    # evidence for the reached child board, or for its next moves, is requested.
    deviation_wdl = {}
    for k, mass in reach.items():
        if mass <= 0: continue
        for move, probability, kind, sample, fixed in evaluator.stops[k]:
            if kind != 'deviation' or probability <= 0: continue
            target = children(k)[move]
            assert target not in evaluator.graph.nodes
            row = rows.setdefault(target, dict(position=target, line=line(target), reach=0.,
                to_move='white' if turn(target) else 'black',
                kind='terminal' if fixed is not None else 'unprepared_reply',
                is_starting_position=target == STARTING_POSITION, unprepared_origins=[],
                repertoire_score=None, database_score=0., games=0, games_source='parent_move_rows',
                counts_white_draw_black=[0,0,0]))
            branch_reach = mass * probability
            database_score = fixed if fixed is not None else score(sample, evaluator.color)
            assert database_score is not None  # a positive reply probability has observations
            row['reach'] += branch_reach
            row['database_score'] += branch_reach * database_score
            row['games'] += sum(sample)
            row['counts_white_draw_black'] = [a+b for a,b in zip(row['counts_white_draw_black'],sample)]
            local_wdl = stopping_wdl(sample, evaluator.color, fixed)
            deviation_wdl.setdefault(target, np.zeros(4))
            deviation_wdl[target] += branch_reach * local_wdl
            row['unprepared_origins'].append(dict(parent_position=k, move=move, reach=branch_reach,
                database_score=database_score, games=sum(sample), counts_white_draw_black=sample,
                outcomes=summarize_outcomes(local_wdl)))
    for row in rows.values():
        if row.get('unprepared_origins'):
            row['database_score'] /= row['reach']
            row['outcomes'] = summarize_outcomes(deviation_wdl[row['position']] / row['reach'])
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
        for move,p,_ in evaluator.edges[k]:
            probability = mass*p
            if probability > 1+1e-10: raise AssertionError('Decision revisited in an acyclic graph')
            decisions.append(dict(position=k, move=move, san=san(k, move),
                                  line=lines[k], reach=min(1.0,probability)))
    reuse = reuse_metrics(decisions, games)
    if not math.isclose(reuse['expected_encounters_per_game'], value[2], abs_tol=1e-10):
        raise AssertionError('Own-decision reach does not reproduce prepared depth')
    stops = stopping_rows(evaluator, starts, None, lines)
    profiles = {'all': position_profile(stops, evaluator.color)}
    for kind in sorted({r['type'] for r in stops}):
        profiles[kind] = position_profile([r for r in stops if r['type'] == kind], evaluator.color)
    unknown = sum(r['reach'] for r in stops if r['type'] == 'unresolved_distribution')
    wdl_values = recursive_wdl(evaluator)
    positions = position_reach_rows(evaluator, starts, reach, lines, wdl_values)
    gaps = gap_distribution(evaluator, starts, entry_probability)
    unanswered = {r['position']: r['reach'] for r in positions
        if r['to_move'] == ('white' if evaluator.color else 'black')
        and r['kind'] in ('theory_leaf', 'unprepared_reply')}
    first_gaps = {r['position']: r['reach'] for r in gaps['gaps']}
    if set(unanswered) != set(first_gaps) or any(
            not math.isclose(mass, first_gaps[k], abs_tol=1e-10) for k, mass in unanswered.items()):
        raise AssertionError('Unanswered position reach must equal canonical first-gap reach')
    return dict(positions=positions,
                outcomes=scope_outcomes(evaluator, starts, wdl_values),
                gap_coverage=gaps,
                reuse=reuse, predictability=predictability_metrics(evaluator,reach,lines),
                position_profiles=profiles, stopping_outcomes=stops,
                unresolved_opponent_distribution_mass=unknown,
                validation=dict(resolved_score=float(value[0]), unresolved_score_mass=float(value[1]),
                                stopping_mass=sum(r['reach'] for r in stops), depth_reproduced=True,
                                unanswered_reach_matches_first_gaps=True))


def analyze(path, cache=DEFAULT_CACHE, games=DEFAULT_GAMES):
    analysis = AnalysisContext(path)
    graph, color, manifest = analysis.graph, analysis.color, analysis.manifest
    evidence = analysis.read_evidence(cache)
    facts = chess_facts(graph,color,evidence)
    lines = position_lines(graph)
    scopes = analysis.scopes()
    evaluators = Evaluators(graph, color, evidence, facts, manifest['configuration'].get('policy', {}),
                            manifest['sparse_threshold'])
    for scope in scopes:
        expected = scope.pop('score')
        if not scope['starts']:
            scope['status'] = Status.UNRESOLVED_ENTRY_WEIGHTS
            continue
        evaluator = evaluators(scope['chapter'])
        scope.update(scope_metrics(evaluator,scope['starts'],lines,games,scope['entry_probability']))
        validation = scope['validation']
        for actual, desired in [(validation['resolved_score'],expected['resolved_contribution']),
                                (validation['unresolved_score_mass'],expected['unresolved_mass'])]:
            if not math.isclose(actual,desired,abs_tol=1e-10):
                raise AssertionError('Character report did not reproduce saved scope score')
        depth = expected.get('prepared_depth',{}).get('expected_moves')
        if depth is not None and not math.isclose(depth,scope['reuse']['expected_encounters_per_game'],abs_tol=1e-10):
            raise AssertionError('Character report did not reproduce saved prepared depth')
        scope['status'] = Status.UNRESOLVED_OPPONENT_DISTRIBUTION if scope['unresolved_opponent_distribution_mass'] else Status.RESOLVED
    analysis.require_source('character analysis')
    result = dict(color=analysis.saved['color'],scopes=scopes,manifest=analysis.companion_manifest(
                games=list(games),source_pgn_unchanged=True,
                outcome_definition='Owner-relative WDL propagated through the exact score policy and stopping rules, including transpositions, cached parent-row deviations and weighted first entries. Missing outcomes remain unresolved.',
                sharpness_definition='400 * (W + D/4 - (W + D/2)**2), normalized outcome variance on a 0-100 scale. Mix WDL before calculating sharpness; no priors or new API requests.',
                gap_definition='First unanswered own-turn board under the selected policy, including cached opponent replies after prepared endpoints and all exact transpositions. Aggregate first-exit mass by board before sqrt(sum(p**2)); unknown distributions remain bounded.',
                weighted_gap_definition='Chapter entry probability multiplied by conditional equivalent gap reach. Entry weights and comparison policy match the chapter score. Overlapping chapters and shared gap boards make these weights non-additive.'))
    return enrich(result, graph)



def main():
    def configure(parser):
        parser.add_argument('--games',nargs='+',type=int,default=list(DEFAULT_GAMES))

    def options(parser, args):
        if any(n <= 0 for n in args.games): parser.error('Games must be positive')
        return dict(games=sorted(set(args.games)))

    stage_main('character', analyze, __doc__, configure, options)


if __name__ == '__main__':
    main()
