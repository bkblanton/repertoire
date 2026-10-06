"""Exact cache-only depth distributions and first-arrival route examples."""
from collections import defaultdict
import math


from .board_cache import children, fen_number, move_text, next_number, san


ENDING_TYPES = ('prepared_endpoint', 'unprepared_reply', 'game_over', 'other_stop', 'unresolved_distribution')


def depth_distribution(evaluator, starts):
    """Propagate board AND elapsed own-move count, retaining transposed histories.

    Missing leaf scores do not obscure depth. Missing reply distributions leave
    a finite interval between moves already played and the longest continuation.
    """
    if not starts:
        return dict(status='unresolved_entry_weights', unit='own_moves', endings=[], survival=[],
                    expected_moves=None, expected_bounds=None, median_moves=None, median_bounds=None)
    if any(not math.isfinite(w) or w < 0 for w in starts.values()) or not math.isclose(sum(starts.values()), 1., abs_tol=1e-10):
        raise ValueError('Depth distribution requires normalized nonnegative starting weights')
    value = evaluator.evaluate(starts)
    mass = {k: defaultdict(float) for k in evaluator.values}
    for k, weight in starts.items():
        if weight: mass[k][0] += weight
    endings = defaultdict(lambda: dict.fromkeys(ENDING_TYPES, 0.))
    unknown = []
    lengths, active = {}, set()

    def remaining(k):
        if k in lengths: return lengths[k]
        if k in active: raise ValueError('Repertoire contains a reachable cycle')
        active.add(k)
        node, fact = evaluator.graph.nodes[k], evaluator.facts[k]
        if fact['outcome'] is not None or (fact['turn'] == evaluator.color and not node.edges):
            result = 0
        elif fact['turn'] == evaluator.color:
            result = 1 + max(remaining(node.edges[m]) for m in evaluator.own_choices(k))
        else:
            targets = set(node.edges.values()) | (fact['possible_targets'] & evaluator.graph.nodes.keys())
            result = max((remaining(target) for target in targets), default=0)
        active.remove(k)
        lengths[k] = result
        return result

    for k in reversed(evaluator.values):
        node, fact = evaluator.graph.nodes[k], evaluator.facts[k]
        reward = int(bool(node.edges) and fact['turn'] == evaluator.color and fact['outcome'] is None)
        for depth, arrival in mass[k].items():
            for _, probability, target in evaluator.edges[k]:
                mass[target][depth + reward] += arrival * probability
            for _, probability, kind, _, fixed in evaluator.stops[k]:
                probability *= arrival
                if not probability: continue
                ending = ('game_over' if fixed is not None else
                          'prepared_endpoint' if kind == 'theory_leaf' else
                          'unprepared_reply' if kind == 'deviation' else
                          'unresolved_distribution' if kind == 'unresolved_distribution' else 'other_stop')
                endings[depth][ending] += probability
                if ending == 'unresolved_distribution':
                    unknown.append((depth, depth + remaining(k), probability))

    total = sum(sum(row.values()) for row in endings.values())
    if not math.isclose(total, sum(starts.values()), abs_tol=1e-10):
        raise AssertionError('Prepared-depth distribution does not conserve probability')
    known = {d: sum(v for kind, v in row.items() if kind != 'unresolved_distribution')
             for d, row in endings.items()}
    maximum = max([*endings, *[high for _, high, _ in unknown]], default=0)
    survival = []
    for depth in range(maximum + 1):
        completed = sum(p for d, p in known.items() if d >= depth)
        low = completed + sum(p for d, _, p in unknown if d >= depth)
        high = completed + sum(p for _, d, p in unknown if d >= depth)
        survival.append(dict(own_moves=depth, probability=low if low == high else None,
                             bounds=[low, high]))
    low = sum(row['bounds'][0] for row in survival[1:])
    high = sum(row['bounds'][1] for row in survival[1:])
    if not math.isclose(low, float(value[2]), abs_tol=1e-10):
        raise AssertionError('Survival curve does not reproduce expected prepared depth')

    def median(upper):
        distribution = defaultdict(float, known)
        for minimum, maximum, p in unknown:
            distribution[maximum if upper else minimum] += p
        cumulative = 0.
        for depth, p in sorted(distribution.items()):
            cumulative += p
            if cumulative >= .5 - 1e-12: return depth
        return None

    median_bounds = [median(False), median(True)]
    return dict(status='resolved' if low == high else 'unresolved_move_distribution', unit='own_moves',
                expected_moves=low if low == high else None, expected_bounds=[low, high],
                median_moves=median_bounds[0] if median_bounds[0] == median_bounds[1] else None,
                median_bounds=median_bounds,
                unresolved_probability=sum(p for _, _, p in unknown),
                endings=[dict(own_moves=d, **endings[d]) for d in sorted(endings)], survival=survival)


def first_entry_examples(evaluator, roots, region, expected_probability=None, expected_weights=None):
    """Sum ALL first arrivals; retain the most likely genuine route to each board.

    Example-route mass is only a subset of its entry-position mass. Paths stop
    on first chapter arrival, so a bypass example cannot pass an earlier entry.
    """
    evaluator.evaluate(roots)
    region = set(region)
    mass = dict.fromkeys(evaluator.values, 0.)
    best, arrivals, witnesses = {}, {}, {}
    for k, weight in roots.items():
        if weight:
            mass[k] += weight
            best[k] = (weight, k, ())
    stopped = 0.
    for k in reversed(evaluator.values):
        if not mass[k]: continue
        if k in region:
            arrivals[k], witnesses[k] = mass[k], best[k]
            continue
        stopped += mass[k] * sum(stop[1] for stop in evaluator.stops[k])
        probability, root, path = best[k]
        for move, p, target in evaluator.edges[k]:
            mass[target] += mass[k] * p
            candidate = (probability * p, root, (*path, move))
            if target not in best or candidate[0] > best[target][0]: best[target] = candidate
    total = sum(arrivals.values())
    if not math.isclose(total + stopped, sum(roots.values()), abs_tol=1e-10):
        raise AssertionError('First-entry route probabilities do not conserve mass')
    if expected_probability is not None and not math.isclose(total, expected_probability, abs_tol=1e-10):
        raise AssertionError('First-entry examples differ from saved chapter reach')
    if not total:
        return dict(status='no_reachable_entry', entry_probability=total, positions=[])
    if expected_weights is not None:
        for k in set(arrivals) | set(expected_weights):
            if not math.isclose(arrivals.get(k, 0.) / total, expected_weights.get(k, 0.), abs_tol=1e-10):
                raise AssertionError('First-entry examples differ from saved entry weights')
    rows = []
    for k, arrival in sorted(arrivals.items(), key=lambda item: (-item[1], item[0])):
        probability, root, path = witnesses[k]
        position, number = root, fen_number(evaluator.graph.nodes[root].fen)
        sans, text = [], []
        for uci in path:
            if position in region:
                raise AssertionError('Example route passes through an earlier chapter entry')
            sans.append(san(position, uci))
            text.append(move_text(position, number, uci))
            position, number = children(position)[uci], next_number(position, number)
        if position != k:
            raise AssertionError('Example route does not reach its entry board')
        rows.append(dict(position=k, conditional_first_entry_weight=arrival / total,
                         example=dict(root_position=root, root_fen=evaluator.graph.nodes[root].fen,
                                      path_uci=list(path), path_san=sans, line=' '.join(text) or '(PGN root)',
                                      root_probability=probability, conditional_probability=probability / total)))
    return dict(status='resolved', entry_probability=total, positions=rows)
