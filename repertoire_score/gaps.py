"""First unanswered positions, including replies beyond prepared endpoints.

This walk uses cached parent move rows only. Exact transpositions can rejoin
prepared positions in any chapter. Small cyclic components are solved as an
absorbing Markov chain; a closed component remains unresolved.
"""
from collections import defaultdict
import math

import numpy as np

from .explorer import counts


def components(edges):
    """Strongly connected components in source-to-target order, without recursion."""
    seen, finished = set(), []
    for root in edges:
        if root in seen:
            continue
        seen.add(root)
        pending = [(root, iter(edges[root]))]
        while pending:
            node, children = pending[-1]
            target = next(children, None)
            if target is None:
                finished.append(node)
                pending.pop()
            elif target not in seen:
                seen.add(target)
                pending.append((target, iter(edges[target])))
    reverse = defaultdict(list)
    for node, targets in edges.items():
        for target in targets:
            reverse[target].append(node)
    seen, result = set(), []
    for root in reversed(finished):
        if root in seen:
            continue
        seen.add(root)
        pending, group = [root], []
        while pending:
            node = pending.pop()
            group.append(node)
            for target in reverse[node]:
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        result.append(group)
    return result


def distribution(evaluator, starts, entry_probability=1.0):
    """Aggregate unconditional first-gap mass by canonical board before squaring."""
    if not starts:
        return dict(status='unavailable entry weights', equivalent_gap_reach=None,
                    equivalent_gap_reach_bounds=None, weighted_equivalent_gap_reach=None,
                    weighted_equivalent_gap_reach_bounds=None, gaps=[])
    if any(not math.isfinite(w) or w < 0 for w in starts.values()) or not math.isclose(sum(starts.values()), 1., abs_tol=1e-10):
        raise ValueError('Gap distribution requires normalized starting weights')
    if entry_probability is not None and (not math.isfinite(entry_probability) or not 0 <= entry_probability <= 1 + 1e-10):
        raise ValueError('Invalid chapter entry probability')
    graph, facts = evaluator.graph, evaluator.facts
    edges, stops = {}, {}
    pending = [k for k, w in starts.items() if w]
    while pending:
        k = pending.pop()
        if k in edges:
            continue
        node, fact = graph.nodes[k], facts[k]
        transitions, absorptions = defaultdict(float), []
        if fact['outcome'] is not None:
            absorptions.append(('terminal', None, 1.))
        elif fact['turn'] == evaluator.color:
            if node.edges:
                for move, probability in evaluator.own_choices(k).items():
                    transitions[node.edges[move]] += probability
            else:
                absorptions.append(('gap', k, 1.))
        else:
            data = evaluator.evidence.get(k)
            total = sum(counts(data)) if data is not None else 0
            if not total:
                absorptions.append(('unresolved', None, 1.))
            else:
                recorded = 0
                for row in data['moves']:
                    observations = sum(counts(row))
                    recorded += observations
                    if not observations:
                        continue
                    probability = observations / total
                    target, terminal = fact['after'][row['uci']]
                    if terminal is not None:
                        absorptions.append(('terminal', None, probability))
                    elif target in graph.nodes:
                        transitions[target] += probability
                    else:
                        absorptions.append(('gap', target, probability))
                if recorded > total:
                    raise ValueError('Inconsistent cached gap response counts')
                if recorded < total:
                    absorptions.append(('unresolved', None, (total - recorded) / total))
        if not math.isclose(sum(transitions.values()) + sum(p for _, _, p in absorptions), 1., abs_tol=1e-10):
            raise AssertionError('Local gap probability is not conserved')
        edges[k], stops[k] = dict(transitions), absorptions
        pending.extend(transitions)

    incoming = defaultdict(float, starts)
    gaps = defaultdict(float)
    terminal_mass = unresolved_mass = 0.
    cyclic_components = closed_components = 0
    for group in components(edges):
        membership = set(group)
        cyclic = len(group) > 1 or any(k in edges[k] for k in group)
        cyclic_components += int(cyclic)
        arrival = np.asarray([incoming[k] for k in group])
        exits = any(stops[k] or any(t not in membership for t in edges[k]) for k in group)
        if not exits:
            # A closed class cannot produce an identified first gap. Do not
            # interpret a canonical-board cycle as a known repetition draw.
            unresolved_mass += float(arrival.sum())
            closed_components += 1
            continue
        if cyclic:
            index = {k: i for i, k in enumerate(group)}
            matrix = np.eye(len(group))
            for k in group:
                for target, probability in edges[k].items():
                    if target in membership:
                        matrix[index[target], index[k]] -= probability
            flow = np.linalg.solve(matrix, arrival)
        else:
            flow = arrival
        if np.any(flow < -1e-10) or not np.isfinite(flow).all():
            raise AssertionError('Invalid gap visit flow')
        for k, mass in zip(group, flow):
            mass = max(0., float(mass))
            for target, probability in edges[k].items():
                if target not in membership:
                    incoming[target] += mass * probability
            for kind, position, probability in stops[k]:
                weight = mass * probability
                if kind == 'gap':
                    gaps[position] += weight
                elif kind == 'terminal':
                    terminal_mass += weight
                else:
                    unresolved_mass += weight
    gap_mass = sum(gaps.values())
    if not math.isclose(gap_mass + terminal_mass + unresolved_mass, 1., abs_tol=1e-9):
        raise AssertionError('First-gap distribution does not conserve probability')
    if unresolved_mass < 1e-12:
        unresolved_mass = 0.
    largest = max(gaps.values(), default=0.)
    collision = sum(p * p for p in gaps.values())
    # Unknown mass can diffuse over unidentified boards or merge into the
    # largest known gap. These are conservative bounds, without a depth prior.
    bounds = [math.sqrt(collision), math.sqrt(min(1., collision + 2 * largest * unresolved_mass + unresolved_mass ** 2))]
    equivalent = bounds[0] if not unresolved_mass else None
    weighted_bounds = [entry_probability * r for r in bounds] if entry_probability is not None else None
    weighted = (entry_probability * equivalent if entry_probability is not None and equivalent is not None
                else 0. if entry_probability == 0 else None)
    return dict(status='resolved' if not unresolved_mass else 'partial: unknown first-gap reach',
                equivalent_gap_reach=equivalent, equivalent_gap_reach_bounds=bounds,
                known_repeat_gap_probability=collision,
                repeat_gap_probability=collision if not unresolved_mass else None,
                largest_gap_reach=largest, gap_mass=gap_mass, terminal_mass=terminal_mass,
                unresolved_mass=unresolved_mass, distinct_gaps=len(gaps),
                entry_probability=entry_probability, weighted_equivalent_gap_reach=weighted,
                weighted_equivalent_gap_reach_bounds=weighted_bounds,
                gaps=[dict(position=k, reach=p) for k, p in sorted(gaps.items(), key=lambda r: (-r[1], r[0])) if p > 0],
                validation=dict(probability_conserved=True, canonical_gaps_merged_before_squaring=True,
                                no_unprepared_child_queries=True, cyclic_components=cyclic_components,
                                closed_components=closed_components))
