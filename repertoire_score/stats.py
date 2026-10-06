"""Weighted correlation helpers shared by the correlation analyses and their report section."""
import numpy as np


def rank(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends-counts+1+ends)/2)[inverse]


def correlation(x, y, weights=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if len(x) < 2 or w.sum() <= 0:
        return float('nan')
    dx, dy = x-np.average(x, weights=w), y-np.average(y, weights=w)
    denominator = np.sqrt(np.sum(w*dx*dx)*np.sum(w*dy*dy))
    return float(np.clip(np.sum(w*dx*dy)/denominator, -1, 1)) if denominator > 0 else float('nan')


def connected_groups(position_sets):
    """Connected components of overlapping sets, including indirect overlap through a third set."""
    remaining, groups = set(range(len(position_sets))), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, pending = {first}, [first]
        while pending:
            i = pending.pop()
            hits = {j for j in remaining if position_sets[i] & position_sets[j]}
            remaining -= hits
            component |= hits
            pending.extend(sorted(hits))
        groups.append(sorted(component))
    return groups


def cell(result, metric):
    """A saved statistic for report tables."""
    value = result[metric]
    if value is None:
        return 'undefined'
    return f"{value:.3f}{'%' if metric == 'slope_pp_per_move' else ''}"
