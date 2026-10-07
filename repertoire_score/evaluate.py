"""DAG evaluation, first-entry weighting and probability conservation checks."""
import numpy as np

from .board_cache import children
from .schema import ScoreSummary
from .status import Status

# Vector components: resolved contribution, unresolved mass, sparse contribution,
# sparse mass, leaf mass, deviation mass, other mass, prior-completed score.
KNOWN, UNKNOWN, SPARSE_KNOWN, SPARSE, LEAF, DEVIATION, OTHER, COMPLETED = range(8)


def stopping_vector(branch, sampled_score, sparse_threshold, width=None):
    result = np.zeros(8 if width is None else (8, width))
    unresolved = branch.fixed_score is None and not sum(branch.counts)
    sparse = branch.fixed_score is None and sum(branch.counts) < sparse_threshold
    s = 0.5 if sampled_score is None else sampled_score
    result[COMPLETED] = s
    result[UNKNOWN] = float(unresolved)
    result[KNOWN] = 0 if unresolved else s
    result[SPARSE] = float(sparse)
    result[SPARSE_KNOWN] = 0 if unresolved or not sparse else s
    # Unresolved replaces the event's resolved category, keeping all masses disjoint.
    if not unresolved:
        category = LEAF if branch.kind == "theory_leaf" else DEVIATION if branch.kind == "deviation" else OTHER
        result[category] = 1
    return result


def backward(model, order, sampled, sparse_threshold, width=None):
    """Each position's component vector. With `width`, every probability and score is an array of that many
    simulated draws and each vector has one column per draw; the tests use this to check the calculated uncertainty."""
    values = {}
    for k in order:
        v = np.zeros(8 if width is None else (8, width))
        for b, (p, s) in zip(model[k].branches, sampled[k]):
            v += p * (values[b.target] if b.target is not None else stopping_vector(b, s, sparse_threshold, width))
        if not np.allclose(v[UNKNOWN]+v[LEAF]+v[DEVIATION]+v[OTHER], 1, atol=1e-9):
            raise AssertionError(f"Probability conservation failed at {k}")
        values[k] = v
    return values


def can_enter(model, order, entries):
    """Whether each position can still reach one of `entries`, including through unresolved replies."""
    result = {}
    for k in order:
        targets = [b.target for b in model[k].branches if b.target is not None] + model[k].potential_targets
        result[k] = k in entries or any(result[t] for t in targets)
    return result


def forward(model, order, sampled, roots, stop_at=(), entering=None):
    """Propagate probability from the roots, stopping at `stop_at`.

    With `entering` (from can_enter), positions that cannot reach an entry are not expanded: their
    mass is absorbed whole and their individual stops are omitted. Entry masses are unchanged.
    Probabilities may be arrays of simulated draws; masses then become arrays of the same length.
    """
    mass = dict.fromkeys(order, 0.)
    for k, weight in roots.items():
        mass[k] += weight
    stops, entries = {}, {}
    absorbed = 0.
    for k in reversed(order):
        if k in stop_at:
            entries[k] = mass[k]
            continue
        if entering is not None and not entering[k]:
            absorbed += mass[k]
            continue
        for j, (b, (p, _)) in enumerate(zip(model[k].branches, sampled[k])):
            flow = mass[k] * p
            if b.target is not None:
                mass[b.target] += flow
            else:
                stops[(k, j)] = flow
    total = sum(stops.values(), 0.)+sum(entries.values(), 0.)+absorbed
    if not np.allclose(total, 1, atol=1e-9):
        raise AssertionError("Forward stopping mass does not sum to one")
    return stops, entries


def reaches(model, order, sampled, roots, stop_at=()):
    """Incoming probability at each position over every route from the roots; transpositions add up.

    Flow is not expanded past `stop_at`, so each of those positions holds its first-entry probability.
    """
    stop_at = set(stop_at)
    mass = dict.fromkeys(order, 0.)
    for k, weight in roots.items():
        if weight:
            mass[k] += weight
    left = 0.
    for k in reversed(order):
        if k in stop_at:
            left += mass[k]
            continue
        for b, (p, _) in zip(model[k].branches, sampled[k]):
            if b.target is None:
                left += mass[k] * p
            else:
                mass[b.target] += mass[k] * p
    if not np.allclose(left, sum(roots.values()), atol=1e-9):
        raise AssertionError("Forward reach does not conserve probability")
    return mass


def better_route(a, b):
    """Whether route `a` beats `b`: more likely, with exact ties going to the earlier root and then earlier moves."""
    return a[0] > b[0] or (a[0] == b[0] and a[1:] < b[1:])


def best_routes(model, order, sampled, roots, stop_at=(), replies=False):
    """The most likely single route to each position, as (probability, root, moves).

    Probabilities always use all routes (see `reaches`); this route only labels a position. Routes are not
    extended past `stop_at`. With `replies`, the boards after unprepared opponent replies get routes too.
    """
    stop_at = set(stop_at)
    best = {k: (w, k, ()) for k, w in roots.items() if w > 0}
    for k in reversed(order):
        if k not in best or k in stop_at:
            continue
        probability, root, moves = best[k]
        for b, (p, _) in zip(model[k].branches, sampled[k]):
            if p <= 0:
                continue
            if b.target is not None:
                target = b.target
            elif replies and b.kind == "deviation":
                target = children(k)[b.move]
            else:
                continue
            candidate = probability * p, root, (*moves, b.move)
            if target not in best or better_route(candidate, best[target]):
                best[target] = candidate
    return best


def summarize(r, posterior) -> ScoreSummary:
    """Empirical score fields from the raw component vector `r`, plus the posterior block from uncertainty.py."""
    u = float(r[UNKNOWN])
    return {
        "raw_empirical_score": float(r[KNOWN]) if u == 0 else None,
        "resolved_contribution": float(r[KNOWN]), "unresolved_mass": u,
        "conditional_bounds": [float(r[KNOWN]), float(r[KNOWN]+u)],
        "sparse_mass": float(r[SPARSE]),
        "sparse_sensitivity": [float(r[KNOWN]-r[SPARSE_KNOWN]), float(r[KNOWN]-r[SPARSE_KNOWN]+r[SPARSE])],
        "masses": {"theory_leaf": float(r[LEAF]), "deviation": float(r[DEVIATION]), "other_stop": float(r[OTHER]), "unresolved": u},
        "posterior": posterior,
    }


def chapter_score(model, order, raw_sample, values, root_weights, entries, posterior):
    """Score conditional on first entry, with `posterior` an uncertainty.Posterior for the same model."""
    # Only positions that can still reach the chapter matter for its entry weights.
    entering = can_enter(model, order, entries)
    raw_stops, raw_entries = forward(model, order, raw_sample, root_weights, stop_at=entries, entering=entering)
    raw_reach = sum(raw_entries.values(), 0.)
    r = sum((raw_entries[k]*values[k] for k in raw_entries), np.zeros(8))
    weights = {k: float(v/raw_reach) if raw_reach else None for k, v in raw_entries.items()}
    unresolved_entry_mass = sum(float(flow) for (k,j),flow in raw_stops.items()
                                if model[k].branches[j].kind == "unresolved_distribution" and entering[k])
    if unresolved_entry_mass > 0 and len(entries) > 1:
        return {"status": Status.UNRESOLVED_ENTRY_WEIGHTS, "entry_probability": None,
                "entry_probability_bounds": [float(raw_reach),float(raw_reach)+unresolved_entry_mass],
                "conditional_score_bounds": [min(values[k][KNOWN] for k in entries),
                                             max(values[k][KNOWN]+values[k][UNKNOWN] for k in entries)]}
    chapter = posterior.chapter(root_weights, entries)
    if raw_reach <= 0:
        if len(entries) == 1:
            k = next(iter(entries))
            summary = summarize(values[k], posterior.mixture({k: 1.}))
            summary["conditional_basis"] = "single entry position, even though root reach is zero"
        else:
            summary = {"status": Status.UNREACHABLE_MULTIPLE_ENTRIES}
    else:
        if chapter['summary'] is None:
            raise ValueError("Posterior first-entry weights are zero; use a stronger prior or explicit entries")
        summary = summarize(r/raw_reach, chapter['summary'])
    summary.update(entry_probability=float(raw_reach), posterior_entry_probability_mean=chapter['entry_probability'],
                   first_entry_weights=weights)
    if unresolved_entry_mass > 0:
        summary["entry_probability"] = None
        summary["posterior_entry_probability_mean"] = None
        summary["entry_probability_bounds"] = [float(raw_reach),float(raw_reach)+unresolved_entry_mass]
    return summary
