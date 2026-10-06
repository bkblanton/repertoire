"""DAG evaluation, first-entry weighting and probability conservation checks."""
import numpy as np

# Vector components: resolved contribution, unresolved mass, sparse contribution,
# sparse mass, leaf mass, deviation mass, other mass, prior-completed score.
KNOWN, UNKNOWN, SPARSE_KNOWN, SPARSE, LEAF, DEVIATION, OTHER, COMPLETED = range(8)


def stopping_vector(branch, sampled_score, sparse_threshold, width):
    result = np.zeros((8, width))
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


def backward(model, order, sampled, sparse_threshold, width=1):
    values = {}
    for k in order:
        v = np.zeros((8, width))
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


def forward(model, order, sampled, roots, width=1, stop_at=(), entering=None):
    """Propagate probability from the roots, stopping at `stop_at`.

    With `entering` (from can_enter), positions that cannot reach an entry are not expanded: their
    mass is absorbed whole and their individual stops are omitted. Entry masses are unchanged.
    """
    mass = {k: np.zeros(width) for k in order}
    for k, weight in roots.items():
        mass[k] += weight
    stops, entries = {}, {}
    absorbed = np.zeros(width)
    for k in reversed(order):
        if k in stop_at:
            entries[k] = mass[k].copy()
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
    total = sum(stops.values(), np.zeros(width))+sum(entries.values(), np.zeros(width))+absorbed
    if not np.allclose(total, 1, atol=1e-9):
        raise AssertionError("Forward stopping mass does not sum to one")
    return stops, entries


def summarize(raw, posterior):
    r = raw[:, 0]
    u = float(r[UNKNOWN])
    result = {
        "raw_empirical_score": float(r[KNOWN]) if u == 0 else None,
        "resolved_contribution": float(r[KNOWN]), "unresolved_mass": u,
        "conditional_bounds": [float(r[KNOWN]), float(r[KNOWN]+u)],
        "sparse_mass": float(r[SPARSE]),
        "sparse_sensitivity": [float(r[KNOWN]-r[SPARSE_KNOWN]), float(r[KNOWN]-r[SPARSE_KNOWN]+r[SPARSE])],
        "masses": {"theory_leaf": float(r[LEAF]), "deviation": float(r[DEVIATION]), "other_stop": float(r[OTHER]), "unresolved": u},
    }
    pu = posterior[UNKNOWN]
    result["posterior"] = {
        "mean": float(posterior[COMPLETED].mean()),
        "credible_interval_95": np.quantile(posterior[COMPLETED], [0.025, 0.975]).tolist(),
        "label": "prior-completed estimate; missing distributions stop unresolved without assuming opponent play" if np.any(pu>0) else "posterior estimate",
        "unresolved_mass_mean": float(pu.mean()),
        "resolved_contribution_mean": float(posterior[KNOWN].mean()),
        "conditional_bounds_mean": [float(posterior[KNOWN].mean()), float((posterior[KNOWN]+pu).mean())],
        "masses_mean": {"theory_leaf": float(posterior[LEAF].mean()), "deviation": float(posterior[DEVIATION].mean()),
                        "other_stop": float(posterior[OTHER].mean()), "unresolved": float(pu.mean())},
    }
    return result


def chapter_score(model, order, raw_sample, post_sample, values, post_values, root_weights, entries, simulations):
    # Only positions that can still reach the chapter matter for its entry weights.
    entering = can_enter(model, order, entries)
    raw_stops, raw_entries = forward(model, order, raw_sample, root_weights, stop_at=entries, entering=entering)
    _, post_entries = forward(model, order, post_sample, root_weights, simulations, entries, entering)
    raw_reach = sum(raw_entries.values(), np.zeros(1))
    post_reach = sum(post_entries.values(), np.zeros(simulations))
    r = sum((raw_entries[k]*values[k] for k in raw_entries), np.zeros((8, 1)))
    p = sum((post_entries[k]*post_values[k] for k in post_entries), np.zeros((8, simulations)))
    weights = {k: float(v[0]/raw_reach[0]) if raw_reach[0] else None for k, v in raw_entries.items()}
    unresolved_entry_mass = sum(float(flow[0]) for (k,j),flow in raw_stops.items()
                                if model[k].branches[j].kind == "unresolved_distribution" and entering[k])
    if unresolved_entry_mass > 0 and len(entries) > 1:
        return {"status": "unresolved_first_entry_weights", "entry_probability": None,
                "entry_probability_bounds": [float(raw_reach[0]),float(raw_reach[0])+unresolved_entry_mass],
                "conditional_score_bounds": [min(values[k][KNOWN,0] for k in entries),
                                             max(values[k][KNOWN,0]+values[k][UNKNOWN,0] for k in entries)]}
    if raw_reach[0] <= 0:
        if len(entries) == 1:
            k = next(iter(entries))
            summary = summarize(values[k], post_values[k])
            summary["conditional_basis"] = "single entry position, even though root reach is zero"
        else:
            summary = {"status": "unreachable_multiple_entries_require_conditional_weights"}
    else:
        if np.any(post_reach <= 0):
            raise ValueError("Posterior first-entry weights underflowed; use a stronger prior or explicit entries")
        summary = summarize(r/raw_reach, p/post_reach)
    summary.update(entry_probability=float(raw_reach[0]), posterior_entry_probability_mean=float(post_reach.mean()), first_entry_weights=weights)
    if unresolved_entry_mass > 0:
        summary["entry_probability"] = None
        summary["posterior_entry_probability_mean"] = None
        summary["entry_probability_bounds"] = [float(raw_reach[0]),float(raw_reach[0])+unresolved_entry_mass]
    return summary
