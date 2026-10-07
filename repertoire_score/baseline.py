"""Chapter entry references with the repertoire's conditional first-entry weights."""
import math
from .status import Status
from .board_cache import owner_outcome
from .explorer import counts
from .model import score


def chapter_entry_baseline(positions, chapter_score, evidence, color, provenance):
    positions = sorted(set(positions))
    if not positions:
        return {"status": Status.ENTRY_CONFIGURATION_REQUIRED, "raw_score": None, "difference_pp": None}
    if len(positions) == 1:
        weights = {positions[0]: 1.0}
    else:
        weights = chapter_score.get("first_entry_weights", {})
        if any(weights.get(k) is None for k in positions):
            return {"status": Status.UNRESOLVED_ENTRY_WEIGHTS, "raw_score": None, "difference_pp": None}
        weights = {k: weights[k] for k in positions}
    if any(not math.isfinite(w) or w < 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-9):
        raise ValueError("Chapter baseline requires conditional first-entry weights summing to one")
    components = []
    known, unresolved = 0.0, 0.0
    for k, weight in weights.items():
        deterministic = owner_outcome(k, color)
        results = counts(evidence[k]) if deterministic is None else None
        entry_score = score(results, color) if deterministic is None else deterministic
        if entry_score is None:
            unresolved += weight
        else:
            known += weight * entry_score
        components.append({"position": k, "conditional_first_entry_weight": weight,
                           "score": entry_score, "sample_count": sum(results) if results is not None else None,
                           "counts_white_draw_black": results,
                           "provenance": provenance.get(k) if deterministic is None else "deterministic chess outcome"})
    baseline = known if unresolved == 0 else None
    repertoire = chapter_score.get("raw_empirical_score")
    return {"status": Status.RESOLVED if baseline is not None else Status.UNRESOLVED_ENTRY_SCORE,
            "basis": "Position-level database scores weighted by conditional first-entry probabilities; no forced repertoire moves after entry",
            "raw_score": baseline, "unresolved_mass": unresolved, "conditional_bounds": [known, known+unresolved],
            "difference_pp": 100*(repertoire-baseline) if repertoire is not None and baseline is not None else None,
            "difference_definition": "Empirical repertoire score minus empirical entry baseline, in percentage points",
            "components": components}
