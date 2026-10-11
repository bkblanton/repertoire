"""Stopping-event ledger and starting-position reference saved with each score."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import chess

from .board_cache import after_fen, position_of, san
from .explorer import counts
from .graph import Graph
from .model import Model, Sampled, score
from .schema import JsonObject, Position, StoppingEvent

if TYPE_CHECKING:
    from .uncertainty import Posterior


def starting_position_reference(data: JsonObject, color: bool, provenance: dict[str, str]) -> JsonObject:
    results = counts(data)
    return {
        "position": chess.STARTING_FEN,
        "sample_count": sum(results),
        "counts_white_draw_black": results,
        "white_score": score(results, chess.WHITE),
        "black_score": score(results, chess.BLACK),
        "owner_score": score(results, color),
        "basis": "Empirical standard starting-position score under the same Explorer "
        "filters, without forcing repertoire moves",
        "provenance": provenance,
    }


def events(
    graph: Graph,
    model: Model,
    raw_sample: Sampled,
    posterior: "Posterior",
    raw_flow: Mapping[tuple[Position, int], float],
    post_flow: Mapping[tuple[Position, int], float],
    color: bool,
    prior_strength: float,
) -> list[StoppingEvent]:
    """Every stopping event under the overall policy; `post_flow` is the posterior-mean flow from `posterior`.

    Occurrences of the same position in a repetition loop share one event per move, with their flows added.
    """
    merged: dict[tuple[Position, int], tuple[Position, float, float]] = {}
    for (k, j), mass in raw_flow.items():
        event = position_of(k), j
        if event in merged:
            node, total, posterior_total = merged[event]
            merged[event] = node, total + mass, posterior_total + post_flow[(k, j)]
        else:
            merged[event] = k, mass, post_flow[(k, j)]
    result: list[StoppingEvent] = []
    for (position, j), (k, mass, posterior_mass) in merged.items():
        b = model[k].branches[j]
        raw_score = raw_sample[k][j][1]
        score_mean, interval = posterior.stop_interval(k, j)
        # Upstream reach and this table are independent, so the mean of their product is the product of means.
        pmass = float(posterior_mass)
        sample = sum(b.counts)
        unresolved = b.fixed_score is None and sample == 0
        n = graph.nodes[position]
        path = list(n.path)
        if b.move:
            path.append(san(position, b.move))
        result.append(
            {
                "parent_position": position,
                "position": after_fen(n.fen, b.move) if b.move else n.fen,
                "move": b.move,
                "representative_path_san": path,
                "chapters": sorted(n.chapters),
                "type": cast(str, b.kind),
                "unresolved": unresolved,
                "score_status": "prior-only; no direct observations"
                if unresolved
                else "deterministic"
                if b.fixed_score is not None
                else "observed",
                "sample_count": sample,
                "counts_white_draw_black": b.counts,
                "probability": float(mass),
                "posterior_probability_mean": pmass,
                "raw_score": raw_score,
                "posterior_score_mean": score_mean,
                "posterior_score_interval_95": interval,
                "contribution": float(mass * raw_score) if raw_score is not None else None,
                "posterior_contribution_mean": pmass * score_mean,
                "uncertainty_priority": pmass * (interval[1] - interval[0]),
                "prior_fraction": 0
                if b.fixed_score is not None
                else (prior_strength / len(model[k].branches) if model[k].mode == "opponent" else prior_strength)
                / (
                    sample
                    + (prior_strength / len(model[k].branches) if model[k].mode == "opponent" else prior_strength)
                ),
                "evidence": "draw by threefold repetition"
                if b.kind == "repetition"
                else "deterministic chess outcome"
                if b.fixed_score is not None
                else "parent move-result table"
                if model[k].mode == "opponent"
                else "no usable opponent distribution"
                if b.kind == "unresolved_distribution"
                else "position result counts",
            }
        )
    return result
