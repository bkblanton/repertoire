"""Stopping-event ledger and starting-position reference saved with each score."""
import numpy as np
import chess
from .model import score
from .explorer import counts


def starting_position_reference(data, color, provenance):
    results = counts(data)
    return {
        "position": chess.STARTING_FEN,
        "sample_count": sum(results),
        "counts_white_draw_black": results,
        "white_score": score(results, chess.WHITE),
        "black_score": score(results, chess.BLACK),
        "owner_score": score(results, color),
        "basis": "Empirical standard starting-position score under the same Explorer filters, without forcing repertoire moves",
        "provenance": provenance,
    }


def events(graph, model, raw_sample, post_sample, raw_flow, post_flow, color, prior_strength):
    result = []
    for (k, j), mass in raw_flow.items():
        b = model[k].branches[j]
        raw_score = raw_sample[k][j][1]
        draws = post_sample[k][j][1]
        pmass = post_flow[(k, j)]
        sample = sum(b.counts)
        unresolved = b.fixed_score is None and sample == 0
        n = graph.nodes[k]
        board = chess.Board(n.fen)
        path = list(n.path)
        if b.move:
            move = chess.Move.from_uci(b.move)
            path.append(board.san(move))
            board.push(move)
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
        result.append({"parent_position": k, "position": board.fen(en_passant="legal"), "move": b.move,
                       "representative_path_san": path, "chapters": sorted(n.chapters), "type": b.kind,
                       "unresolved": unresolved, "score_status": "prior-only; no direct observations" if unresolved else "deterministic" if b.fixed_score is not None else "observed",
                       "sample_count": sample, "counts_white_draw_black": b.counts,
                       "probability": float(mass[0]), "posterior_probability_mean": float(pmass.mean()),
                       "raw_score": raw_score, "posterior_score_mean": float(np.mean(draws)),
                       "posterior_score_interval_95": interval,
                       "contribution": float(mass[0]*raw_score) if raw_score is not None else None,
                       "posterior_contribution_mean": float(np.mean(pmass*draws)),
                       "uncertainty_priority": float(pmass.mean())*(interval[1]-interval[0]),
                       "contribution_sd": float(np.std(pmass*draws)),
                       "prior_fraction": 0 if b.fixed_score is not None else
                       (prior_strength/len(model[k].branches) if model[k].mode == "opponent" else prior_strength)/
                       (sample+(prior_strength/len(model[k].branches) if model[k].mode == "opponent" else prior_strength)),
                       "evidence": "deterministic chess outcome" if b.fixed_score is not None else
                       "parent move-result table" if model[k].mode == "opponent" else
                       "no usable opponent distribution" if b.kind == "unresolved_distribution" else "position result counts"})
    return result
