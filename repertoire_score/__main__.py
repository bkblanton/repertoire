import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import chess
import numpy as np
from .graph import parse, conflicts, resolve, topology, infer_entries, key, chapter_region, region_entries
from .explorer import Explorer, DEFAULT_FILTERS
from .model import prepare, outcome, empirical, draws
from .evaluate import backward, forward, summarize, chapter_score, COMPLETED, KNOWN
from .report import events, markdown, starting_position_reference
from .baseline import chapter_entry_baseline
from .render import update_report_outputs
from .transitions import chapter_transitions


def entry_positions(value, graph):
    positions = []
    for item in value:
        if isinstance(item, str):
            position = key(chess.Board(item if len(item.split()) == 6 else item+" 0 1"))
        else:
            board = chess.Board(item.get("root_fen", chess.STARTING_FEN))
            for move in item["path"]:
                try:
                    board.push_uci(move)
                except ValueError:
                    board.push_san(move)
            position = key(board)
        if position not in graph.nodes:
            raise ValueError(f"Chapter entry not in repertoire: {position}")
        positions.append(position)
    return sorted(set(positions))


def analyze(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    color = args.color == "white"
    graph = parse(args.pgn, config.get("exclude", []))
    inspection = {"chapters": graph.chapters, "nodes": len(graph.nodes), "conflicts": conflicts(graph, color), "entries": infer_entries(graph)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    if args.command == "inspect":
        print(f"{len(graph.chapters)} chapters; {len(graph.nodes)} positions; {len(inspection['conflicts'])} own-move conflicts")
        return
    transitions = resolve(graph, color, config.get("policy", {}))
    absolute_reach = "root_weights" in config or key(chess.Board()) in graph.roots
    if "root_weights" in config:
        root_weights = config["root_weights"]
        if any(k not in graph.nodes or not 0 <= v <= 1 for k, v in root_weights.items()) or abs(sum(root_weights.values())-1)>1e-10:
            raise ValueError("Invalid root weights")
    elif key(chess.Board()) in graph.roots:
        root_weights = {key(chess.Board()): 1.0}
    elif len(graph.roots) == 1:
        root_weights = {graph.roots[0]: 1.0}
    else:
        raise ValueError("Disconnected custom roots require explicit root_weights for an overall score")
    entries, entry_status, regions = {}, {}, {}
    known_chapters = {c['id'] for c in graph.chapters}
    if set(config.get('chapter_regions', {})) - known_chapters:
        raise ValueError('Chapter region configuration contains unknown chapter IDs')
    for c in graph.chapters:
        cid = c['id']
        if cid in config.get('chapter_regions', {}):
            if cid in config.get('entries', {}):
                raise ValueError(f'Configure either chapter_regions or entries, not both: {cid}')
            definition = config['chapter_regions'][cid]
            anchors = entry_positions(definition['anchors'], graph)
            members = chapter_region(graph, cid, anchors)
            entries[cid] = region_entries(transitions, list(dict.fromkeys([*root_weights, *graph.roots])), members)
            regions[cid] = {'anchors': anchors, 'positions': sorted(members),
                            'description': definition.get('description', 'Chapter-owned continuations from the configured subject anchors')}
            entry_status[cid] = 'First arrival anywhere in the chapter region, including shared positions and later transpositions'
        elif cid in config.get('entries', {}):
            entries[cid] = entry_positions(config['entries'][cid], graph)
            entry_status[cid] = config.get('entry_description', 'explicit configuration')
        else:
            entries[cid] = inspection['entries'][cid]['positions']
            entry_status[cid] = inspection['entries'][cid]['status']
    analysis_roots = list(root_weights) + [p for positions in entries.values() for p in positions]
    order = topology(transitions, analysis_roots)
    explorer = Explorer(args.cache, dict(DEFAULT_FILTERS, **config.get("filters", {})), args.offline, args.refresh)
    evidence = {}
    try:
        required = [k for k in order if outcome(chess.Board(graph.nodes[k].fen), color) is None and
                    (not transitions[k] or chess.Board(graph.nodes[k].fen).turn != color)]
        starting_position = key(chess.Board())
        baseline_positions = [p for positions in entries.values() for p in positions
                              if outcome(chess.Board(p + " 0 1"), color) is None]
        required = list(dict.fromkeys([starting_position, *required, *baseline_positions]))
        for i, k in enumerate(required):
            evidence[k] = explorer.get(k)
            if i % 25 == 0 or i+1 == len(required):
                print(f"Evidence {i+1}/{len(required)}", flush=True)
    finally:
        explorer.close()
    model = prepare(graph, transitions, order, color, evidence)
    raw_sample = empirical(model, color)
    post_sample = draws(model, color, args.simulations, args.seed, args.prior)
    raw_values = backward(model, order, raw_sample, args.sparse_threshold)
    post_values = backward(model, order, post_sample, args.sparse_threshold, args.simulations)
    raw_root = sum(w*raw_values[k] for k, w in root_weights.items())
    post_root = sum(w*post_values[k] for k, w in root_weights.items())
    raw_flow, _ = forward(model, order, raw_sample, root_weights)
    post_flow, _ = forward(model, order, post_sample, root_weights, args.simulations)
    ledger = events(graph, model, raw_sample, post_sample, raw_flow, post_flow, color, sum(args.prior))
    if not np.isclose(sum(e["contribution"] or 0 for e in ledger), raw_root[KNOWN, 0], atol=1e-9):
        raise AssertionError("Raw weighted stopping contributions differ from root value")
    decomposition = sum((post_flow[k, j]*post_sample[k][j][1] for k, j in post_flow), np.zeros(args.simulations))
    if not np.allclose(decomposition, post_root[COMPLETED], atol=1e-9):
        raise AssertionError("Posterior weighted stopping contributions differ from root value")
    chapters = []
    reachable = set(topology(transitions, list(root_weights)))
    for c in graph.chapters:
        positions = entries[c["id"]]
        if not positions:
            summary = {"status": "entry_configuration_required"}
        elif all(p not in reachable for p in positions) and c["root"] not in reachable and len(positions) == 1:
            summary = summarize(raw_values[positions[0]], post_values[positions[0]])
            summary["entry_probability"] = None
            summary["conditional_basis"] = "disconnected custom-FEN chapter; no absolute root weight"
        else:
            summary = chapter_score(model, order, raw_sample, post_sample, raw_values, post_values, root_weights, positions, args.simulations)
        if not absolute_reach:
            summary["probability_conditional_on_custom_root"] = summary.get("entry_probability")
            summary["entry_probability"] = None
            summary["posterior_entry_probability_mean"] = None
        chapters.append({"id": c["id"], "name": c["name"], "url": c["url"], "entry_status": entry_status[c["id"]],
                         "region": regions.get(c['id']),
                         "entries": [{"position": p, "path": graph.nodes[p].path,
                                      "conditional_first_entry_weight": summary.get('first_entry_weights', {}).get(p)} for p in positions], "score": summary,
                         "entry_baseline": chapter_entry_baseline(positions, summary, evidence, color, explorer.provenance)})
    destination_sets = {cid: region['positions'] if (region := regions.get(cid)) else positions
                        for cid, positions in entries.items()}
    transition_rows = chapter_transitions(model, order, raw_sample, chapters, destination_sets)
    sensitivity = []
    # Symmetric weak and stronger priors expose changes from the default.
    for prior in ([0.1, 0.1, 0.1], [2., 2., 2.]):
        alt_samples = draws(model, color, args.simulations, args.seed, prior)
        alt_values = backward(model, order, alt_samples, args.sparse_threshold, args.simulations)
        alt_root = sum(w*alt_values[k] for k, w in root_weights.items())
        sensitivity.append({"prior": prior, "overall": summarize(raw_root, alt_root), "chapters": {
            c["id"]: chapter_score(model, order, raw_sample, alt_samples, raw_values, alt_values, root_weights, entries[c["id"]], args.simulations)
            for c in graph.chapters if entries[c["id"]]}})
        del alt_samples, alt_values
    report = {"color": args.color, "overall": summarize(raw_root, post_root), "chapters": chapters,
              "chapter_transitions": transition_rows,
              "starting_position_reference": starting_position_reference(evidence[starting_position], color, explorer.provenance[starting_position]),
              "events": ledger, "prior_sensitivity": sensitivity,
              "manifest": {"created_at": datetime.now(timezone.utc).isoformat(), "input_path": str(Path(args.pgn).resolve()),
                           "input_sha256": hashlib.sha256(Path(args.pgn).read_bytes()).hexdigest(), "configuration": config,
                           "filters": explorer.filters, "endpoint": "https://explorer.lichess.org/lichess", "evidence": explorer.provenance,
                           "prior": args.prior, "seed": args.seed, "simulations": args.simulations, "sparse_threshold": args.sparse_threshold,
                           "positions": len(graph.nodes), "evaluated_positions": len(order), "root_weights": root_weights,
                           "overall_basis": "supplied root weights" if "root_weights" in config else "standard starting position" if absolute_reach else "conditional on custom PGN root",
                           "tolerance_score_points": args.tolerance,
                           "tolerance_met": float((post_root[COMPLETED].max()-post_root[COMPLETED].min())*100) <= args.tolerance},
              "diagnostics": {"policy_conflicts": inspection["conflicts"], "entry_inspection": inspection["entries"],
                              "cycles": [], "sanity_checks_passed": True}}
    report["manifest"]["tolerance_met"] = np.ptp(np.quantile(post_root[COMPLETED], [0.025, 0.975]))*100 <= args.tolerance
    report["manifest"]["tolerance_met"] = bool(report["manifest"]["tolerance_met"])
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    update_report_outputs(output.with_suffix(".json"))
    print(json.dumps({"overall": report["overall"], "report": str(output.with_suffix('.md').resolve())}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Expected scores under a forced repertoire policy")
    parser.add_argument("command", choices=["inspect", "run"])
    parser.add_argument("pgn")
    parser.add_argument("--color", required=True, choices=["white", "black"])
    parser.add_argument("--config")
    parser.add_argument("--output", default="reports/repertoire")
    parser.add_argument("--cache", default=".cache/explorer")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--prior", nargs=3, type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--sparse-threshold", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1.0)
    args = parser.parse_args()
    if args.simulations < 100 or min(args.prior) <= 0 or args.sparse_threshold < 1 or args.tolerance <= 0:
        parser.error("Require at least 100 simulations, positive prior, sparse threshold and tolerance")
    try:
        analyze(args)
    except (ValueError, RuntimeError) as exc:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        diagnostic = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc),
                      "scores_available": False, "cache_preserved": True}
        output.with_suffix(".error.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        print(f"Analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
