"""Score one repertoire: expected results under a fixed move policy, from cached or fetched Explorer evidence."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import chess
import numpy as np
from . import SCHEMA_VERSION
from .graph import parse, conflicts, resolve, topology, infer_entries, key, chapter_region, region_entries, chapter_policy_overrides
from .board_cache import STARTING_POSITION, owner_outcome, turn
from .context import DEFAULT_CACHE
from .explorer import Explorer, DEFAULT_FILTERS, add_token_option, apply_token_file
from .model import prepare, empirical
from .evaluate import backward, forward, summarize, chapter_score, KNOWN
from .uncertainty import METHOD as UNCERTAINTY_METHOD, Posterior
from .ledger import events, starting_position_reference
from .attribution import enrich
from .baseline import chapter_entry_baseline
from .render import update_report_outputs
from .layout import data_json, report_directory
from .transitions import chapter_transitions, hitting_bounds
from .depth import prepared_depth_values, summarize_depth, chapter_prepared_depth


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
    for conflict in inspection['conflicts']:
        conflict['selected_moves'] = {move: weight for move, (_, weight) in transitions[conflict['position']].items()}
        conflict['resolution'] = 'explicit configuration' if conflict['position'] in config.get('policy', {}) else 'first PGN move in first chapter order'
    profiles = [{'overrides': {}, 'transitions': transitions, 'chapters': []}]
    profile_ids = {'{}': 0}
    chapter_profiles = {}
    for c in graph.chapters:
        overrides = chapter_policy_overrides(graph, color, transitions, c['id'])
        identity = json.dumps(overrides, sort_keys=True)
        if identity not in profile_ids:
            profile_ids[identity] = len(profiles)
            profiles.append({'overrides': overrides,
                             'transitions': resolve(graph, color, dict(config.get('policy', {}), **overrides)), 'chapters': []})
        index = profile_ids[identity]
        profiles[index]['chapters'].append(c['id'])
        chapter_profiles[c['id']] = index
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
        chapter_transitions_map = profiles[chapter_profiles[cid]]['transitions']
        if cid in config.get('chapter_regions', {}):
            if cid in config.get('entries', {}):
                raise ValueError(f'Configure either chapter_regions or entries, not both: {cid}')
            definition = config['chapter_regions'][cid]
            anchors = entry_positions(definition['anchors'], graph)
            members = chapter_region(graph, cid, anchors)
            entries[cid] = region_entries(chapter_transitions_map, list(dict.fromkeys([*root_weights, *graph.roots])), members)
            regions[cid] = {'anchors': anchors, 'positions': sorted(members),
                            'description': definition.get('description', 'Chapter-owned continuations from the configured subject anchors')}
            entry_status[cid] = 'First arrival anywhere in the chapter region, including shared positions and later transpositions'
        elif cid in config.get('entries', {}):
            entries[cid] = entry_positions(config['entries'][cid], graph)
            entry_status[cid] = config.get('entry_description', 'explicit configuration')
        else:
            anchors = inspection['entries'][cid]['candidates']
            if not anchors:
                anchors = [next((k for k in c['mainline'][1:] if k in graph.nodes and chess.Board(graph.nodes[k].fen).turn == color), c['root'])]
            members = chapter_region(graph, cid, anchors)
            entries[cid] = region_entries(chapter_transitions_map, list(dict.fromkeys([*root_weights, *graph.roots])), members)
            if not entries[cid]:
                # Unique positions may all belong to discarded own variations.
                anchors = [next((k for k in c['mainline'][1:] if k in graph.nodes and chess.Board(graph.nodes[k].fen).turn == color), c['root'])]
                members = chapter_region(graph, cid, anchors)
                entries[cid] = region_entries(chapter_transitions_map, list(dict.fromkeys([*root_weights, *graph.roots])), members)
            regions[cid] = {'anchors': anchors, 'positions': sorted(members),
                            'description': 'All first chapter-unique positions and chapter-owned descendants; if none can be entered under the chapter policy, the first opponent reply or PGN root is used'}
            entry_status[cid] = 'Automatic chapter region, evaluated under the chapter comparison policy'
    analysis_roots = list(root_weights) + [p for positions in entries.values() for p in positions]
    for profile in profiles:
        profile['order'] = topology(profile['transitions'], analysis_roots)
    order = profiles[0]['order']
    explorer = Explorer(args.cache, dict(DEFAULT_FILTERS, **config.get("filters", {})), args.offline, args.refresh)
    evidence = {}
    try:
        required = [k for profile in profiles for k in profile['order']
                    if owner_outcome(k, color) is None and (not profile['transitions'][k] or turn(k) != color)]
        starting_position = STARTING_POSITION
        baseline_positions = [p for positions in entries.values() for p in positions if owner_outcome(p, color) is None]
        required = list(dict.fromkeys([starting_position, *required, *baseline_positions]))
        for i, k in enumerate(required):
            evidence[k] = explorer.get(k)
            if i % 25 == 0 or i+1 == len(required):
                print(f"Evidence {i+1}/{len(required)}", flush=True)
    finally:
        explorer.close()
    for profile in profiles:
        profile['model'] = prepare(graph, profile['transitions'], profile['order'], color, evidence)
        profile['raw'] = empirical(profile['model'], color)
        profile['values'] = backward(profile['model'], profile['order'], profile['raw'], args.sparse_threshold)
        profile['depth'] = prepared_depth_values(profile['model'], profile['order'], profile['raw'])
        profile['reachable'] = set(topology(profile['transitions'], list(root_weights)))

    def conditional(profile, c, posterior):
        positions = entries[c['id']]
        if not positions:
            summary = {'status': 'entry_configuration_required'}
        elif all(p not in profile['reachable'] for p in positions) and c['root'] not in profile['reachable'] and len(positions) == 1:
            summary = summarize(profile['values'][positions[0]], posterior.mixture({positions[0]: 1.}))
            summary.update(entry_probability=None, conditional_basis='disconnected custom-FEN chapter; no absolute root weight')
        else:
            summary = chapter_score(profile['model'], profile['order'], profile['raw'],
                                    profile['values'], root_weights, positions, posterior)
        if not absolute_reach:
            summary['probability_conditional_on_custom_root'] = summary.get('entry_probability')
            summary['entry_probability'] = None
            summary['posterior_entry_probability_mean'] = None
        return summary

    chapter_results = {}
    global_profile = profiles[0]
    model, raw_sample, raw_values, depth_values = (global_profile[k] for k in ('model', 'raw', 'values', 'depth'))
    raw_root = sum(w*raw_values[k] for k, w in root_weights.items())
    for index, profile in enumerate(profiles):
        posterior = Posterior(profile['model'], profile['order'], color, args.prior, args.sparse_threshold)
        if index == 0:
            overall_posterior = posterior.mixture(root_weights)
            raw_flow, _ = forward(model, order, raw_sample, root_weights)
            post_flow, _ = forward(model, order, posterior.sample, root_weights)
            ledger = events(graph, model, raw_sample, posterior, raw_flow, post_flow, color, sum(args.prior))
            if not np.isclose(sum(e['contribution'] or 0 for e in ledger), raw_root[KNOWN, 0], atol=1e-9):
                raise AssertionError('Raw weighted stopping contributions differ from root value')
            if not np.isclose(sum(e['posterior_contribution_mean'] for e in ledger), overall_posterior['mean'], atol=1e-9):
                raise AssertionError('Posterior weighted stopping contributions differ from root value')
        for c in graph.chapters:
            if c['id'] not in profile['chapters']:
                continue
            positions = entries[c['id']]
            summary = conditional(profile, c, posterior)
            summary['prepared_depth'] = chapter_prepared_depth(profile['depth'], positions, summary)
            destination = set(regions[c['id']]['positions']) if c['id'] in regions else set(positions)
            actual_hits = hitting_bounds(model, order, raw_sample, destination)
            low, high = (sum(w*actual_hits[k][i] for k,w in root_weights.items()) for i in (0,1))
            summary['overall_policy_entry_probability'] = low if low == high and absolute_reach else None
            summary['overall_policy_entry_probability_bounds'] = [low, high] if absolute_reach else None
            summary['entry_probability_basis'] = 'chapter comparison policy from the repertoire roots'
            chapter_results[c['id']] = {
                'id': c['id'], 'name': c['name'], 'url': c['url'], 'entry_status': entry_status[c['id']],
                'policy_overrides': profile['overrides'],
                'policy_basis': 'chapter first choices, then overall policy' if profile['overrides'] else 'overall policy',
                'region': regions.get(c['id']),
                'entries': [{'position': p, 'path': graph.nodes[p].path,
                             'conditional_first_entry_weight': summary.get('first_entry_weights', {}).get(p)} for p in positions],
                'score': summary, 'entry_baseline': chapter_entry_baseline(positions, summary, evidence, color, explorer.provenance)}
    chapters = [chapter_results[c['id']] for c in graph.chapters]
    destination_sets = {cid: region['positions'] if (region := regions.get(cid)) else positions
                        for cid, positions in entries.items()}
    transition_rows = []
    for profile in profiles:
        rows = chapter_transitions(profile['model'], profile['order'], profile['raw'], chapters, destination_sets)
        transition_rows.extend(dict(row, policy_basis='source chapter comparison policy') for row in rows
                               if row['source_id'] in profile['chapters'])
    sensitivity = []
    # Symmetric weak and stronger priors expose changes from the default.
    for prior in ([0.1, 0.1, 0.1], [2., 2., 2.]):
        item = {'prior': prior, 'chapters': {}}
        for index, profile in enumerate(profiles):
            alternative = Posterior(profile['model'], profile['order'], color, prior, args.sparse_threshold)
            if index == 0:
                item['overall'] = summarize(raw_root, alternative.mixture(root_weights))
            for c in graph.chapters:
                if c['id'] in profile['chapters'] and entries[c['id']]:
                    item['chapters'][c['id']] = conditional(profile, c, alternative)
        sensitivity.append(item)
    interval = overall_posterior['credible_interval_95']
    report = {"color": args.color, "overall": summarize(raw_root, overall_posterior), "chapters": chapters,
              "chapter_transitions": transition_rows,
              "starting_position_reference": starting_position_reference(evidence[starting_position], color, explorer.provenance[starting_position]),
              "events": ledger, "prior_sensitivity": sensitivity,
              "manifest": {"created_at": datetime.now(timezone.utc).isoformat(), "input_path": str(Path(args.pgn).resolve()),
                           "input_sha256": hashlib.sha256(Path(args.pgn).read_bytes()).hexdigest(), "configuration": config,
                           "filters": explorer.filters, "endpoint": "https://explorer.lichess.org/lichess", "evidence": explorer.provenance,
                           "prior": args.prior, "uncertainty_method": UNCERTAINTY_METHOD, "sparse_threshold": args.sparse_threshold,
                           "positions": len(graph.nodes), "evaluated_positions": len({k for p in profiles for k in p['order']}),
                           "overall_policy_evaluated_positions": len(order), "root_weights": root_weights,
                           "conflict_resolution": "explicit policy overrides, otherwise first PGN move in first chapter order",
                           "schema_version": SCHEMA_VERSION,
                           "traversal_rule": "At every prepared opponent-turn board, expand cached reply rows, including after the last recorded PGN move. Immediate transpositions into any known repertoire board resume preparation. Stop at an unanswered own-turn board, unprepared reply, terminal outcome, or unresolved evidence.",
                           "chapter_policy_semantics": "chapter first choices from roots, overall policy elsewhere; chapter comparison reach is separate from overall-policy region reach",
                           "policy_profile_count": len(profiles),
                           "overall_basis": "supplied root weights" if "root_weights" in config else "standard starting position" if absolute_reach else "conditional on custom PGN root",
                           "tolerance_score_points": args.tolerance,
                           "tolerance_met": bool((interval[1] - interval[0]) * 100 <= args.tolerance)},
              "diagnostics": {"policy_conflicts": inspection["conflicts"], "entry_inspection": inspection["entries"],
                              "cycles": [], "sanity_checks_passed": True}}
    report['overall']['prepared_depth'] = summarize_depth(depth_values, root_weights)
    report['manifest']['prepared_depth_definition'] = (
        'Expected remaining own prepared moves before a deviation, theory leaf, or terminal outcome; '
        'includes an available own move at entry, uses the merged repertoire and empirical opponent probabilities, '
        'and uses chapter comparison policies and first-entry weights. No depth cutoff, discount, or bonus for entry itself. '
        'Missing move distributions or first-entry weights remain unresolved with conditional bounds.')
    enrich(report, graph)
    output.with_suffix(".json").write_text(data_json(report), encoding="utf-8")
    update_report_outputs(output.with_suffix(".json"))
    print(json.dumps({"overall": report["overall"], "report": str((report_directory(output)/'report.md').resolve())}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Expected scores under a forced repertoire policy")
    parser.add_argument("command", choices=["inspect", "run"])
    parser.add_argument("pgn")
    parser.add_argument("--color", required=True, choices=["white", "black"])
    parser.add_argument("--config")
    parser.add_argument("--output", default="reports/data/repertoire")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--prior", nargs=3, type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument("--sparse-threshold", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1.0)
    add_token_option(parser)
    args = parser.parse_args()
    apply_token_file(parser, args)
    if min(args.prior) <= 0 or args.sparse_threshold < 1 or args.tolerance <= 0:
        parser.error("Require a positive prior, sparse threshold and tolerance")
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
