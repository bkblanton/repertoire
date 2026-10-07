"""Score one repertoire: expected results under a fixed move policy, from cached or fetched Explorer evidence."""
import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
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
from .schema import Chapter, ChapterScore
from .evaluate import backward, forward, summarize, chapter_score, KNOWN
from .uncertainty import METHOD as UNCERTAINTY_METHOD, Posterior
from .ledger import events, starting_position_reference
from .attribution import enrich
from .baseline import chapter_entry_baseline
from .render import update_report_outputs
from .layout import data_json, report_directory
from .transitions import chapter_transitions, hitting_bounds
from .depth import prepared_depth_values, summarize_depth, chapter_prepared_depth
from .status import Status


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


@dataclass
class Plan:
    """What to score: policy profiles, root weights and each chapter's entry positions."""
    profiles: list
    root_weights: dict
    absolute_reach: bool
    entries: dict
    entry_status: dict
    regions: dict

    @property
    def overall(self):
        return self.profiles[0]


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def inspect_repertoire(graph, color):
    return {"chapters": graph.chapters, "nodes": len(graph.nodes), "conflicts": conflicts(graph, color), "entries": infer_entries(graph)}


def policy_profiles(graph, color, policy):
    """The overall policy, then one profile per distinct set of chapter-local first choices."""
    transitions = resolve(graph, color, policy)
    profiles = [{'overrides': {}, 'transitions': transitions, 'chapters': []}]
    profile_ids = {'{}': 0}
    for c in graph.chapters:
        overrides = chapter_policy_overrides(graph, color, transitions, c['id'])
        identity = json.dumps(overrides, sort_keys=True)
        if identity not in profile_ids:
            profile_ids[identity] = len(profiles)
            profiles.append({'overrides': overrides,
                             'transitions': resolve(graph, color, dict(policy, **overrides)), 'chapters': []})
        profiles[profile_ids[identity]]['chapters'].append(c['id'])
    return profiles


def root_weighting(graph, config):
    """Starting weights for the overall score, and whether its reach is absolute."""
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
    return root_weights, absolute_reach


def chapter_entries(graph, color, config, profiles, root_weights, inspection):
    """Each chapter's first-entry positions, how they were chosen, and its region."""
    chapter_profiles = {cid: profile for profile in profiles for cid in profile['chapters']}
    entries, entry_status, regions = {}, {}, {}
    known_chapters = {c['id'] for c in graph.chapters}
    if set(config.get('chapter_regions', {})) - known_chapters:
        raise ValueError('Chapter region configuration contains unknown chapter IDs')
    starts = list(dict.fromkeys([*root_weights, *graph.roots]))
    for c in graph.chapters:
        cid = c['id']
        chapter_transitions_map = chapter_profiles[cid]['transitions']
        if cid in config.get('chapter_regions', {}):
            if cid in config.get('entries', {}):
                raise ValueError(f'Configure either chapter_regions or entries, not both: {cid}')
            definition = config['chapter_regions'][cid]
            anchors = entry_positions(definition['anchors'], graph)
            members = chapter_region(graph, cid, anchors)
            entries[cid] = region_entries(chapter_transitions_map, starts, members)
            regions[cid] = {'anchors': anchors, 'positions': sorted(members),
                            'description': definition.get('description', 'Chapter-owned continuations from the configured subject anchors')}
            entry_status[cid] = 'First arrival anywhere in the chapter region, including shared positions and later transpositions'
        elif cid in config.get('entries', {}):
            entries[cid] = entry_positions(config['entries'][cid], graph)
            entry_status[cid] = config.get('entry_description', 'explicit configuration')
        else:
            # The first own-turn mainline position, or the chapter root.
            fallback = [next((k for k in c['mainline'][1:] if k in graph.nodes and turn(k) == color), c['root'])]
            anchors = inspection['entries'][cid]['candidates'] or fallback
            members = chapter_region(graph, cid, anchors)
            entries[cid] = region_entries(chapter_transitions_map, starts, members)
            if not entries[cid]:
                # Unique positions may all belong to discarded own variations.
                anchors = fallback
                members = chapter_region(graph, cid, anchors)
                entries[cid] = region_entries(chapter_transitions_map, starts, members)
            regions[cid] = {'anchors': anchors, 'positions': sorted(members),
                            'description': 'All first chapter-unique positions and chapter-owned descendants; if none can be entered under the chapter policy, the first opponent reply or PGN root is used'}
            entry_status[cid] = 'Automatic chapter region, evaluated under the chapter comparison policy'
    return entries, entry_status, regions


def plan_repertoire(graph, color, config, inspection):
    profiles = policy_profiles(graph, color, config.get('policy', {}))
    root_weights, absolute_reach = root_weighting(graph, config)
    entries, entry_status, regions = chapter_entries(graph, color, config, profiles, root_weights, inspection)
    analysis_roots = list(root_weights) + [p for positions in entries.values() for p in positions]
    for profile in profiles:
        profile['order'] = topology(profile['transitions'], analysis_roots)
    return Plan(profiles, root_weights, absolute_reach, entries, entry_status, regions)


def required_positions(plan, color):
    """Every table the scores need: opponent turns, unanswered own turns, chapter entries and the start."""
    required = [k for profile in plan.profiles for k in profile['order']
                if owner_outcome(k, color) is None and (not profile['transitions'][k] or turn(k) != color)]
    baseline_positions = [p for positions in plan.entries.values() for p in positions if owner_outcome(p, color) is None]
    return list(dict.fromkeys([STARTING_POSITION, *required, *baseline_positions]))


def collect_evidence(explorer, positions):
    evidence = {}
    for i, k in enumerate(positions):
        evidence[k] = explorer.get(k)
        if i % 25 == 0 or i+1 == len(positions):
            print(f"Evidence {i+1}/{len(positions)}", flush=True)
    return evidence


def evaluate_profiles(graph, color, plan, evidence, sparse_threshold):
    for profile in plan.profiles:
        profile['model'] = prepare(graph, profile['transitions'], profile['order'], color, evidence)
        profile['raw'] = empirical(profile['model'], color)
        profile['values'] = backward(profile['model'], profile['order'], profile['raw'], sparse_threshold)
        profile['depth'] = prepared_depth_values(profile['model'], profile['order'], profile['raw'])
        profile['reachable'] = set(topology(profile['transitions'], list(plan.root_weights)))


def conditional(plan, profile, c, posterior) -> ChapterScore:
    """A chapter's score conditional on first entry, under its comparison policy."""
    positions = plan.entries[c['id']]
    if not positions:
        summary = {'status': Status.ENTRY_CONFIGURATION_REQUIRED}
    elif all(p not in profile['reachable'] for p in positions) and c['root'] not in profile['reachable'] and len(positions) == 1:
        summary = summarize(profile['values'][positions[0]], posterior.mixture({positions[0]: 1.}))
        summary.update(entry_probability=None, conditional_basis='disconnected custom-FEN chapter; no absolute root weight')
    else:
        summary = chapter_score(profile['model'], profile['order'], profile['raw'],
                                profile['values'], plan.root_weights, positions, posterior)
    if not plan.absolute_reach:
        summary['probability_conditional_on_custom_root'] = summary.get('entry_probability')
        summary['entry_probability'] = None
        summary['posterior_entry_probability_mean'] = None
    return summary


def chapter_result(graph, color, plan, profile, c, posterior, evidence, provenance) -> Chapter:
    overall = plan.overall
    positions = plan.entries[c['id']]
    summary = conditional(plan, profile, c, posterior)
    summary['prepared_depth'] = chapter_prepared_depth(profile['depth'], positions, summary)
    destination = set(plan.regions[c['id']]['positions']) if c['id'] in plan.regions else set(positions)
    actual_hits = hitting_bounds(overall['model'], overall['order'], overall['raw'], destination)
    low, high = (sum(w*actual_hits[k][i] for k,w in plan.root_weights.items()) for i in (0,1))
    summary['overall_policy_entry_probability'] = low if low == high and plan.absolute_reach else None
    summary['overall_policy_entry_probability_bounds'] = [low, high] if plan.absolute_reach else None
    summary['entry_probability_basis'] = 'chapter comparison policy from the repertoire roots'
    return {
        'id': c['id'], 'name': c['name'], 'url': c['url'], 'entry_status': plan.entry_status[c['id']],
        'policy_overrides': profile['overrides'],
        'policy_basis': 'chapter first choices, then overall policy' if profile['overrides'] else 'overall policy',
        'region': plan.regions.get(c['id']),
        'entries': [{'position': p, 'path': graph.nodes[p].path,
                     'conditional_first_entry_weight': summary.get('first_entry_weights', {}).get(p)} for p in positions],
        'score': summary, 'entry_baseline': chapter_entry_baseline(positions, summary, evidence, color, provenance)}


def prior_sensitivity(graph, color, plan, raw_root, sparse_threshold):
    """Overall and chapter posteriors under symmetric weak and stronger priors."""
    sensitivity = []
    for prior in ([0.1, 0.1, 0.1], [2., 2., 2.]):
        item = {'prior': prior, 'chapters': {}}
        for index, profile in enumerate(plan.profiles):
            alternative = Posterior(profile['model'], profile['order'], color, prior, sparse_threshold)
            if index == 0:
                item['overall'] = summarize(raw_root, alternative.mixture(plan.root_weights))
            for c in graph.chapters:
                if c['id'] in profile['chapters'] and plan.entries[c['id']]:
                    item['chapters'][c['id']] = conditional(plan, profile, c, alternative)
        sensitivity.append(item)
    return sensitivity


def score_repertoire(graph, color, plan, evidence, provenance, prior, sparse_threshold):
    """Overall and chapter scores, the stopping-event ledger, transitions and prior sensitivity."""
    evaluate_profiles(graph, color, plan, evidence, sparse_threshold)
    overall, root_weights = plan.overall, plan.root_weights
    model, order, raw_sample = overall['model'], overall['order'], overall['raw']
    raw_root = sum(w*overall['values'][k] for k, w in root_weights.items())
    chapter_results = {}
    for index, profile in enumerate(plan.profiles):
        posterior = Posterior(profile['model'], profile['order'], color, prior, sparse_threshold)
        if index == 0:
            overall_posterior = posterior.mixture(root_weights)
            raw_flow, _ = forward(model, order, raw_sample, root_weights)
            post_flow, _ = forward(model, order, posterior.sample, root_weights)
            ledger = events(graph, model, raw_sample, posterior, raw_flow, post_flow, color, sum(prior))
            if not np.isclose(sum(e['contribution'] or 0 for e in ledger), raw_root[KNOWN], atol=1e-9):
                raise AssertionError('Raw weighted stopping contributions differ from root value')
            if not np.isclose(sum(e['posterior_contribution_mean'] for e in ledger), overall_posterior['mean'], atol=1e-9):
                raise AssertionError('Posterior weighted stopping contributions differ from root value')
        for c in graph.chapters:
            if c['id'] in profile['chapters']:
                chapter_results[c['id']] = chapter_result(graph, color, plan, profile, c, posterior, evidence, provenance)
    chapters = [chapter_results[c['id']] for c in graph.chapters]
    destination_sets = {cid: region['positions'] if (region := plan.regions.get(cid)) else positions
                        for cid, positions in plan.entries.items()}
    transition_rows = []
    for profile in plan.profiles:
        rows = chapter_transitions(profile['model'], profile['order'], profile['raw'], chapters, destination_sets)
        transition_rows.extend(dict(row, policy_basis='source chapter comparison policy') for row in rows
                               if row['source_id'] in profile['chapters'])
    result = summarize(raw_root, overall_posterior)
    result['prepared_depth'] = summarize_depth(overall['depth'], root_weights)
    return dict(overall=result, chapters=chapters, chapter_transitions=transition_rows, events=ledger,
                prior_sensitivity=prior_sensitivity(graph, color, plan, raw_root, sparse_threshold))


def analyze(args):
    config = load_config(args.config)
    color = args.color == "white"
    graph = parse(args.pgn, config.get("exclude", []))
    inspection = inspect_repertoire(graph, color)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    if args.command == "inspect":
        print(f"{len(graph.chapters)} chapters; {len(graph.nodes)} positions; {len(inspection['conflicts'])} own-move conflicts")
        return
    plan = plan_repertoire(graph, color, config, inspection)
    for conflict in inspection['conflicts']:
        conflict['selected_moves'] = {move: weight for move, (_, weight) in plan.overall['transitions'][conflict['position']].items()}
        conflict['resolution'] = 'explicit configuration' if conflict['position'] in config.get('policy', {}) else 'first PGN move in first chapter order'
    explorer = Explorer(args.cache, dict(DEFAULT_FILTERS, **config.get("filters", {})), args.offline, args.refresh)
    try:
        evidence = collect_evidence(explorer, required_positions(plan, color))
    finally:
        explorer.close()
    scores = score_repertoire(graph, color, plan, evidence, explorer.provenance, args.prior, args.sparse_threshold)
    interval = scores['overall']['posterior']['credible_interval_95']
    report = {"color": args.color, "overall": scores['overall'], "chapters": scores['chapters'],
              "chapter_transitions": scores['chapter_transitions'],
              "starting_position_reference": starting_position_reference(evidence[STARTING_POSITION], color, explorer.provenance[STARTING_POSITION]),
              "events": scores['events'], "prior_sensitivity": scores['prior_sensitivity'],
              "manifest": {"created_at": datetime.now(timezone.utc).isoformat(), "input_path": str(Path(args.pgn).resolve()),
                           "input_sha256": hashlib.sha256(Path(args.pgn).read_bytes()).hexdigest(), "configuration": config,
                           "filters": explorer.filters, "endpoint": "https://explorer.lichess.org/lichess", "evidence": explorer.provenance,
                           "prior": args.prior, "uncertainty_method": UNCERTAINTY_METHOD, "sparse_threshold": args.sparse_threshold,
                           "positions": len(graph.nodes), "evaluated_positions": len({k for p in plan.profiles for k in p['order']}),
                           "overall_policy_evaluated_positions": len(plan.overall['order']), "root_weights": plan.root_weights,
                           "conflict_resolution": "explicit policy overrides, otherwise first PGN move in first chapter order",
                           "schema_version": SCHEMA_VERSION,
                           "traversal_rule": "At every prepared opponent-turn board, expand cached reply rows, including after the last recorded PGN move. Immediate transpositions into any known repertoire board resume preparation. Stop at an unanswered own-turn board, unprepared reply, terminal outcome, or unresolved evidence.",
                           "chapter_policy_semantics": "chapter first choices from roots, overall policy elsewhere; chapter comparison reach is separate from overall-policy region reach",
                           "policy_profile_count": len(plan.profiles),
                           "overall_basis": "supplied root weights" if "root_weights" in config else "standard starting position" if plan.absolute_reach else "conditional on custom PGN root",
                           "tolerance_score_points": args.tolerance,
                           "tolerance_met": bool((interval[1] - interval[0]) * 100 <= args.tolerance)},
              "diagnostics": {"policy_conflicts": inspection["conflicts"], "entry_inspection": inspection["entries"],
                              "cycles": [], "sanity_checks_passed": True}}
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
