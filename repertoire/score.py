"""Score one repertoire: expected results under a fixed move policy, from cached or fetched Explorer evidence."""

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import chess
import numpy as np

from . import SCHEMA_VERSION
from .attribution import enrich
from .baseline import chapter_entry_baseline
from .board_cache import STARTING_POSITION, owner_outcome, san, turn
from .context import DEFAULT_CACHE
from .depth import chapter_prepared_depth, prepared_depth_values, summarize_depth
from .evaluate import (
    COMPLETED,
    KNOWN,
    UNKNOWN,
    backward,
    chapter_score,
    forward,
    reaches,
    select_alternatives,
    summarize,
)
from .explorer import DEFAULT_FILTERS, Explorer, add_token_option, apply_token_file, collect
from .graph import (
    Graph,
    GraphChapter,
    InferredEntries,
    Policy,
    Transitions,
    alternative_transitions,
    chapter_policy_overrides,
    chapter_positions,
    conflicts,
    infer_entries,
    key,
    parse,
    reachable,
    resolve,
    topology,
)
from .layout import data_json, report_directory
from .ledger import events, starting_position_reference
from .model import Evidence, Model, Sampled, empirical, prepare
from .render import update_report_outputs
from .schema import (
    Alternative,
    Chapter,
    ChapterScore,
    JsonObject,
    Position,
    ScoreSummary,
    StoppingEvent,
)
from .status import Status
from .transitions import chapter_transitions, hitting_bounds
from .uncertainty import METHOD as UNCERTAINTY_METHOD
from .uncertainty import Posterior


# One policy profile; evaluate_profiles adds the evaluated fields.
class Profile(TypedDict):
    overrides: dict[Position, str]
    transitions: Transitions
    chapters: list[str]
    order: NotRequired[list[Position]]
    model: NotRequired[Model]
    raw: NotRequired[Sampled]
    values: NotRequired[dict[Position, np.ndarray]]
    depth: NotRequired[dict[Position, tuple[float, float]]]
    reachable: NotRequired[set[Position]]


class Inspection(TypedDict):
    chapters: list[GraphChapter]
    nodes: int
    conflicts: list[JsonObject]
    entries: dict[str, InferredEntries]


class Sensitivity(TypedDict):
    prior: list[float]
    chapters: dict[str, ChapterScore]
    overall: NotRequired[ScoreSummary]


class Scores(TypedDict):
    overall: ScoreSummary
    chapters: list[Chapter]
    chapter_transitions: list[JsonObject]
    events: list[StoppingEvent]
    prior_sensitivity: list[Sensitivity]


def entry_positions(value: Iterable[str | JsonObject], graph: Graph) -> list[Position]:
    positions = []
    for item in value:
        if isinstance(item, str):
            position = key(chess.Board(item if len(item.split()) == 6 else item + " 0 1"))
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
    """What to score: policy profiles, root weights, each chapter's entry positions and the positions it prepares."""

    profiles: list[Profile]
    root_weights: dict[Position, float]
    absolute_reach: bool
    entries: dict[str, list[Position]]
    entry_status: dict[str, str]
    prepared: dict[str, list[Position]]
    # Every chapter alternative selectable: what evidence planning must cover before alternatives are scored.
    candidates: Transitions
    reachable: list[Position]

    @property
    def roots(self) -> list[Position]:
        """The repertoire roots and every chapter entry: where evaluation starts."""
        return list(dict.fromkeys([*self.root_weights, *(p for positions in self.entries.values() for p in positions)]))

    @property
    def overall(self) -> Profile:
        return self.profiles[0]


CONFIG = "configs/{}.json"


def load_config(path: str | Path | None) -> JsonObject:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def config_path(color: str, path: str | None = None) -> str:
    """An explicit configuration, else configs/<color>.json, created without overrides when missing."""
    if path:
        return path
    default = Path(CONFIG.format(color))
    if not default.is_file():
        default.parent.mkdir(parents=True, exist_ok=True)
        default.write_text(json.dumps({"policy": {}}, indent=2) + "\n", encoding="utf-8")
        print(f"Created {default} with no overrides", flush=True)
    return str(default)


def inspect_repertoire(graph: Graph, color: bool) -> Inspection:
    return {
        "chapters": graph.chapters,
        "nodes": len(graph.nodes),
        "conflicts": conflicts(graph, color),
        "entries": infer_entries(graph, color),
    }


def policy_profiles(graph: Graph, color: bool, policy: Policy) -> list[Profile]:
    """The overall policy, then one profile per distinct set of chapter-local first choices."""
    transitions = resolve(graph, color, policy)
    profiles: list[Profile] = [{'overrides': {}, 'transitions': transitions, 'chapters': []}]
    profile_ids = {'{}': 0}
    for c in graph.chapters:
        overrides = chapter_policy_overrides(graph, color, transitions, c['id'])
        identity = json.dumps(overrides, sort_keys=True)
        if identity not in profile_ids:
            profile_ids[identity] = len(profiles)
            profiles.append(
                {
                    'overrides': overrides,
                    'transitions': resolve(graph, color, dict(policy, **overrides)),
                    'chapters': [],
                }
            )
        profiles[profile_ids[identity]]['chapters'].append(c['id'])
    return profiles


def root_weighting(graph: Graph, config: JsonObject) -> tuple[dict[Position, float], bool]:
    """Starting weights for the overall score, and whether its reach is absolute."""
    absolute_reach = "root_weights" in config or key(chess.Board()) in graph.roots
    if "root_weights" in config:
        root_weights = config["root_weights"]
        if (
            any(k not in graph.nodes or not 0 <= v <= 1 for k, v in root_weights.items())
            or abs(sum(root_weights.values()) - 1) > 1e-10
        ):
            raise ValueError("Invalid root weights")
    elif key(chess.Board()) in graph.roots:
        root_weights = {key(chess.Board()): 1.0}
    elif len(graph.roots) == 1:
        root_weights = {graph.roots[0]: 1.0}
    else:
        raise ValueError("Disconnected custom roots require explicit root_weights for an overall score")
    return root_weights, absolute_reach


def chapter_entries(
    graph: Graph, color: bool, config: JsonObject, inspection: Inspection
) -> tuple[dict[str, list[Position]], dict[str, str], dict[str, list[Position]]]:
    """Each chapter's entry positions, how they were chosen, and the positions it prepares from them.

    Entries are automatic unless `entries` configures them. A chapter is reached only at its entries, by
    any move order; later positions it shares with other chapters do not count as reaching it.
    """
    if 'chapter_regions' in config:
        raise ValueError(
            'chapter_regions is no longer supported: entries are found automatically; '
            'use entries only for chapters whose automatic entries are wrong'
        )
    configured = config.get('entries', {})
    if set(configured) - {c['id'] for c in graph.chapters}:
        raise ValueError('Entry configuration contains unknown chapter IDs')
    entries, entry_status, prepared = {}, {}, {}
    for c in graph.chapters:
        cid = c['id']
        if cid in configured:
            entries[cid] = entry_positions(configured[cid], graph)
            entry_status[cid] = config.get('entry_description', 'explicit configuration')
        elif automatic := inspection['entries'][cid]['positions']:
            entries[cid] = automatic
            entry_status[cid] = 'Automatic: the first positions on its lines that no other chapter continues from'
        else:
            # Every position is also prepared by another chapter: the first own-turn mainline position, or the root.
            entries[cid] = [next((k for k in c['mainline'][1:] if k in graph.nodes and turn(k) == color), c['root'])]
            entry_status[cid] = (
                'Automatic fallback: other chapters continue from all of its positions, so its first own-turn '
                'mainline position is used'
            )
        prepared[cid] = sorted(chapter_positions(graph, cid, entries[cid]))
    return entries, entry_status, prepared


def plan_repertoire(
    graph: Graph,
    color: bool,
    config: JsonObject,
    inspection: Inspection,
    selected: Mapping[Position, str] | None = None,
) -> Plan:
    """What to score. `selected` holds the winners of competing chapter alternatives, once they are known."""
    policy = config.get('policy', {})
    profiles = policy_profiles(graph, color, dict(policy, **(selected or {})))
    root_weights, absolute_reach = root_weighting(graph, config)
    entries, entry_status, prepared = chapter_entries(graph, color, config, inspection)
    analysis_roots = list(root_weights) + [p for positions in entries.values() for p in positions]
    for profile in profiles:
        profile['order'] = topology(profile['transitions'], analysis_roots)
    candidates = alternative_transitions(graph, color, policy)
    return Plan(
        profiles,
        root_weights,
        absolute_reach,
        entries,
        entry_status,
        prepared,
        candidates,
        reachable(candidates, [*analysis_roots, *graph.roots]),
    )


def required_positions(plan: Plan, color: bool) -> list[Position]:
    """Every table the scores need: opponent turns, unanswered own turns, chapter entries and the start."""
    required = [
        k
        for profile in plan.profiles
        for k in profile['order']
        if owner_outcome(k, color) is None and (not profile['transitions'][k] or turn(k) != color)
    ]
    # Boards only a competing chapter alternative reaches are needed to score that alternative.
    required += [
        k for k in plan.reachable if owner_outcome(k, color) is None and (not plan.candidates[k] or turn(k) != color)
    ]
    baseline_positions = [
        p for positions in plan.entries.values() for p in positions if owner_outcome(p, color) is None
    ]
    return list(dict.fromkeys([STARTING_POSITION, *required, *baseline_positions]))


def parent_positions(plan: Plan, color: bool) -> list[Position]:
    """Own-move parents: the tables the vulnerabilities stage compares each selected move against."""
    parents = [
        k
        for profile in plan.profiles
        for k in profile['order']
        if owner_outcome(k, color) is None and profile['transitions'][k] and turn(k) == color
    ]
    parents += [
        k for k in plan.reachable if owner_outcome(k, color) is None and plan.candidates[k] and turn(k) == color
    ]
    return list(dict.fromkeys(parents))


def evaluate_profiles(graph: Graph, color: bool, plan: Plan, evidence: Evidence, sparse_threshold: int) -> None:
    for profile in plan.profiles:
        profile['model'] = prepare(graph, profile['transitions'], profile['order'], color, evidence)
        profile['raw'] = empirical(profile['model'], color)
        profile['values'] = backward(profile['model'], profile['order'], profile['raw'], sparse_threshold)
        profile['depth'] = prepared_depth_values(profile['model'], profile['order'], profile['raw'])
        profile['reachable'] = set(topology(profile['transitions'], list(plan.root_weights)))


def conditional(plan: Plan, profile: Profile, c: GraphChapter, posterior: Posterior) -> ChapterScore:
    """A chapter's score conditional on first entry, under its comparison policy."""
    positions = plan.entries[c['id']]
    summary: ChapterScore
    if not positions:
        summary = {'status': Status.ENTRY_CONFIGURATION_REQUIRED}
    elif (
        all(p not in profile['reachable'] for p in positions)
        and c['root'] not in profile['reachable']
        and len(positions) == 1
    ):
        summary = cast(ChapterScore, summarize(profile['values'][positions[0]], posterior.mixture({positions[0]: 1.0})))
        summary['entry_probability'] = None
        summary['conditional_basis'] = 'disconnected custom-FEN chapter; no absolute root weight'
    else:
        summary = chapter_score(
            profile['model'],
            profile['order'],
            profile['raw'],
            profile['values'],
            plan.root_weights,
            positions,
            posterior,
        )
    if not plan.absolute_reach:
        summary['probability_conditional_on_custom_root'] = summary.get('entry_probability')
        summary['entry_probability'] = None
        summary['posterior_entry_probability_mean'] = None
    return summary


def chapter_result(
    graph: Graph,
    color: bool,
    plan: Plan,
    profile: Profile,
    c: GraphChapter,
    posterior: Posterior,
    evidence: Evidence,
    provenance: Mapping[Position, dict[str, str]],
) -> Chapter:
    overall = plan.overall
    positions = plan.entries[c['id']]
    summary = conditional(plan, profile, c, posterior)
    summary['prepared_depth'] = chapter_prepared_depth(profile['depth'], positions, summary)
    actual_hits = hitting_bounds(overall['model'], overall['order'], overall['raw'], set(positions))
    low, high = (sum(w * actual_hits[k][i] for k, w in plan.root_weights.items()) for i in (0, 1))
    summary['overall_policy_entry_probability'] = low if low == high and plan.absolute_reach else None
    summary['overall_policy_entry_probability_bounds'] = [low, high] if plan.absolute_reach else None
    summary['entry_probability_basis'] = 'chapter comparison policy from the repertoire roots'
    return {
        'id': c['id'],
        'name': c['name'],
        'url': c['url'],
        'entry_status': plan.entry_status[c['id']],
        'policy_overrides': profile['overrides'],
        'policy_basis': 'chapter first choices, then overall policy' if profile['overrides'] else 'overall policy',
        'prepared_positions': plan.prepared[c['id']],
        'entries': [
            {
                'position': p,
                'path': graph.nodes[p].path,
                'conditional_first_entry_weight': summary.get('first_entry_weights', {}).get(p),
            }
            for p in positions
        ],
        'score': summary,
        'entry_baseline': chapter_entry_baseline(positions, summary, evidence, color, provenance),
    }


def prior_sensitivity(
    graph: Graph, color: bool, plan: Plan, raw_root: np.ndarray, sparse_threshold: int
) -> list[Sensitivity]:
    """Overall and chapter posteriors under symmetric weak and stronger priors."""
    sensitivity: list[Sensitivity] = []
    for prior in ([0.1, 0.1, 0.1], [2.0, 2.0, 2.0]):
        item: Sensitivity = {'prior': prior, 'chapters': {}}
        for index, profile in enumerate(plan.profiles):
            alternative = Posterior(profile['model'], profile['order'], color, prior, sparse_threshold)
            if index == 0:
                item['overall'] = summarize(raw_root, alternative.mixture(plan.root_weights))
            for c in graph.chapters:
                if c['id'] in profile['chapters'] and plan.entries[c['id']]:
                    item['chapters'][c['id']] = conditional(plan, profile, c, alternative)
        sensitivity.append(item)
    return sensitivity


def score_repertoire(
    graph: Graph,
    color: bool,
    plan: Plan,
    evidence: Evidence,
    provenance: Mapping[Position, dict[str, str]],
    prior: list[float],
    sparse_threshold: int,
) -> Scores:
    """Overall and chapter scores, the stopping-event ledger, transitions and prior sensitivity."""
    evaluate_profiles(graph, color, plan, evidence, sparse_threshold)
    overall, root_weights = plan.overall, plan.root_weights
    model, order, raw_sample = overall['model'], overall['order'], overall['raw']
    raw_root = cast(np.ndarray, sum(w * overall['values'][k] for k, w in root_weights.items()))
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
            if not np.isclose(
                sum(e['posterior_contribution_mean'] for e in ledger), overall_posterior['mean'], atol=1e-9
            ):
                raise AssertionError('Posterior weighted stopping contributions differ from root value')
        for c in graph.chapters:
            if c['id'] in profile['chapters']:
                chapter_results[c['id']] = chapter_result(
                    graph, color, plan, profile, c, posterior, evidence, provenance
                )
    chapters = [chapter_results[c['id']] for c in graph.chapters]
    transition_rows: list[JsonObject] = []
    for profile in plan.profiles:
        rows = chapter_transitions(profile['model'], profile['order'], profile['raw'], chapters, plan.entries)
        transition_rows.extend(
            dict(row, policy_basis='source chapter comparison policy')
            for row in rows
            if row['source_id'] in profile['chapters']
        )
    result = summarize(raw_root, overall_posterior)
    result['prepared_depth'] = summarize_depth(overall['depth'], root_weights)
    return dict(
        overall=result,
        chapters=chapters,
        chapter_transitions=transition_rows,
        events=ledger,
        prior_sensitivity=prior_sensitivity(graph, color, plan, raw_root, sparse_threshold),
    )


def alternative_rows(graph: Graph, color: bool, plan: Plan, selection: Mapping[Position, dict]) -> list[Alternative]:
    """Each board where chapters compete: every alternative's score there, its reach and the winner."""
    overall = plan.overall
    reach = reaches(overall['model'], overall['order'], overall['raw'], plan.root_weights)
    rows: list[Alternative] = []
    for k, choice in selection.items():
        node = graph.nodes[k]
        rows.append(
            {
                'position': k,
                'path': node.path,
                'reach': reach.get(k, 0.0) if plan.absolute_reach else None,
                'selected': choice['selected'],
                'options': [
                    {
                        'move': move,
                        'san': san(k, move),
                        'chapters': [cid for cid, recorded in node.chapter_moves.items() if recorded[:1] == [move]],
                        'score': float(value[KNOWN]) if value[UNKNOWN] == 0 else None,
                        'resolved_contribution': float(value[KNOWN]),
                        'unresolved_mass': float(value[UNKNOWN]),
                        'prior_completed_score': float(value[COMPLETED]),
                    }
                    for move, value in choice['scores'].items()
                ],
            }
        )
    return sorted(rows, key=lambda r: (-(r['reach'] or 0.0), len(r['path']), r['path']))


def analyze(args: argparse.Namespace) -> JsonObject | None:
    config = load_config(args.config)
    color = args.color == "white"
    graph = parse(args.pgn, config.get("exclude", []))
    inspection = inspect_repertoire(graph, color)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    if args.command == "inspect":
        print(
            f"{len(graph.chapters)} chapters; {len(graph.nodes)} positions; "
            f"{len(inspection['conflicts'])} own-move conflicts"
        )
        for c in graph.chapters:
            found = inspection['entries'][c['id']]
            paths = '; '.join(' '.join(path) or 'start' for path in sorted(found['paths'])) or 'none (fallback applies)'
            print(f"{c['id']} {c['name']}: {paths}")
        return None
    plan = plan_repertoire(graph, color, config, inspection)
    explorer = Explorer(args.cache, dict(DEFAULT_FILTERS, **config.get("filters", {})), args.offline, args.refresh)
    try:
        evidence = collect(explorer, required_positions(plan, color), f"{args.color} scores")
    finally:
        explorer.close()
    # Competing chapter alternatives are scored first; the overall policy plays each winner.
    selection = select_alternatives(
        graph, color, config.get('policy', {}), evidence, [*plan.roots, *graph.roots], args.sparse_threshold
    )
    selected = {k: choice['selected'] for k, choice in selection.items()}
    if selected:
        plan = plan_repertoire(graph, color, config, inspection, selected)
    for conflict in inspection['conflicts']:
        conflict['selected_moves'] = {
            move: weight for move, (_, weight) in plan.overall['transitions'][conflict['position']].items()
        }
        conflict['resolution'] = (
            'explicit configuration'
            if conflict['position'] in config.get('policy', {})
            else 'highest repertoire score among chapter alternatives'
            if conflict['position'] in selection
            else 'first PGN move in first chapter order'
        )
    scores = score_repertoire(graph, color, plan, evidence, explorer.provenance, args.prior, args.sparse_threshold)
    interval = scores['overall']['posterior']['credible_interval_95']
    report: JsonObject = {
        "color": args.color,
        "overall": scores['overall'],
        "chapters": scores['chapters'],
        "chapter_transitions": scores['chapter_transitions'],
        "alternatives": alternative_rows(graph, color, plan, selection),
        "starting_position_reference": starting_position_reference(
            evidence[STARTING_POSITION], color, explorer.provenance[STARTING_POSITION]
        ),
        "events": scores['events'],
        "prior_sensitivity": scores['prior_sensitivity'],
        "manifest": {
            "created_at": datetime.now(UTC).isoformat(),
            "input_path": str(Path(args.pgn).resolve()),
            "input_sha256": hashlib.sha256(Path(args.pgn).read_bytes()).hexdigest(),
            "configuration": config,
            "filters": explorer.filters,
            "endpoint": "https://explorer.lichess.org/lichess",
            "evidence": explorer.provenance,
            "prior": args.prior,
            "uncertainty_method": UNCERTAINTY_METHOD,
            "sparse_threshold": args.sparse_threshold,
            "positions": len(graph.nodes),
            "evaluated_positions": len({k for p in plan.profiles for k in p['order']}),
            "overall_policy_evaluated_positions": len(plan.overall['order']),
            "root_weights": plan.root_weights,
            "conflict_resolution": "explicit policy overrides; where chapters record different first moves, the "
            "alternative with the highest repertoire score; otherwise first PGN move in first chapter order",
            "selected_alternatives": selected,
            "schema_version": SCHEMA_VERSION,
            "traversal_rule": "At every prepared opponent-turn board, expand cached reply rows, "
            "including after the last recorded PGN move. Immediate transpositions "
            "into any known repertoire board resume preparation. Stop at an "
            "unanswered own-turn board, unprepared reply, terminal outcome, or "
            "unresolved evidence.",
            "chapter_policy_semantics": "chapter first choices from roots, overall policy elsewhere; chapter "
            "comparison reach is separate from overall-policy entry reach",
            "policy_profile_count": len(plan.profiles),
            "overall_basis": "supplied root weights"
            if "root_weights" in config
            else "standard starting position"
            if plan.absolute_reach
            else "conditional on custom PGN root",
            "tolerance_score_points": args.tolerance,
            "tolerance_met": bool((interval[1] - interval[0]) * 100 <= args.tolerance),
        },
        "diagnostics": {
            "policy_conflicts": inspection["conflicts"],
            "entry_inspection": inspection["entries"],
            "cycles": [],
            "sanity_checks_passed": True,
        },
    }
    report['manifest']['prepared_depth_definition'] = (
        'Expected remaining own prepared moves before a deviation, theory leaf, or terminal outcome; '
        'includes an available own move at entry, uses the merged repertoire and empirical opponent probabilities, '
        'and uses chapter comparison policies and first-entry weights. No '
        'depth cutoff, discount, or bonus for entry itself. '
        'Missing move distributions or first-entry weights remain unresolved with conditional bounds.'
    )
    enrich(report, graph)
    output.with_suffix(".json").write_text(data_json(report), encoding="utf-8")
    update_report_outputs(output.with_suffix(".json"))
    return {"overall": report["overall"], "report": str((report_directory(output) / 'report.md').resolve())}


def main() -> None:
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
        summary = analyze(args)
    except (ValueError, RuntimeError) as exc:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        diagnostic = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "scores_available": False,
            "cache_preserved": True,
        }
        output.with_suffix(".error.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        print(f"Analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if summary is not None:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
