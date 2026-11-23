"""Automatic chapter ordering by move-prefix tree and union first-entry reach.

Each chapter is anchored at the deepest common ancestor position of its active entries in the repertoire DAG
(using that ancestor's most likely route from the repertoire roots), then branches at every move split are
ordered recursively by the union first-entry reach of the chapters in each branch under the overall policy.
"""

import json
from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .board_cache import position_of
from .context import AnalysisContext
from .evaluate import Route, Weights, best_routes, better_route, reaches
from .graph import Graph, Transitions, resolve, topology
from .model import Evidence, Model, Sampled, empirical, prepare
from .schema import JsonObject, Position

CONFIG_KEY = 'chapter_order'
DEFAULT_ORDER = 'study'
MODES = ('study', 'opening')

PathKey = tuple[str, ...]
ProfileModel = tuple[Transitions, list[Position], Model, Sampled]


def order_from_config(config: Mapping[str, object]) -> str:
    """The chapter order of a configuration: 'study' (default) or 'opening'."""
    value = config.get(CONFIG_KEY, DEFAULT_ORDER)
    if not isinstance(value, str) or value not in MODES:
        raise ValueError(f"{CONFIG_KEY} must be 'study' or 'opening': {value!r}")
    return value


def configured_order(color: str, config: str | Path | None = None) -> str:
    """The chapter order in a configuration file, by default configs/<color>.json, or 'study' without one."""
    path = Path(config) if config else Path(f'configs/{color}.json')
    return order_from_config(json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {})


@dataclass
class _Node:
    chapters: list[str] = field(default_factory=list)
    children: dict[str, '_Node'] = field(default_factory=dict)


def _common_prefix(paths: Sequence[PathKey]) -> PathKey:
    if not paths:
        return ()
    prefix: list[str] = []
    for items in zip(*paths):
        if len(set(items)) != 1:
            break
        prefix.append(items[0])
    return tuple(prefix)


def _shortest_path(
    transitions: Transitions, roots: Sequence[Position], targets: Collection[Position]
) -> dict[Position, PathKey]:
    """Shortest structural move path from `roots` to each target in `transitions`, ignoring probabilities."""
    remaining = set(targets)
    found: dict[Position, PathKey] = {}
    seen: set[Position] = set()
    queue: deque[tuple[Position, PathKey]] = deque()
    for root in roots:
        if root in transitions and root not in seen:
            seen.add(root)
            queue.append((root, (position_of(root),)))
    while queue and remaining:
        k, path = queue.popleft()
        pos = position_of(k)
        if pos in remaining:
            found[pos] = path
            remaining.discard(pos)
            if not remaining:
                break
        for move, (target, _) in sorted(transitions.get(k, {}).items()):
            if target is not None and target not in seen:
                seen.add(target)
                queue.append((target, (*path, move)))
    return found


def _chapter_structural_path(
    graph: Graph, cid: str, root: Position, targets: Collection[Position]
) -> dict[Position, PathKey]:
    """Shortest path along one chapter's own recorded moves from its PGN root, for disconnected custom FENs."""
    remaining = set(targets)
    found: dict[Position, PathKey] = {}
    seen: set[Position] = {root}
    queue: deque[tuple[Position, PathKey]] = deque([(root, (root,))])
    while queue and remaining:
        k, path = queue.popleft()
        if k in remaining:
            found[k] = path
            remaining.discard(k)
            if not remaining:
                break
        node = graph.nodes.get(k)
        if node is None:
            continue
        for move in sorted(node.chapter_moves.get(cid, ())):
            target = node.edges[move]
            if target not in seen:
                seen.add(target)
                queue.append((target, (*path, move)))
    return found


def _profile_model(
    graph: Graph,
    color: bool,
    policy: Mapping[Position, str | Mapping[str, float]],
    evidence: Evidence,
    roots: Weights,
) -> ProfileModel:
    transitions = resolve(graph, color, policy)
    order = topology(transitions, list(roots))
    model = prepare(graph, transitions, order, color, evidence)
    raw = empirical(model, color)
    return transitions, order, model, raw


def _union_reach(profile: ProfileModel, roots: Weights, entries: frozenset[Position]) -> float:
    """Exact probability that a game from `roots` first enters at least one position in `entries`."""
    if not entries:
        return 0.0
    _, order, model, raw = profile
    mass = reaches(model, order, raw, roots, stop_at=entries)
    return float(sum(m for k, m in mass.items() if position_of(k) in entries))


def _common_ancestor_route(
    order: Sequence[Position],
    model: Model,
    raw: Sampled,
    mass: Mapping[Position, float],
    routes: Mapping[Position, Route],
    entry_set: Collection[Position],
    witnesses: Mapping[Position, Route],
) -> PathKey:
    """Most likely route to the deepest common ancestor position of all active entries in the repertoire DAG.

    Finding the common ancestor position before picking a representative route keeps multi-entry chapters
    transposition-invariant when one entry's most likely route arrives via a transposition from another branch.
    """
    if len(witnesses) == 1:
        _, root, moves = next(iter(witnesses.values()))
        return (position_of(root), *moves)
    bit = {pos: 1 << i for i, pos in enumerate(witnesses)}
    all_bits = (1 << len(witnesses)) - 1
    reach_entries: dict[Position, int] = {}
    for k in order:
        pos = position_of(k)
        if pos in bit:
            reach_entries[k] = bit[pos]
            continue
        if pos in entry_set:
            reach_entries[k] = 0
            continue
        mask = 0
        for b, (p, _) in zip(model[k].branches, raw[k]):
            if p > 0 and b.target is not None:
                mask |= reach_entries[b.target]
        reach_entries[k] = mask
    ancestor_set = {k for k in order if mass[k] > 0 and reach_entries[k] == all_bits}
    if not ancestor_set:
        return ()
    has_descendant: dict[Position, bool] = {}
    for k in order:
        has_descendant[k] = any(
            p > 0 and b.target is not None and (b.target in ancestor_set or has_descendant[b.target])
            for b, (p, _) in zip(model[k].branches, raw[k])
        )
    maximal = [k for k in order if k in ancestor_set and not has_descendant[k]]
    # When incomparable branches both reach every entry (a transposition diamond), prefer the highest-reach
    # position, breaking ties by route length and then better_route's deterministic ordering.
    best_node = max(
        maximal,
        key=lambda k: (round(mass[k], 12), len(routes[k][2]), routes[k][0]),
    )
    for candidate in maximal:
        if (
            round(mass[candidate], 12) == round(mass[best_node], 12)
            and len(routes[candidate][2]) == len(routes[best_node][2])
            and better_route(routes[candidate], routes[best_node])
        ):
            best_node = candidate
    _, root, moves = routes[best_node]
    return (position_of(root), *moves)


def _chapter_anchor(
    graph: Graph,
    cid: str,
    chapter_root: Position,
    entries: Sequence[Position],
    profile: ProfileModel,
    roots: Weights,
) -> PathKey:
    """Most likely route to the deepest DAG common ancestor of a chapter's active entries."""
    if not entries:
        return (chapter_root,)
    entry_set = set(entries)
    transitions, order, model, raw = profile
    mass = reaches(model, order, raw, roots, stop_at=entry_set)
    routes = best_routes(model, order, raw, roots, stop_at=entry_set)
    witnesses: dict[Position, Route] = {}
    for k in reversed(order):
        if mass[k] <= 0:
            continue
        pos = position_of(k)
        if pos not in entry_set:
            continue
        candidate = routes[k]
        if pos not in witnesses or better_route(candidate, witnesses[pos]):
            witnesses[pos] = candidate
    if witnesses:
        return _common_ancestor_route(order, model, raw, mass, routes, entry_set, witnesses)
    # Fall back to structural routes when no entry has positive empirical reach (zero-game replies, missing tables,
    # or disconnected custom-FEN chapters).
    structural = _shortest_path(transitions, list(roots), entry_set)
    if structural:
        return _common_prefix([structural[e] for e in entries if e in structural])
    local = _chapter_structural_path(graph, cid, chapter_root, entry_set)
    if local:
        return _common_prefix([local[e] for e in entries if e in local])
    return (chapter_root,)


def order_chapters(
    analysis: AnalysisContext,
    evidence: Evidence,
    mode: str = DEFAULT_ORDER,
    full: ProfileModel | None = None,
) -> JsonObject:
    """Resolve chapter order for one scored color: 'study' (PGN order) or 'opening' (move-prefix tree)."""
    mode = order_from_config({CONFIG_KEY: mode})
    chapters = analysis.saved['chapters']
    study_ids = [c['id'] for c in chapters]
    if mode == 'study' or len(study_ids) <= 1:
        return dict(mode=mode, ids=study_ids)

    graph, color, roots = analysis.graph, analysis.color, analysis.roots
    overall = full if full is not None else _profile_model(graph, color, analysis.policy, evidence, roots)
    profiles: dict[str, ProfileModel] = {'{}': overall}

    def profile_for(overrides: Mapping[str, str]) -> tuple[str, ProfileModel]:
        key = json.dumps(overrides, sort_keys=True)
        if key not in profiles:
            profiles[key] = _profile_model(graph, color, dict(analysis.policy, **overrides), evidence, roots)
        return key, profiles[key]

    chapter_roots = {c['id']: c['root'] for c in graph.chapters}
    study_index = {cid: i for i, cid in enumerate(study_ids)}
    entries_by_id: dict[str, list[Position]] = {}
    profile_key_by_id: dict[str, str] = {}
    is_alternative: dict[str, bool] = {}

    root_node = _Node()
    for chapter in chapters:
        cid = chapter['id']
        entries = [e['position'] for e in chapter['entries']]
        overrides = chapter.get('policy_overrides', {})
        pkey, profile = profile_for(overrides)
        entries_by_id[cid] = entries
        profile_key_by_id[cid] = pkey
        is_alternative[cid] = bool(overrides)
        anchor = _chapter_anchor(graph, cid, chapter_roots[cid], entries, profile, roots)
        node = root_node
        for token in anchor:
            node = node.children.setdefault(token, _Node())
        node.chapters.append(cid)

    reach_cache: dict[tuple[str, frozenset[Position]], float] = {}

    def cached_reach(pkey: str, positions: frozenset[Position]) -> float:
        cache_key = (pkey, positions)
        if cache_key not in reach_cache:
            reach_cache[cache_key] = _union_reach(profiles[pkey], roots, positions)
        return reach_cache[cache_key]

    def group_key(cids: Sequence[str]) -> tuple[float, bool, float, int]:
        all_entries = frozenset(p for cid in cids for p in entries_by_id[cid])
        overall_reach = cached_reach('{}', all_entries)
        # Secondary tie-breaker (primarily for unplayed alternatives where overall reach is 0): compute each
        # comparison policy profile's union first-entry reach without double-counting competing profiles.
        by_profile: dict[str, set[Position]] = {}
        for cid in cids:
            by_profile.setdefault(profile_key_by_id[cid], set()).update(entries_by_id[cid])
        comparison_reach = max(
            (cached_reach(pkey, frozenset(positions)) for pkey, positions in by_profile.items()),
            default=0.0,
        )
        return (
            -round(overall_reach, 12),
            all(is_alternative[cid] for cid in cids),
            -round(comparison_reach, 12),
            min(study_index[cid] for cid in cids),
        )

    def collect_ids(node: _Node) -> list[str]:
        result = list(node.chapters)
        for child in node.children.values():
            result.extend(collect_ids(child))
        return result

    def emit(node: _Node) -> list[str]:
        # Each chapter anchored directly here competes as a leaf sibling alongside child sub-openings, so a
        # rare sidelines catch-all sorts after major sub-openings while a high-reach main chapter sorts first.
        items: list[tuple[list[str], _Node | None]] = [([cid], None) for cid in node.chapters]
        for child in node.children.values():
            items.append((collect_ids(child), child))
        if len(items) > 1:
            items.sort(key=lambda item: group_key(item[0]))
        ordered: list[str] = []
        for cids, sub in items:
            ordered.extend(cids if sub is None else emit(sub))
        return ordered

    ordered_ids = emit(root_node)
    if len(ordered_ids) != len(study_ids) or set(ordered_ids) != set(study_ids):
        raise AssertionError('Chapter reordering must be a permutation of the repertoire chapters')
    return dict(mode=mode, ids=ordered_ids)


def saved_chapter_order(
    coverage_data: Mapping[str, Any] | None, chapters: Sequence[Mapping[str, Any]]
) -> tuple[str, list[str] | None]:
    """The saved chapter order mode and ID list from a matching coverage analysis, if present and valid."""
    if not coverage_data:
        return DEFAULT_ORDER, None
    data = coverage_data.get('chapter_order')
    if not isinstance(data, Mapping):
        return DEFAULT_ORDER, None
    mode = data.get('mode', DEFAULT_ORDER)
    ids = data.get('ids')
    if mode not in MODES or not isinstance(ids, list):
        return DEFAULT_ORDER, None
    expected = {c['id'] for c in chapters}
    if len(ids) != len(chapters) or set(ids) != expected:
        return DEFAULT_ORDER, None
    return mode, list(ids)


def apply_order[T](items: Sequence[T], ids: Sequence[str] | None, key_fn: Callable[[T], str]) -> list[T]:
    """Reorder `items` to match `ids`, else keep original order."""
    if not ids:
        return list(items)
    lookup = {key_fn(item): item for item in items}
    if len(lookup) != len(ids) or set(lookup) != set(ids):
        raise ValueError('Chapter order IDs do not match chapters')
    return [lookup[cid] for cid in ids]


def saved_order_for_score(
    score_path: str | Path, saved: Mapping[str, Any], report_sha256: str
) -> tuple[str, list[str] | None]:
    """Read the chapter order from a matching sibling coverage JSON, or ('study', None) when absent or stale."""
    companion = Path(score_path).with_suffix('.coverage.json')
    if not companion.is_file():
        return DEFAULT_ORDER, None
    try:
        data = json.loads(companion.read_text(encoding='utf-8'))
    except ValueError:
        return DEFAULT_ORDER, None
    manifest = data.get('manifest', {})
    score_manifest = saved.get('manifest', {})
    if (
        data.get('color') != saved.get('color')
        or manifest.get('schema_version') != SCHEMA_VERSION
        or manifest.get('report_sha256') != report_sha256
        or manifest.get('input_sha256') != score_manifest.get('input_sha256')
        or manifest.get('filters') != score_manifest.get('filters')
    ):
        return DEFAULT_ORDER, None
    return saved_chapter_order(data, saved.get('chapters', ()))
