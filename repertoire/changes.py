"""Compare the current build against the previous completed build and maintain the most recent change report.

Only one baseline snapshot (`.build-snapshot.json`) and one change report (`changes.json` + `changes.md`) are kept.
When a build introduces no substantive changes (no chapter, line, order, move-selection, configuration, evidence or
metric changes), the previous change report is preserved untouched.
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import chess.pgn

from . import SCHEMA_VERSION
from .attribution import write_text
from .board_cache import canonical, children, fen_number, fen_position, move_text, next_number, route_line, san, turn
from .chapter_order import DEFAULT_ORDER, saved_chapter_order
from .context import selected_policy
from .coverage import goal_from_config
from .graph import chapter_ids, parse_games, read_games
from .layout import data_json, report_directory
from .schema import JsonObject, Position
from .score import load_config
from .status import Status

SNAPSHOT_VERSION = 1
REPORT_VERSION = 1
SNAPSHOT_NAME = '.build-snapshot.json'
DATA_NAME = 'changes.json'
REPORT_NAME = 'changes.md'
EPSILON = 1e-9

# Recursive ordered move tree for one chapter: each item is [uci, children] in PGN variation order.
MoveTree = list[Any]


def _write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, data_json(value))


def _same_float(a: float | None, b: float | None, tol: float = EPSILON) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def _same_bounds(a: Sequence[float] | None, b: Sequence[float] | None, tol: float = EPSILON) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return len(a) == len(b) and all(_same_float(x, y, tol) for x, y in zip(a, b))


def extract_tree(game: chess.pgn.Game, cid: str, exclusions: Collection[str] = ()) -> MoveTree:
    """Ordered UCI move tree for one chapter, respecting configured exclusions and merging duplicate siblings."""
    root: MoveTree = []
    stack: list[tuple[chess.pgn.GameNode, chess.Board, MoveTree]] = [(game, game.board(), root)]
    while stack:
        node, board, target = stack.pop()
        k = canonical(board)
        by_move = {item[0]: item[1] for item in target}
        pending: list[tuple[chess.pgn.GameNode, chess.Board, MoveTree]] = []
        for child in node.variations:
            uci = child.move.uci()
            if f'{cid}:{k}:{uci}' in exclusions:
                continue
            after = board.copy(stack=False)
            after.push(child.move)
            if uci in by_move:
                sub = by_move[uci]
            else:
                sub = []
                target.append([uci, sub])
                by_move[uci] = sub
            pending.append((child, after, sub))
        stack.extend(reversed(pending))
    return root


def tree_stats(tree: MoveTree) -> tuple[int, int]:
    """(maximal root-to-leaf line count, total move edges) in a chapter's move tree."""
    lines = moves = 0
    stack: list[MoveTree] = [tree]
    while stack:
        items = stack.pop()
        for _, sub in items:
            moves += 1
            if sub:
                stack.append(sub)
            else:
                lines += 1
    return lines, moves


def tree_edges(root_fen: str, tree: MoveTree) -> set[tuple[Position, str]]:
    """Every (canonical_position, uci) edge recorded anywhere in a chapter's move tree."""
    edges: set[tuple[Position, str]] = set()
    stack: list[tuple[Position, MoveTree]] = [(fen_position(root_fen), tree)]
    while stack:
        pos, items = stack.pop()
        if not items:
            continue
        next_pos = children(pos)
        for uci, sub in items:
            edges.add((pos, uci))
            if sub:
                stack.append((next_pos[uci], sub))
    return edges


def _san_path_text(root_fen: str, path: Sequence[str]) -> str:
    """Format an unnumbered SAN path from `root_fen` with conventional move numbers."""
    if not path:
        return ''
    white_to_move = turn(fen_position(root_fen))
    move_num = fen_number(root_fen)
    tokens: list[str] = []
    for item in path:
        tokens.append(f'{move_num}.{item}' if white_to_move else f'{move_num}...{item}')
        if not white_to_move:
            move_num += 1
        white_to_move = not white_to_move
    return ' '.join(tokens)


def _load_matching_companion(score_path: Path, score: JsonObject, score_sha256: str, family: str) -> JsonObject | None:
    companion = score_path.with_suffix(f'.{family}.json')
    if not companion.is_file():
        return None
    try:
        data: JsonObject = json.loads(companion.read_text(encoding='utf-8'))
    except ValueError:
        return None
    manifest = data.get('manifest', {})
    score_manifest = score['manifest']
    if (
        data.get('color') != score.get('color')
        or manifest.get('schema_version') != SCHEMA_VERSION
        or manifest.get('report_sha256') != score_sha256
        or manifest.get('input_sha256') != score_manifest.get('input_sha256')
        or manifest.get('filters') != score_manifest.get('filters')
    ):
        return None
    return data


def _scope_map(data: JsonObject | None, key: str = 'scopes') -> dict[str, JsonObject]:
    return {s['id']: s for s in data.get(key, [])} if data else {}


def _capture_color(score_path: Path, pgn_path: Path, config_path: Path | None) -> JsonObject:
    raw_score = score_path.read_bytes()
    score_sha256 = hashlib.sha256(raw_score).hexdigest()
    score: JsonObject = json.loads(raw_score)
    manifest = score['manifest']
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'{score_path} was written by a different program version')
    color: str = score['color']
    color_bool = color == 'white'
    prefix = color[0].upper()

    character = _load_matching_companion(score_path, score, score_sha256, 'character')
    coverage_data = _load_matching_companion(score_path, score, score_sha256, 'coverage')
    engine_data = _load_matching_companion(score_path, score, score_sha256, 'engine')

    char_scopes = _scope_map(character)
    engine_resolved = (engine_data or {}).get('status') == Status.RESOLVED
    engine_scopes = _scope_map(engine_data) if engine_resolved else {}

    exclusions = list(manifest['configuration'].get('exclude', []))
    games = read_games(pgn_path)
    cids = chapter_ids(games, exclusions)
    graph = parse_games(games, exclusions)

    trees: dict[str, tuple[str, MoveTree]] = {}
    for game, cid in zip(games, cids):
        if cid is None:
            continue
        trees[cid] = (game.board().fen(en_passant='legal'), extract_tree(game, cid, exclusions))

    # Position labels and reach from character analysis when available.
    char_overall = char_scopes.get('overall', {})
    position_info: dict[Position, tuple[str, float | None]] = {
        row['position']: (row['line'], row.get('reach')) for row in char_overall.get('positions', [])
    }

    study_ids = [c['id'] for c in score['chapters']]
    order_mode, ordered_ids = saved_chapter_order(coverage_data, score['chapters'])
    report_ids = ordered_ids or list(study_ids)
    study_index = {cid: i for i, cid in enumerate(study_ids, 1)}
    report_index = {cid: i for i, cid in enumerate(report_ids, 1)}

    loaded_config = load_config(config_path) if config_path and Path(config_path).is_file() else {}
    coverage_goal = coverage_data['goal'] if coverage_data else goal_from_config(loaded_config)

    # Selected own moves across the merged repertoire.
    policy = selected_policy(manifest)
    explicit_policy = manifest['configuration'].get('policy', {})
    selected_alts = manifest.get('selected_alternatives', {})
    selected_moves: dict[Position, JsonObject] = {}
    chapter_root_fens = {cid: root_fen for cid, (root_fen, _) in trees.items()}
    study_order_rank = {cid: i for i, cid in enumerate(study_ids)}

    for pos, node in graph.nodes.items():
        if turn(pos) != color_bool or not node.edges:
            continue
        chosen = policy.get(pos)
        if chosen is None:
            chosen = next(iter(node.edges))
        weights = (
            {chosen: 1.0} if isinstance(chosen, str) else {m: float(w) for m, w in sorted(chosen.items()) if w > 0}
        )
        source = (
            'explicit_policy'
            if pos in explicit_policy
            else 'alternative'
            if pos in selected_alts
            else 'single'
            if len(node.edges) == 1
            else 'first_recorded'
        )
        first_cid = min(node.chapters, key=lambda cid: study_order_rank.get(cid, len(study_ids)))
        fallback_root = chapter_root_fens.get(first_cid, chess.STARTING_FEN)
        route, reach = position_info.get(pos, (_san_path_text(fallback_root, node.path), None))
        selected_moves[pos] = dict(
            moves=weights,
            sans={m: san(pos, m) for m in weights},
            source=source,
            line=route,
            reach=reach,
        )

    chapters: dict[str, JsonObject] = {}
    for c in score['chapters']:
        cid = c['id']
        root_fen, tree = trees[cid]
        line_count, move_count = tree_stats(tree)
        cs = c['score']
        c_char = char_scopes.get(cid, {})
        c_gap = c_char.get('gap_coverage') or {}
        c_reuse = c_char.get('reuse') or {}
        c_engine = engine_scopes.get(cid, {}).get('exits', {}) if engine_resolved else {}
        entries = []
        for entry in c['entries']:
            epos = entry['position']
            eline = position_info[epos][0] if epos in position_info else _san_path_text(root_fen, entry.get('path', []))
            entries.append(dict(position=epos, line=eline))
        chapters[cid] = dict(
            id=cid,
            name=c['name'],
            url=c.get('url'),
            study_index=study_index[cid],
            report_index=report_index[cid],
            code=f'{prefix}{report_index[cid]}',
            root_fen=root_fen,
            tree=tree,
            line_count=line_count,
            move_count=move_count,
            is_alternative=bool(c.get('policy_overrides')),
            entries=entries,
            entry_probability=cs.get('entry_probability'),
            overall_policy_entry_probability=cs.get('overall_policy_entry_probability'),
            score=cs.get('raw_empirical_score'),
            score_bounds=cs.get('conditional_bounds'),
            baseline=c.get('entry_baseline', {}).get('raw_score'),
            prepared_depth=cs.get('prepared_depth', {}).get('expected_answers'),
            equivalent_gap_reach=c_gap.get('equivalent_gap_reach'),
            equivalent_gap_reach_bounds=c_gap.get('equivalent_gap_reach_bounds'),
            distinct_decisions=c_reuse.get('reachable_distinct_decisions'),
            engine_exits_score=c_engine.get('engine_score') if engine_resolved else None,
        )

    alternatives: dict[Position, JsonObject] = {}
    for alt in score.get('alternatives', []):
        pos = alt['position']
        route = (
            position_info[pos][0] if pos in position_info else _san_path_text(chess.STARTING_FEN, alt.get('path', []))
        )
        alternatives[pos] = dict(
            position=pos,
            line=route,
            reach=alt.get('reach'),
            selected=alt['selected'],
            selected_san=san(pos, alt['selected']),
            options=[
                dict(
                    move=opt['move'],
                    san=opt['san'],
                    score=opt.get('score'),
                    prior_completed_score=opt.get('prior_completed_score'),
                    chapters=list(opt.get('chapters', [])),
                    sideline=bool(opt.get('sideline')),
                )
                for opt in alt.get('options', [])
            ],
        )

    overall = score['overall']
    overall_gap = char_overall.get('gap_coverage') or {}
    overall_reuse = char_overall.get('reuse') or {}

    coverage_summary: JsonObject | None = None
    if coverage_data:
        row = (coverage_data.get('at_goal', {}).get('protected') or {}).get('row') or next(
            (r for r in coverage_data.get('levels', []) if r.get('goal') == coverage_data['goal']), None
        )
        full = coverage_data.get('levels', [None])[-1]
        if row is not None and full is not None:
            coverage_summary = dict(
                goal=coverage_data['goal'],
                decisions=row['decisions'],
                total_decisions=full['decisions'],
                score=row['score'].get('raw_empirical_score'),
                score_bounds=row['score'].get('conditional_bounds'),
                leaving_earlier=row.get('leaving_earlier'),
            )

    engine_summary: JsonObject | None = None
    if engine_data is not None:
        exits = engine_scopes.get('overall', {}).get('exits', {}) if engine_resolved else {}
        total_reach = exits.get('reach') or 0.0
        engine_summary = dict(
            resolved=engine_resolved,
            evaluated_positions=len(engine_data.get('positions', {})),
            exits_score=exits.get('engine_score') if engine_resolved else None,
            exits_database_score=exits.get('database_score') if engine_resolved else None,
            evaluated_share=(exits.get('evaluated_reach', 0.0) / total_reach)
            if engine_resolved and total_reach
            else None,
        )

    evidence_index = {
        pos: f"{info.get('cache_key', '')[:16]}:{info.get('retrieved_at', '')}"
        for pos, info in manifest.get('evidence', {}).items()
    }

    return dict(
        color=color,
        input_sha256=manifest['input_sha256'],
        overall_basis=manifest.get('overall_basis'),
        overall=dict(
            score=overall.get('raw_empirical_score'),
            score_bounds=overall.get('conditional_bounds'),
            unresolved_mass=overall.get('unresolved_mass'),
            baseline=score.get('starting_position_reference', {}).get('owner_score'),
            prepared_depth=overall.get('prepared_depth', {}).get('expected_answers'),
            equivalent_gap_reach=overall_gap.get('equivalent_gap_reach'),
            equivalent_gap_reach_bounds=overall_gap.get('equivalent_gap_reach_bounds'),
            distinct_decisions=overall_reuse.get('reachable_distinct_decisions'),
            positions=manifest.get('positions'),
            chapters_count=len(score['chapters']),
            coverage=coverage_summary,
            engine=engine_summary,
        ),
        chapter_order_mode=order_mode,
        study_order=study_ids,
        report_order=report_ids,
        chapters=chapters,
        selected_moves=selected_moves,
        alternatives=alternatives,
        config=dict(
            chapter_order=order_mode,
            coverage_goal=coverage_goal,
            critical_coverage=loaded_config.get('critical_coverage'),
            keep=loaded_config.get('keep'),
            trimmed_pgn_notations=bool(loaded_config.get('trimmed_pgn_notations', False)),
            policy=manifest['configuration'].get('policy', {}),
            exclude=exclusions,
            entries=manifest['configuration'].get('entries', {}),
            root_weights=manifest['configuration'].get('root_weights', {}),
            filters=manifest['filters'],
            prior=manifest['prior'],
            sparse_threshold=manifest['sparse_threshold'],
        ),
        evidence=evidence_index,
    )


def _combined_snapshot(colors: Mapping[str, JsonObject]) -> JsonObject | None:
    if set(colors) != {'white', 'black'}:
        return None
    white, black = colors['white'], colors['black']
    if white['config']['filters'] != black['config']['filters']:
        return None
    if any(c.get('overall_basis') != 'standard starting position' for c in (white, black)):
        return None

    def average(values: Sequence[float | None]) -> float | None:
        return None if any(v is None for v in values) else sum(cast('Sequence[float]', values)) / 2

    wo, bo = white['overall'], black['overall']
    score = average([wo['score'], bo['score']])
    color_bounds = [wo['score_bounds'], bo['score_bounds']]
    bounds = (
        [cast(float, average([b[i] for b in cast('list[list[float]]', color_bounds)])) for i in (0, 1)]
        if score is None and all(color_bounds)
        else None
    )
    decisions = (
        wo['distinct_decisions'] + bo['distinct_decisions']
        if wo['distinct_decisions'] is not None and bo['distinct_decisions'] is not None
        else None
    )
    return dict(
        score=score,
        score_bounds=bounds,
        baseline=average([wo['baseline'], bo['baseline']]),
        prepared_depth=average([wo['prepared_depth'], bo['prepared_depth']]),
        chapters_count=wo['chapters_count'] + bo['chapters_count'],
        distinct_decisions=decisions,
    )


def _file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def source_hashes(
    score_paths: Sequence[str | Path],
    pgns: Mapping[str, str | Path],
    configs: Mapping[str, str | Path | None],
    *,
    digest_fn: Callable[[str | Path], str | None] = _file_sha256,
) -> dict[str, dict[str, str | None]]:
    """SHA-256 digests of every input file a snapshot reads, so unchanged builds skip re-parsing."""
    result: dict[str, dict[str, str | None]] = {}
    for color, score_path in zip(('white', 'black'), score_paths):
        path = Path(score_path)
        cfg = configs.get(color)
        result[color] = dict(
            score=digest_fn(path),
            pgn=digest_fn(pgns[color]),
            config=digest_fn(cfg) if cfg else None,
            character=digest_fn(path.with_suffix('.character.json')),
            coverage=digest_fn(path.with_suffix('.coverage.json')),
            engine=digest_fn(path.with_suffix('.engine.json')),
        )
    return result


def capture(
    score_paths: Sequence[str | Path],
    pgns: Mapping[str, str | Path],
    configs: Mapping[str, str | Path | None],
    *,
    built_at: str | None = None,
    hashes: Mapping[str, Mapping[str, str | None]] | None = None,
) -> JsonObject:
    """Build a compact snapshot of the current completed build for future change reports."""
    colors: dict[str, JsonObject] = {}
    for score_path in score_paths:
        path = Path(score_path)
        color = json.loads(path.read_text(encoding='utf-8'))['color']
        cfg = configs.get(color)
        colors[color] = _capture_color(path, Path(pgns[color]), Path(cfg) if cfg else None)
    return dict(
        version=SNAPSHOT_VERSION,
        schema_version=SCHEMA_VERSION,
        built_at=built_at or datetime.now(UTC).isoformat(),
        source_hashes=hashes if hashes is not None else source_hashes(score_paths, pgns, configs),
        colors=colors,
        combined=_combined_snapshot(colors),
    )


def snapshot_path(directory: str | Path) -> Path:
    return Path(directory) / SNAPSHOT_NAME


def data_path(directory: str | Path) -> Path:
    return Path(directory) / DATA_NAME


def report_path(directory: str | Path) -> Path:
    return report_directory(Path(directory) / 'white.json') / REPORT_NAME


def load_snapshot(directory: str | Path) -> JsonObject | None:
    path = snapshot_path(directory)
    if not path.is_file():
        return None
    try:
        data: JsonObject = json.loads(path.read_text(encoding='utf-8'))
    except ValueError:
        return None
    if (
        data.get('version') != SNAPSHOT_VERSION
        or data.get('schema_version') != SCHEMA_VERSION
        or not isinstance(data.get('colors'), dict)
    ):
        return None
    return data


def save_snapshot(directory: str | Path, snapshot: JsonObject) -> None:
    _write_json_atomic(snapshot_path(directory), snapshot)


def bootstrap_snapshot(
    directory: str | Path,
    pgns: Mapping[str, str | Path],
    configs: Mapping[str, str | Path | None],
) -> bool:
    """Capture an initial baseline from an already-built directory before `studies.export` overwrites PGNs."""
    directory = Path(directory)
    if snapshot_path(directory).is_file():
        return False
    score_paths = [directory / f'{color}.json' for color in ('white', 'black')]
    if not all(path.is_file() for path in score_paths):
        return False
    try:
        for color, score_path in zip(('white', 'black'), score_paths):
            pgn = Path(pgns[color])
            if not pgn.is_file():
                return False
            score = json.loads(score_path.read_text(encoding='utf-8'))
            manifest = score.get('manifest', {})
            if manifest.get('schema_version') != SCHEMA_VERSION:
                return False
            if hashlib.sha256(pgn.read_bytes()).hexdigest() != manifest.get('input_sha256'):
                return False
        built_at = None
        state_file = directory / '.build-state.json'
        if state_file.is_file():
            try:
                built_at = json.loads(state_file.read_text(encoding='utf-8')).get('last_run', {}).get('created_at')
            except ValueError:
                built_at = None
        snapshot = capture(score_paths, pgns, configs, built_at=built_at)
        save_snapshot(directory, snapshot)
        return True
    except (OSError, ValueError, KeyError):
        return False


def _lcs(a: Sequence[str], b: Sequence[str]) -> list[str]:
    """Longest common subsequence of distinct chapter IDs, breaking ties by smallest total rank displacement."""
    m, n = len(a), len(b)
    pos_a = {cid: i for i, cid in enumerate(a)}
    pos_b = {cid: j for j, cid in enumerate(b)}
    dp: list[list[tuple[int, int]]] = [[(0, 0)] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if a[i] == b[j]:
                length, neg_disp = dp[i + 1][j + 1]
                dp[i][j] = (length + 1, neg_disp - abs(pos_a[a[i]] - pos_b[b[j]]))
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    result: list[str] = []
    i = j = 0
    while i < m and j < n:
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return result


def _diff_order(
    old_ids: Sequence[str],
    new_ids: Sequence[str],
    new_chapters: Mapping[str, JsonObject],
    prefix: str,
) -> JsonObject:
    old_set, new_set = set(old_ids), set(new_ids)
    old_common = [cid for cid in old_ids if cid in new_set]
    new_common = [cid for cid in new_ids if cid in old_set]
    old_pos = {cid: i for i, cid in enumerate(old_ids, 1)}
    new_pos = {cid: i for i, cid in enumerate(new_ids, 1)}
    reordered = old_common != new_common
    keep = set(_lcs(old_common, new_common)) if reordered else set(new_common)

    def entry(cid: str) -> JsonObject:
        return dict(
            id=cid,
            name=new_chapters[cid]['name'],
            old_index=old_pos[cid],
            new_index=new_pos[cid],
            old_code=f'{prefix}{old_pos[cid]}',
            new_code=f'{prefix}{new_pos[cid]}',
        )

    moved = [entry(cid) for cid in new_common if cid not in keep]
    shifted = [entry(cid) for cid in new_common if cid in keep and old_pos[cid] != new_pos[cid]]
    return dict(reordered=reordered, moved=moved, shifted=shifted)


def _collect_leaf_paths(tree: MoveTree) -> list[list[str]]:
    """All maximal move sequences from a subtree root to its leaves, in PGN order."""
    if not tree:
        return [[]]
    paths: list[list[str]] = []
    stack: list[tuple[list[str], MoveTree]] = [([], tree)]
    while stack:
        prefix, items = stack.pop()
        for uci, sub in reversed(items):
            extended = [*prefix, uci]
            if sub:
                stack.append((extended, sub))
            else:
                paths.append(extended)
    return paths


def _edge_index(
    chapters: Mapping[str, JsonObject],
) -> tuple[dict[str, set[tuple[Position, str]]], dict[tuple[Position, str], set[str]]]:
    by_chapter: dict[str, set[tuple[Position, str]]] = {}
    by_edge: dict[tuple[Position, str], set[str]] = {}
    for cid, chapter in chapters.items():
        edges = tree_edges(chapter['root_fen'], chapter['tree'])
        by_chapter[cid] = edges
        for edge in edges:
            by_edge.setdefault(edge, set()).add(cid)
    return by_chapter, by_edge


def _line_record(
    *,
    color_bool: bool,
    cid: str,
    branch_pos: Position,
    branch_num: int,
    prefix_text: str,
    moves_uci: Sequence[str],
    is_mainline: bool,
    had_siblings: bool,
    is_addition: bool,
    old_repertoire_edges: Mapping[tuple[Position, str], set[str]],
    new_repertoire_edges: Mapping[tuple[Position, str], set[str]],
    edges_removed_from: Mapping[tuple[Position, str], set[str]],
    edges_added_to: Mapping[tuple[Position, str], set[str]],
) -> JsonObject:
    continuation, end_pos, _ = route_line(branch_pos, branch_num, moves_uci)
    full_line = f'{prefix_text} {continuation}'.strip()
    pos = branch_pos
    edges: list[tuple[Position, str]] = []
    own_moves = 0
    for uci in moves_uci:
        edges.append((pos, uci))
        if turn(pos) == color_bool:
            own_moves += 1
        pos = children(pos)[uci]

    scope: str
    related: list[str] = []
    if is_addition:
        if any(e not in old_repertoire_edges for e in edges):
            scope = 'new'
        else:
            moved_from = sorted({other for e in edges for other in edges_removed_from.get(e, ()) if other != cid})
            also_in = sorted({other for e in edges for other in new_repertoire_edges.get(e, ()) if other != cid})
            if moved_from:
                scope, related = 'moved', moved_from
            elif also_in:
                scope, related = 'existing', also_in
            else:
                scope = 'existing_in_chapter'
    else:
        if any(e not in new_repertoire_edges for e in edges):
            scope = 'removed'
        else:
            moved_to = sorted({other for e in edges for other in edges_added_to.get(e, ()) if other != cid})
            still_in = sorted({other for e in edges for other in new_repertoire_edges.get(e, ()) if other != cid})
            if moved_to:
                scope, related = 'moved', moved_to
            elif still_in:
                scope, related = 'kept_elsewhere', still_in
            else:
                scope = 'kept_in_chapter'

    return dict(
        branch_position=branch_pos,
        branch_line=prefix_text,
        end_position=end_pos,
        continuation=continuation,
        full_line=full_line,
        plies=len(moves_uci),
        own_moves=own_moves,
        is_mainline=is_mainline,
        replaced_mainline=is_mainline and had_siblings,
        scope=scope,
        related_chapters=related,
    )


def _diff_chapter_tree(
    *,
    color_bool: bool,
    cid: str,
    old_chapter: JsonObject,
    new_chapter: JsonObject,
    old_repertoire_edges: Mapping[tuple[Position, str], set[str]],
    new_repertoire_edges: Mapping[tuple[Position, str], set[str]],
    edges_removed_from: Mapping[tuple[Position, str], set[str]],
    edges_added_to: Mapping[tuple[Position, str], set[str]],
) -> JsonObject:
    old_root_fen, new_root_fen = old_chapter['root_fen'], new_chapter['root_fen']
    old_tree, new_tree = old_chapter['tree'], new_chapter['tree']
    added: list[JsonObject] = []
    deleted: list[JsonObject] = []
    reordered: list[JsonObject] = []
    added_moves = deleted_moves = 0

    def record_subtree(
        branch_pos: Position,
        branch_num: int,
        prefix_text: str,
        uci: str,
        sub: MoveTree,
        *,
        is_mainline: bool,
        had_siblings: bool,
        is_addition: bool,
    ) -> int:
        _, sub_moves = tree_stats(sub)
        for tail in _collect_leaf_paths(sub):
            rec = _line_record(
                color_bool=color_bool,
                cid=cid,
                branch_pos=branch_pos,
                branch_num=branch_num,
                prefix_text=prefix_text,
                moves_uci=[uci, *tail],
                is_mainline=is_mainline,
                had_siblings=had_siblings,
                is_addition=is_addition,
                old_repertoire_edges=old_repertoire_edges,
                new_repertoire_edges=new_repertoire_edges,
                edges_removed_from=edges_removed_from,
                edges_added_to=edges_added_to,
            )
            (added if is_addition else deleted).append(rec)
        return 1 + sub_moves

    root_changed = old_root_fen != new_root_fen
    if root_changed:
        old_pos, old_num = fen_position(old_root_fen), fen_number(old_root_fen)
        for i, (uci, sub) in enumerate(old_tree):
            deleted_moves += record_subtree(
                old_pos, old_num, '', uci, sub, is_mainline=i == 0, had_siblings=False, is_addition=False
            )
        new_pos, new_num = fen_position(new_root_fen), fen_number(new_root_fen)
        for i, (uci, sub) in enumerate(new_tree):
            added_moves += record_subtree(
                new_pos, new_num, '', uci, sub, is_mainline=i == 0, had_siblings=False, is_addition=True
            )
    else:
        start_pos, start_num = fen_position(new_root_fen), fen_number(new_root_fen)
        stack: list[tuple[Position, int, str, MoveTree, MoveTree]] = [(start_pos, start_num, '', old_tree, new_tree)]
        while stack:
            pos, num, prefix_text, o_items, n_items = stack.pop()
            o_map = {uci: sub for uci, sub in o_items}
            n_map = {uci: sub for uci, sub in n_items}
            o_order = [uci for uci, _ in o_items]
            n_order = [uci for uci, _ in n_items]

            for i, uci in enumerate(n_order):
                if uci not in o_map:
                    added_moves += record_subtree(
                        pos,
                        num,
                        prefix_text,
                        uci,
                        n_map[uci],
                        is_mainline=i == 0,
                        had_siblings=bool(o_order),
                        is_addition=True,
                    )

            for i, uci in enumerate(o_order):
                if uci not in n_map:
                    deleted_moves += record_subtree(
                        pos,
                        num,
                        prefix_text,
                        uci,
                        o_map[uci],
                        is_mainline=i == 0,
                        had_siblings=len(o_order) > 1,
                        is_addition=False,
                    )

            o_common = [m for m in o_order if m in n_map]
            n_common = [m for m in n_order if m in o_map]
            if o_common != n_common:
                reordered.append(
                    dict(
                        position=pos,
                        line=prefix_text,
                        side='own' if turn(pos) == color_bool else 'opponent',
                        old_order=[move_text(pos, num, m) for m in o_common],
                        new_order=[move_text(pos, num, m) for m in n_common],
                        old_mainline=move_text(pos, num, o_order[0]) if o_order else None,
                        new_mainline=move_text(pos, num, n_order[0]) if n_order else None,
                        mainline_changed=bool(o_order and n_order and o_order[0] != n_order[0]),
                    )
                )

            next_pos = children(pos)
            nxt_num = next_number(pos, num)
            shared = [uci for uci in n_order if uci in o_map]
            for uci in reversed(shared):
                token = move_text(pos, num, uci)
                extended = f'{prefix_text} {token}'.strip()
                stack.append((next_pos[uci], nxt_num, extended, o_map[uci], n_map[uci]))

    return dict(
        id=cid,
        name=new_chapter['name'],
        code=new_chapter['code'],
        url=new_chapter.get('url'),
        root_changed=root_changed,
        added=added,
        deleted=deleted,
        reordered=reordered,
        added_moves=added_moves,
        deleted_moves=deleted_moves,
    )


def _same_policy_move(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same_float(float(a[k]), float(b[k])) for k in a)
    return False


def _format_policy_value(pos: Position, value: Any) -> str:
    if isinstance(value, str):
        return san(pos, value)
    if isinstance(value, Mapping):
        return ', '.join(f'{san(pos, m)} ({100 * float(w):.0f}%)' for m, w in sorted(value.items()))
    return str(value)


def _diff_config(
    old_cfg: JsonObject,
    new_cfg: JsonObject,
    old_selected: Mapping[Position, JsonObject],
    new_selected: Mapping[Position, JsonObject],
) -> list[JsonObject]:
    items: list[JsonObject] = []
    if old_cfg.get('chapter_order', DEFAULT_ORDER) != new_cfg.get('chapter_order', DEFAULT_ORDER):
        items.append(
            dict(
                key='chapter_order',
                old=old_cfg.get('chapter_order', DEFAULT_ORDER),
                new=new_cfg.get('chapter_order', DEFAULT_ORDER),
            )
        )
    if old_cfg.get('coverage_goal') != new_cfg.get('coverage_goal'):
        items.append(dict(key='coverage_goal', old=old_cfg.get('coverage_goal'), new=new_cfg.get('coverage_goal')))
    if old_cfg.get('critical_coverage') != new_cfg.get('critical_coverage'):
        items.append(
            dict(key='critical_coverage', old=old_cfg.get('critical_coverage'), new=new_cfg.get('critical_coverage'))
        )
    if old_cfg.get('keep') != new_cfg.get('keep'):
        items.append(dict(key='keep', old=old_cfg.get('keep'), new=new_cfg.get('keep')))
    if old_cfg.get('trimmed_pgn_notations', False) != new_cfg.get('trimmed_pgn_notations', False):
        items.append(
            dict(
                key='trimmed_pgn_notations',
                old=bool(old_cfg.get('trimmed_pgn_notations', False)),
                new=bool(new_cfg.get('trimmed_pgn_notations', False)),
            )
        )

    old_policy, new_policy = old_cfg.get('policy', {}), new_cfg.get('policy', {})
    policy_changes: list[JsonObject] = []
    for pos in sorted(set(old_policy) | set(new_policy)):
        in_old, in_new = pos in old_policy, pos in new_policy
        if in_old and in_new and _same_policy_move(old_policy[pos], new_policy[pos]):
            continue
        info = new_selected.get(pos) or old_selected.get(pos) or {}
        policy_changes.append(
            dict(
                position=pos,
                line=info.get('line', ''),
                action='added' if not in_old else 'removed' if not in_new else 'changed',
                old=_format_policy_value(pos, old_policy[pos]) if in_old else None,
                new=_format_policy_value(pos, new_policy[pos]) if in_new else None,
            )
        )
    if policy_changes:
        items.append(dict(key='policy', changes=policy_changes))

    old_exc, new_exc = set(old_cfg.get('exclude', [])), set(new_cfg.get('exclude', []))
    if old_exc != new_exc:
        items.append(dict(key='exclude', added=sorted(new_exc - old_exc), removed=sorted(old_exc - new_exc)))

    old_ent, new_ent = old_cfg.get('entries', {}), new_cfg.get('entries', {})
    if old_ent != new_ent:
        changed_cids = sorted(cid for cid in set(old_ent) | set(new_ent) if old_ent.get(cid) != new_ent.get(cid))
        items.append(
            dict(
                key='entries',
                added=[cid for cid in changed_cids if cid not in old_ent],
                removed=[cid for cid in changed_cids if cid not in new_ent],
                changed=[cid for cid in changed_cids if cid in old_ent and cid in new_ent],
            )
        )

    if old_cfg.get('root_weights', {}) != new_cfg.get('root_weights', {}):
        items.append(dict(key='root_weights', old=old_cfg.get('root_weights', {}), new=new_cfg.get('root_weights', {})))

    old_filters, new_filters = old_cfg.get('filters', {}), new_cfg.get('filters', {})
    if old_filters != new_filters:
        filter_changes = {
            k: dict(old=old_filters.get(k), new=new_filters.get(k))
            for k in sorted(set(old_filters) | set(new_filters))
            if old_filters.get(k) != new_filters.get(k)
        }
        items.append(dict(key='filters', changes=filter_changes))

    if old_cfg.get('prior') != new_cfg.get('prior'):
        items.append(dict(key='prior', old=old_cfg.get('prior'), new=new_cfg.get('prior')))

    if old_cfg.get('sparse_threshold') != new_cfg.get('sparse_threshold'):
        items.append(
            dict(key='sparse_threshold', old=old_cfg.get('sparse_threshold'), new=new_cfg.get('sparse_threshold'))
        )
    return items


def _overall_changed(old_o: JsonObject, new_o: JsonObject) -> bool:
    if not (
        _same_float(old_o.get('score'), new_o.get('score'))
        and _same_bounds(old_o.get('score_bounds'), new_o.get('score_bounds'))
        and _same_float(old_o.get('baseline'), new_o.get('baseline'))
        and _same_float(old_o.get('prepared_depth'), new_o.get('prepared_depth'))
        and _same_float(old_o.get('equivalent_gap_reach'), new_o.get('equivalent_gap_reach'))
        and _same_bounds(old_o.get('equivalent_gap_reach_bounds'), new_o.get('equivalent_gap_reach_bounds'))
        and old_o.get('distinct_decisions') == new_o.get('distinct_decisions')
        and old_o.get('chapters_count') == new_o.get('chapters_count')
    ):
        return True
    oc, nc = old_o.get('coverage'), new_o.get('coverage')
    if (oc is None) != (nc is None):
        return True
    if oc is not None and nc is not None:
        if not (
            oc.get('goal') == nc.get('goal')
            and oc.get('decisions') == nc.get('decisions')
            and oc.get('total_decisions') == nc.get('total_decisions')
            and _same_float(oc.get('score'), nc.get('score'))
            and _same_bounds(oc.get('score_bounds'), nc.get('score_bounds'))
            and _same_float(oc.get('leaving_earlier'), nc.get('leaving_earlier'))
        ):
            return True
    return False


def _engine_changed(old_e: JsonObject | None, new_e: JsonObject | None) -> bool:
    if old_e is None or new_e is None:
        return old_e is not new_e
    return not (
        old_e.get('resolved') == new_e.get('resolved')
        and old_e.get('evaluated_positions') == new_e.get('evaluated_positions')
        and _same_float(old_e.get('exits_score'), new_e.get('exits_score'))
        and _same_float(old_e.get('evaluated_share'), new_e.get('evaluated_share'))
    )


def _chapter_metrics_changed(old_c: JsonObject, new_c: JsonObject) -> bool:
    return not (
        _same_float(old_c.get('entry_probability'), new_c.get('entry_probability'))
        and _same_float(old_c.get('overall_policy_entry_probability'), new_c.get('overall_policy_entry_probability'))
        and _same_float(old_c.get('score'), new_c.get('score'))
        and _same_bounds(old_c.get('score_bounds'), new_c.get('score_bounds'))
        and _same_float(old_c.get('baseline'), new_c.get('baseline'))
        and _same_float(old_c.get('prepared_depth'), new_c.get('prepared_depth'))
        and _same_float(old_c.get('equivalent_gap_reach'), new_c.get('equivalent_gap_reach'))
        and _same_bounds(old_c.get('equivalent_gap_reach_bounds'), new_c.get('equivalent_gap_reach_bounds'))
        and old_c.get('distinct_decisions') == new_c.get('distinct_decisions')
        and _same_float(old_c.get('engine_exits_score'), new_c.get('engine_exits_score'))
    )


def _chapter_summary(chapter: JsonObject) -> JsonObject:
    return {
        k: chapter.get(k)
        for k in (
            'id',
            'name',
            'url',
            'code',
            'study_index',
            'report_index',
            'line_count',
            'move_count',
            'is_alternative',
            'entry_probability',
            'overall_policy_entry_probability',
            'score',
            'score_bounds',
            'baseline',
            'prepared_depth',
            'equivalent_gap_reach',
            'equivalent_gap_reach_bounds',
            'distinct_decisions',
            'engine_exits_score',
        )
    }


def _diff_played_moves(
    old_selected: Mapping[Position, JsonObject], new_selected: Mapping[Position, JsonObject]
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for pos in set(old_selected) & set(new_selected):
        old_item, new_item = old_selected[pos], new_selected[pos]
        if _same_policy_move(old_item['moves'], new_item['moves']):
            continue
        osrc, nsrc = old_item['source'], new_item['source']
        if nsrc == 'explicit_policy' and osrc != 'explicit_policy':
            reason = 'Policy override added'
        elif nsrc == 'explicit_policy' and osrc == 'explicit_policy':
            reason = 'Policy override changed'
        elif osrc == 'explicit_policy' and nsrc != 'explicit_policy':
            reason = 'Policy override removed'
        elif nsrc == 'alternative' and osrc == 'alternative':
            reason = 'Alternative winner changed'
        elif nsrc == 'alternative':
            reason = 'Competing alternative won'
        elif osrc == 'alternative':
            reason = 'Competing alternative removed'
        else:
            reason = 'Move replaced'
        rows.append(
            dict(
                position=pos,
                line=new_item['line'] or old_item['line'],
                reach=new_item['reach'],
                old_moves=old_item['moves'],
                new_moves=new_item['moves'],
                old_label=_format_selected(old_item),
                new_label=_format_selected(new_item),
                reason=reason,
            )
        )
    return sorted(rows, key=lambda r: (-(r['reach'] or 0.0), r['line'], r['position']))


def _format_selected(item: JsonObject) -> str:
    moves: Mapping[str, float] = item['moves']
    sans: Mapping[str, str] = item['sans']
    if len(moves) == 1:
        uci = next(iter(moves))
        return sans[uci]
    return ', '.join(f'{sans[m]} ({100 * w:.0f}%)' for m, w in sorted(moves.items()))


def _diff_alternatives(old_alts: Mapping[Position, JsonObject], new_alts: Mapping[Position, JsonObject]) -> JsonObject:
    added = [
        new_alts[p] for p in sorted(set(new_alts) - set(old_alts), key=lambda p: (-(new_alts[p]['reach'] or 0.0), p))
    ]
    removed = [
        old_alts[p] for p in sorted(set(old_alts) - set(new_alts), key=lambda p: (-(old_alts[p]['reach'] or 0.0), p))
    ]
    changed: list[JsonObject] = []
    for pos in sorted(set(old_alts) & set(new_alts), key=lambda p: (-(new_alts[p]['reach'] or 0.0), p)):
        old_a, new_a = old_alts[pos], new_alts[pos]
        old_moves = [o['move'] for o in old_a['options']]
        new_moves = [o['move'] for o in new_a['options']]
        if old_a['selected'] != new_a['selected'] or set(old_moves) != set(new_moves):
            changed.append(
                dict(
                    position=pos,
                    line=new_a['line'],
                    reach=new_a['reach'],
                    old_selected=old_a['selected_san'],
                    new_selected=new_a['selected_san'],
                    old_options=[o['san'] for o in old_a['options']],
                    new_options=[o['san'] for o in new_a['options']],
                )
            )
    return dict(added=added, removed=removed, changed=changed)


def _diff_color(old: JsonObject, new: JsonObject) -> JsonObject:
    color: str = new['color']
    color_bool = color == 'white'
    prefix = color[0].upper()

    old_chapters: Mapping[str, JsonObject] = old['chapters']
    new_chapters: Mapping[str, JsonObject] = new['chapters']
    old_ids_set, new_ids_set = set(old_chapters), set(new_chapters)

    added_chapters = [_chapter_summary(new_chapters[cid]) for cid in new['report_order'] if cid not in old_ids_set]
    deleted_chapters = [_chapter_summary(old_chapters[cid]) for cid in old['report_order'] if cid not in new_ids_set]
    renamed_chapters = [
        dict(
            id=cid,
            code=new_chapters[cid]['code'],
            url=new_chapters[cid].get('url'),
            old_name=old_chapters[cid]['name'],
            new_name=new_chapters[cid]['name'],
        )
        for cid in new['report_order']
        if cid in old_ids_set and old_chapters[cid]['name'] != new_chapters[cid]['name']
    ]

    report_order = _diff_order(old['report_order'], new['report_order'], new_chapters, prefix)
    study_order = _diff_order(old['study_order'], new['study_order'], new_chapters, '#')

    old_by_chapter, old_repertoire_edges = _edge_index(old_chapters)
    new_by_chapter, new_repertoire_edges = _edge_index(new_chapters)
    edges_removed_from: dict[tuple[Position, str], set[str]] = {}
    edges_added_to: dict[tuple[Position, str], set[str]] = {}
    for cid in old_ids_set | new_ids_set:
        o_edges = old_by_chapter.get(cid, set())
        n_edges = new_by_chapter.get(cid, set())
        for edge in o_edges - n_edges:
            edges_removed_from.setdefault(edge, set()).add(cid)
        for edge in n_edges - o_edges:
            edges_added_to.setdefault(edge, set()).add(cid)

    line_changes: list[JsonObject] = []
    line_changes_by_id: dict[str, JsonObject] = {}
    for cid in new['report_order']:
        if cid not in old_ids_set:
            continue
        diff = _diff_chapter_tree(
            color_bool=color_bool,
            cid=cid,
            old_chapter=old_chapters[cid],
            new_chapter=new_chapters[cid],
            old_repertoire_edges=old_repertoire_edges,
            new_repertoire_edges=new_repertoire_edges,
            edges_removed_from=edges_removed_from,
            edges_added_to=edges_added_to,
        )
        if diff['root_changed'] or diff['added'] or diff['deleted'] or diff['reordered']:
            line_changes.append(diff)
            line_changes_by_id[cid] = diff

    changed_chapters: list[JsonObject] = []
    for cid in new['report_order']:
        if cid not in old_ids_set:
            continue
        oc, nc = old_chapters[cid], new_chapters[cid]
        old_entries = [e['position'] for e in oc['entries']]
        new_entries = [e['position'] for e in nc['entries']]
        entries_changed = old_entries != new_entries
        alt_changed = bool(oc.get('is_alternative')) != bool(nc.get('is_alternative'))
        metrics_changed = _chapter_metrics_changed(oc, nc)
        ch_lines = line_changes_by_id.get(cid)
        if metrics_changed or entries_changed or alt_changed or ch_lines is not None:
            changed_chapters.append(
                dict(
                    id=cid,
                    name=nc['name'],
                    code=nc['code'],
                    url=nc.get('url'),
                    before=_chapter_summary(oc),
                    after=_chapter_summary(nc),
                    entries_changed=entries_changed,
                    old_entries=oc['entries'],
                    new_entries=nc['entries'],
                    alternative_changed=alt_changed,
                    lines_added=len(ch_lines['added']) if ch_lines else 0,
                    lines_deleted=len(ch_lines['deleted']) if ch_lines else 0,
                    lines_reordered=len(ch_lines['reordered']) if ch_lines else 0,
                )
            )

    played_moves = _diff_played_moves(old.get('selected_moves', {}), new.get('selected_moves', {}))
    alternatives = _diff_alternatives(old.get('alternatives', {}), new.get('alternatives', {}))
    config_changes = _diff_config(
        old['config'], new['config'], old.get('selected_moves', {}), new.get('selected_moves', {})
    )

    old_ev, new_ev = old.get('evidence', {}), new.get('evidence', {})
    refreshed = sum(1 for pos in set(old_ev) & set(new_ev) if old_ev[pos] != new_ev[pos])
    evidence_diff = dict(
        tables_before=len(old_ev),
        tables_after=len(new_ev),
        refreshed_positions=refreshed,
        changed=refreshed > 0,
    )

    old_engine, new_engine = old['overall'].get('engine'), new['overall'].get('engine')
    engine_diff = dict(before=old_engine, after=new_engine, changed=_engine_changed(old_engine, new_engine))
    overall_diff = dict(
        before=old['overall'], after=new['overall'], changed=_overall_changed(old['overall'], new['overall'])
    )

    lines_added_total = sum(len(item['added']) for item in line_changes)
    lines_deleted_total = sum(len(item['deleted']) for item in line_changes)
    lines_reordered_total = sum(len(item['reordered']) for item in line_changes)
    alt_changed_count = len(alternatives['added']) + len(alternatives['removed']) + len(alternatives['changed'])

    counts = dict(
        chapters_added=len(added_chapters),
        chapters_deleted=len(deleted_chapters),
        chapters_renamed=len(renamed_chapters),
        chapters_reordered=len(report_order['moved']),
        study_chapters_reordered=len(study_order['moved']),
        chapters_changed=len(changed_chapters),
        lines_added=lines_added_total,
        lines_deleted=lines_deleted_total,
        lines_reordered=lines_reordered_total,
        played_moves_changed=len(played_moves),
        alternatives_changed=alt_changed_count,
        config_changed=len(config_changes),
        evidence_refreshed=refreshed,
        engine_changed=int(engine_diff['changed']),
    )

    has_changes = any(
        (
            added_chapters,
            deleted_chapters,
            renamed_chapters,
            report_order['reordered'],
            study_order['reordered'],
            changed_chapters,
            line_changes,
            played_moves,
            alt_changed_count > 0,
            config_changes,
            evidence_diff['changed'],
            engine_diff['changed'],
            overall_diff['changed'],
        )
    )

    current_chapters = {
        cid: dict(code=ch['code'], name=ch['name'], url=ch.get('url')) for cid, ch in new_chapters.items()
    }
    previous_chapters = {
        cid: dict(code=ch['code'], name=ch['name'], url=ch.get('url')) for cid, ch in old_chapters.items()
    }

    return dict(
        color=color,
        has_changes=has_changes,
        counts=counts,
        overall=overall_diff,
        config=config_changes,
        evidence=evidence_diff,
        engine=engine_diff,
        chapter_order_mode=dict(before=old['chapter_order_mode'], after=new['chapter_order_mode']),
        chapters=dict(
            added=added_chapters,
            deleted=deleted_chapters,
            renamed=renamed_chapters,
            report_order=report_order,
            study_order=study_order,
            changed=changed_chapters,
        ),
        played_moves=played_moves,
        alternatives=alternatives,
        lines=line_changes,
        current_chapters=current_chapters,
        previous_chapters=previous_chapters,
    )


def _combined_changed(old_c: JsonObject | None, new_c: JsonObject | None) -> bool:
    if old_c is None or new_c is None:
        return old_c is not new_c
    return not (
        _same_float(old_c.get('score'), new_c.get('score'))
        and _same_bounds(old_c.get('score_bounds'), new_c.get('score_bounds'))
        and _same_float(old_c.get('baseline'), new_c.get('baseline'))
        and _same_float(old_c.get('prepared_depth'), new_c.get('prepared_depth'))
        and old_c.get('chapters_count') == new_c.get('chapters_count')
        and old_c.get('distinct_decisions') == new_c.get('distinct_decisions')
    )


def compare(previous: JsonObject, current: JsonObject) -> JsonObject:
    """Compute a complete structured diff between two build snapshots."""
    colors: dict[str, JsonObject] = {}
    for color in ('white', 'black'):
        if color in previous['colors'] and color in current['colors']:
            colors[color] = _diff_color(previous['colors'][color], current['colors'][color])
    old_combined, new_combined = previous.get('combined'), current.get('combined')
    combined = (
        dict(before=old_combined, after=new_combined, changed=_combined_changed(old_combined, new_combined))
        if old_combined is not None or new_combined is not None
        else None
    )
    has_changes = any(c['has_changes'] for c in colors.values()) or bool(combined and combined['changed'])
    return dict(
        version=REPORT_VERSION,
        previous_built_at=previous.get('built_at'),
        current_built_at=current.get('built_at'),
        has_changes=has_changes,
        combined=combined,
        colors=colors,
    )


def _plural(count: int, noun: str) -> str:
    return f'{count:,} {noun}' + ('' if count == 1 else 's')


def color_summary_parts(color_diff: JsonObject) -> list[str]:
    """Short human-readable phrases summarizing what changed for one color."""
    c = color_diff['counts']
    parts: list[str] = []
    if c['chapters_added']:
        parts.append(f"{_plural(c['chapters_added'], 'chapter')} added")
    if c['chapters_deleted']:
        parts.append(f"{_plural(c['chapters_deleted'], 'chapter')} deleted")
    if c['chapters_renamed']:
        parts.append(f"{_plural(c['chapters_renamed'], 'chapter')} renamed")
    if c['chapters_reordered']:
        parts.append(f"{_plural(c['chapters_reordered'], 'chapter')} reordered")
    elif c['study_chapters_reordered']:
        parts.append(f"{_plural(c['study_chapters_reordered'], 'study chapter')} reordered")
    if c['lines_added']:
        parts.append(f"{_plural(c['lines_added'], 'line')} added")
    if c['lines_deleted']:
        parts.append(f"{_plural(c['lines_deleted'], 'line')} deleted")
    if c['lines_reordered']:
        parts.append(f"{_plural(c['lines_reordered'], 'variation')} reordered")
    if c['played_moves_changed']:
        parts.append(f"{_plural(c['played_moves_changed'], 'played move')} changed")
    if c['config_changed']:
        parts.append('configuration updated')
    if c['evidence_refreshed']:
        parts.append(f"evidence refreshed ({_plural(c['evidence_refreshed'], 'position')})")
    engine = color_diff.get('engine') or {}
    engine_status_flipped = bool(
        engine.get('changed')
        and (engine.get('before') or {}).get('resolved') != (engine.get('after') or {}).get('resolved')
    )
    if c['engine_changed'] and (engine_status_flipped or not parts):
        parts.append('engine evaluations updated')
    if not parts and color_diff['has_changes']:
        parts.append('metrics updated')
    return parts


def summary_text(diff: JsonObject) -> str:
    """One-line summary of a change report for CLI output and the top of changes.md."""
    if not diff.get('has_changes'):
        return 'No changes since the last build.'
    clauses: list[str] = []
    unchanged: list[str] = []
    for color in ('white', 'black'):
        item = diff['colors'].get(color)
        if not item:
            continue
        if item['has_changes']:
            clauses.append(f"{color.title()}: {', '.join(color_summary_parts(item))}.")
        else:
            unchanged.append(color.title())
    if unchanged:
        clauses.append(f"{' and '.join(unchanged)} had no changes.")
    return ' '.join(clauses)


def write_report(directory: str | Path, folder: str | Path, diff: JsonObject) -> tuple[Path, Path]:
    """Atomically write the most recent change report (JSON + Markdown), replacing any previous one."""
    from .report.changes import render

    json_file = Path(directory) / DATA_NAME
    md_file = Path(folder) / REPORT_NAME
    md_file.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(json_file, diff)
    write_text(md_file, render(diff))
    return json_file, md_file


def main() -> None:
    parser = argparse.ArgumentParser(description='Show or re-render the most recent build change report.')
    parser.add_argument('--directory', default='reports/data', help='Build data directory (default: reports/data)')
    parser.add_argument('--render', action='store_true', help='Re-render changes.md from the saved changes.json')
    args = parser.parse_args()
    json_file = data_path(args.directory)
    md_file = report_path(args.directory)
    if not json_file.is_file():
        print('No change report yet. Run `repertoire build` after making a change to generate one.', flush=True)
        return
    try:
        diff: JsonObject = json.loads(json_file.read_text(encoding='utf-8'))
    except ValueError as exc:
        parser.exit(1, f'Could not read {json_file}: {exc}\n')
    if args.render or not md_file.is_file():
        from .report.changes import render

        md_file.parent.mkdir(parents=True, exist_ok=True)
        write_text(md_file, render(diff))
    print(summary_text(diff), flush=True)
    print(f'Change report: {md_file.resolve()}', flush=True)


if __name__ == '__main__':
    main()
