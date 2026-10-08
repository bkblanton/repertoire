"""Compare alternative preparation from candidate studies or PGNs against your saved repertoire."""

import argparse
import io
import json
import math
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import chess
import chess.pgn
import httpx
import numpy as np

from . import SCHEMA_VERSION, studies
from .alternatives import (
    CANDIDATE_PREFIX,
    Choice,
    Decision,
    Option,
    Scenarios,
    assignments,
    changed,
    count_assignments,
    find_decisions,
    groups,
    repertoire_reference,
    required_tables,
)
from .board_cache import STARTING_POSITION, canonical, children, fen_number, move_text, san, turn
from .character import scope_metrics
from .context import DEFAULT_CACHE, AnalysisContext, file_sha256
from .evaluate import COMPLETED, KNOWN, UNKNOWN, Route, Weights, backward, reaches, select_alternatives
from .explorer import CacheMiss, Explorer, add_token_option, apply_token_file, counts, survey
from .graph import Graph, Transitions, parse_games, read_games, resolve, topology
from .layout import data_json, report_directory
from .model import Evidence, MissingEvidence, Model, Sampled, empirical, prepare, score
from .preparation import Evaluator, chess_facts, line_text, position_lines
from .ratings import Context, unavailable
from .render import update_report_outputs
from .schema import JsonObject, Position
from .score import load_config
from .uncertainty import METHOD as UNCERTAINTY_METHOD
from .uncertainty import Posterior, difference_interval, paired_variance

COMPARISON_VERSION = 1
CANDIDATES = Path(studies.DIRECTORY) / 'candidates'
REGISTRY = 'comparisons.json'
# Exhaustive search per group of linked decision points; larger groups improve one point at a time.
SEARCH_LIMIT = 256
REPLY_ROWS = 8
GAP_ROWS = 5


@dataclass
class Scenario:
    choice: Choice
    graph: Graph
    policy: dict[Position, str | Mapping[str, float]]
    transitions: Transitions
    order: list[Position]
    model: Model
    raw: Sampled
    values: dict[Position, np.ndarray]
    selection: dict[Position, JsonObject]
    superseded: list[Position]


def plural(count: int, noun: str) -> str:
    return f'{count:,} {noun}' + ('' if count == 1 else 's')


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-') or 'comparison'


def is_study(value: str) -> bool:
    try:
        studies.study_id(value)
    except ValueError:
        return False
    return True


def parse_entry(text: str) -> Position:
    """A position from FEN, or from moves such as '1.e4 e5 2.Nc3' played from the starting position."""
    text = text.strip()
    if '/' in text:
        board = chess.Board(text if len(text.split()) == 6 else text + ' 0 1')
        return canonical(board)
    board = chess.Board()
    for token in text.split():
        token = re.sub(r'^\d+\.(\.\.)?', '', token)
        if token:
            board.push_san(token)
    return canonical(board)


def text_games(text: str, label: str | Path) -> list[chess.pgn.Game]:
    games = []
    stream = io.StringIO(text)
    while (game := chess.pgn.read_game(stream)) is not None:
        games.append(game)
    if not games:
        raise ValueError(f'No PGN chapters in {label}')
    return games


def comparable(text: str) -> list[str]:
    return studies.comparable(text)


def save_export(path: Path, text: str) -> bool:
    """Write an exported study after checking it parses; leave the file alone when only Date headers changed."""
    parse_games(text_games(text, path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and comparable(path.read_text(encoding='utf-8-sig')) == comparable(text):
        return False
    temporary = path.with_suffix('.pgn.tmp')
    temporary.write_text(text, encoding='utf-8', newline='\n')
    temporary.replace(path)
    return True


def source_files(inputs: Sequence[str], name: str, directory: str | Path = CANDIDATES) -> list[Path]:
    """Where each input's PGN is read: the file itself, or the saved export of a study URL."""
    studies_given = [value for value in inputs if not Path(value).is_file()]
    result, count = [], 0
    for value in inputs:
        if Path(value).is_file():
            result.append(Path(value))
        else:
            count += 1
            result.append(Path(directory) / (f'{name}.pgn' if len(studies_given) == 1 else f'{name}-{count}.pgn'))
    return result


def load_inputs(
    inputs: Sequence[str],
    name: str | None = None,
    export: bool = True,
    directory: str | Path = CANDIDATES,
    client: httpx.Client | None = None,
) -> tuple[list[chess.pgn.Game], str, list[JsonObject]]:
    """Candidate games in priority order, the comparison name and where each input came from.

    Study URLs are exported into `directory`; with `export` False the last export is reused.
    """
    downloads: dict[int, str]
    files: dict[int, Path]
    downloads, files = {}, {}
    for i, value in enumerate(inputs):
        if Path(value).is_file():
            files[i] = Path(value)
        elif is_study(value):
            downloads[i] = value
        else:
            raise ValueError(f'Not a PGN file or Lichess study: {value}')
    texts: dict[int, str] = {}
    if downloads and export:
        token = os.environ.get('LICHESS_TOKEN', '').strip()
        owned = client is None
        client = client or httpx.Client(headers={'Authorization': f'Bearer {token}'} if token else {}, timeout=60)
        try:
            for i, value in downloads.items():
                texts[i] = studies.download(client, value, url=studies.chapter_url(value), orientation=True)
        finally:
            if owned:
                client.close()
    games = {i: read_games(path) for i, path in files.items()}
    games.update({i: text_games(text, inputs[i]) for i, text in texts.items()})
    if name is None:
        first = next((games[i] for i in sorted(games)), None)
        headers = first[0].headers if first else {}
        name = slug(
            headers.get('StudyName')
            or (files[min(files)].stem if files and min(files) == min(games, default=-1) else '')
            or headers.get('ChapterName', 'comparison')
        )
    sources: list[JsonObject]
    sources, paths = [], source_files(inputs, name, directory)
    for i, value in enumerate(inputs):
        if i in downloads:
            path = paths[i]
            if i in texts:
                updated = save_export(path, texts[i])
                print(
                    f'{value}: {len(games[i])} chapters, {"updated" if updated else "unchanged"} ({path})', flush=True
                )
            elif path.is_file():
                games[i] = read_games(path)
            else:
                raise ValueError(f'No saved export of {value} at {path}; run without --no-export or --offline')
            sources.append(dict(input=value, kind='lichess', export_url=studies.chapter_url(value), path=str(path)))
        else:
            sources.append(dict(input=value, kind='file', path=str(Path(value).resolve())))
        sources[-1].update(sha256=file_sha256(sources[-1]['path']), chapters=len(games[i]))
    return [g for i in range(len(inputs)) for g in games[i]], name, sources


def common_entry(candidate: Graph, decisions: Sequence[Decision]) -> Position:
    """The last board every top-level decision's line shares, or the decision itself when there is one."""
    top = [d for d in decisions if d.parent is None]
    if len(top) == 1:
        return top[0].position
    if any(c['root'] != STARTING_POSITION for c in candidate.chapters):
        return STARTING_POSITION
    common = []
    for moves in zip(*(candidate.nodes[d.position].path for d in top)):
        if len(set(moves)) > 1:
            break
        common.append(moves[0])
    board = chess.Board()
    for move in common:
        board.push_san(move)
    return canonical(board)


def planned_tables(entry: JsonObject, pgn: str | Path, config: JsonObject) -> list[Position]:
    """The tables a saved comparison needs, planned from the repertoire PGN before it is scored.

    Competing chapter alternatives are not scored yet, so the plan follows each chapter's first moves; a
    comparison step may still fetch the odd table when the scored repertoire plays a different move.
    """
    games, _, _ = load_inputs(entry['sources'], entry['name'], export=False)
    color = entry['color'] == 'white'
    repertoire_games = read_games(pgn)
    repertoire = parse_games(repertoire_games, config.get('exclude', []))
    policy = config.get('policy', {})
    candidate = parse_games(games)
    decisions = find_decisions(candidate, color, repertoire_reference(resolve(repertoire, color, policy)))
    if not decisions:
        return []
    scenarios = Scenarios(candidate, games, decisions, repertoire_games, color, config.get('exclude', []))
    graph, transitions, positions = scenarios.superset(policy, [STARTING_POSITION, *repertoire.roots])
    target = parse_entry(entry['entry']) if entry.get('entry') else common_entry(candidate, decisions)
    return list(
        dict.fromkeys(
            [*required_tables(graph, transitions, positions, color), *(d.position for d in decisions), target]
        )
    )


def registry_tables(
    saved: Iterable[JsonObject], pgns: Mapping[str, str | Path], configs: Mapping[str, str | Path | None]
) -> dict[str, list[Position]]:
    """{color: tables} for every saved comparison, so one fetch pass covers them too."""
    result: dict[str, list[Position]] = {}
    for entry in saved:
        try:
            tables = planned_tables(entry, pgns[entry['color']], load_config(configs[entry['color']]))
        except (ValueError, FileNotFoundError) as exc:
            print(f"Comparison {entry['name']}: not planned ({exc})", flush=True)
            continue
        result.setdefault(entry['color'], []).extend(tables)
    return result


def infer_color(games: Iterable[chess.pgn.Game]) -> str | None:
    tags = {g.headers.get('Orientation', '').lower() for g in games}
    if tags <= {'white'} and tags:
        return 'white'
    if tags <= {'black'} and tags:
        return 'black'
    return None


class Comparison:
    """Every scenario a comparison needs, evaluated on demand over the same cached evidence."""

    def __init__(
        self, analysis: AnalysisContext, candidate_games: list[chess.pgn.Game], entry: str | None = None
    ) -> None:
        self.analysis = analysis
        self.color = analysis.color
        manifest = analysis.manifest
        self.config_policy = manifest['configuration'].get('policy', {})
        self.sparse, self.prior = manifest['sparse_threshold'], manifest['prior']
        self.roots = manifest['root_weights']
        self.repertoire_games = read_games(analysis.source)
        self.repertoire = analysis.graph
        self.reference_transitions = resolve(self.repertoire, self.color, analysis.policy)
        self.candidate = parse_games(candidate_games)
        self.decisions = find_decisions(self.candidate, self.color, repertoire_reference(self.reference_transitions))
        if not self.decisions:
            raise ValueError('The candidate plays the same moves as your repertoire everywhere it has preparation')
        self.scenarios = Scenarios(
            self.candidate,
            candidate_games,
            self.decisions,
            self.repertoire_games,
            self.color,
            manifest['configuration'].get('exclude', []),
        )
        self.entry, self.entry_basis = (parse_entry(entry), 'given') if entry else (self.common_entry(), 'common')
        self.evidence: Evidence = {}
        self.cache: dict[tuple[tuple[int, int], ...], Scenario]
        self.posteriors: dict[tuple[tuple[int, int], ...], Posterior]
        self.cache, self.posteriors = {}, {}

    def common_entry(self) -> Position:
        return common_entry(self.candidate, self.decisions)

    @property
    def starts(self) -> list[Position]:
        return [*self.roots, *self.repertoire.roots]

    def required(self) -> list[Position]:
        """Every table a scenario or the report can use: replies, unanswered boards, decisions and the entry."""
        graph, transitions, positions = self.scenarios.superset(self.config_policy, self.starts)
        tables = required_tables(graph, transitions, positions, self.color)
        extra = [STARTING_POSITION, self.entry, *(d.position for d in self.decisions)]
        return list(dict.fromkeys([*self.analysis.manifest['evidence'], *tables, *extra]))

    def evaluate(self, choice: Choice) -> Scenario:
        key = choice.key()
        if key in self.cache:
            return self.cache[key]
        graph = self.scenarios.graph(choice)
        fixed = self.scenarios.fixed(choice)
        # Candidate lines decide the boards they record; explicit overrides apply everywhere else.
        adopted = {
            k
            for k, n in graph.nodes.items()
            if turn(k) == self.color and any(c.startswith(CANDIDATE_PREFIX) and m for c, m in n.chapter_moves.items())
        }
        policy = {k: v for k, v in self.config_policy.items() if k not in adopted}
        superseded = sorted(k for k in self.config_policy if k in adopted)
        policy.update(fixed)
        extra = [k for k in [self.entry, *(d.position for d in self.decisions)] if k in graph.nodes]
        try:
            selection = select_alternatives(
                graph, self.color, policy, self.evidence, [*self.starts, *extra], self.sparse
            )
            full = dict(policy, **{k: c['selected'] for k, c in selection.items()})
            transitions = resolve(graph, self.color, full)
            order = topology(transitions, list(dict.fromkeys([*self.roots, *extra])))
            model = prepare(graph, transitions, order, self.color, self.evidence)
        except MissingEvidence as exc:
            raise ValueError(f'Missing Explorer table for {exc}; run without --offline to fetch it') from None
        raw = empirical(model, self.color)
        values = backward(model, order, raw, self.sparse)
        result = Scenario(choice, graph, full, transitions, order, model, raw, values, selection, superseded)
        self.cache[key] = result
        return result

    def root(self, scenario: Scenario) -> float:
        return sum(w * scenario.values[k][KNOWN] for k, w in self.roots.items())

    def value(self, scenario: Scenario, k: Position) -> float | None:
        """Score at a board; without preparation there, the board's own database score."""
        if k in scenario.values:
            v = scenario.values[k]
            return float(v[KNOWN]) if v[UNKNOWN] == 0 else None
        return score(counts(self.evidence[k]), self.color) if k in self.evidence else None

    def choice_for(self, decision: Decision, index: int) -> Choice:
        """Choose `index` at `decision`, with every enclosing option chosen so the decision applies."""
        choice, d, i = Choice(), decision, index
        while True:
            choice[d.id] = i
            if d.parent is None:
                return choice
            d, i = self.decisions[d.parent[0]], cast(tuple[int, int], d.parent)[1]

    def objective(self, choice: Choice, members: Iterable[int]) -> tuple[float, float, float, int]:
        s = self.evaluate(choice)
        local = sum(self.value(s, self.decisions[i].position) or 0.0 for i in members)
        entry = self.value(s, self.entry)
        return (self.root(s), -1.0 if entry is None else entry, local, -len(choice.key()))

    def search(
        self, members: Sequence[int], allowed: Callable[[Decision], list[int]], label: str
    ) -> tuple[Choice, JsonObject]:
        """The best choice within one group of linked decisions, and how it was found."""
        total = count_assignments(self.decisions, members, allowed, SEARCH_LIMIT)
        if total <= SEARCH_LIMIT:
            best = max(assignments(self.decisions, members, allowed), key=lambda c: self.objective(c, members))
            return best, dict(method='exhaustive', combinations=total)
        print(f'{label}: {total}+ combinations; improving one decision point at a time', flush=True)
        current = Choice({i: allowed(self.decisions[i])[0] for i in members if self.decisions[i].parent is None})
        improved = True
        while improved:
            improved = False
            for i in members:
                d = self.decisions[i]
                if d.id not in {x.id for x in current.active(self.decisions)}:
                    continue
                for index in allowed(d):
                    trial = Choice({**current, d.id: index})
                    if self.objective(trial, members) > self.objective(current, members):
                        current, improved = trial, True
        return current, dict(method='coordinate ascent', combinations=None)

    def posterior(self, scenario: Scenario) -> Posterior:
        key = scenario.choice.key()
        if key not in self.posteriors:
            self.posteriors[key] = Posterior(scenario.model, scenario.order, self.color, self.prior, self.sparse)
        return self.posteriors[key]

    def interval(self, scenario: Scenario, starts: Weights) -> list[float] | None:
        """Paired 95% interval of the score change from the current repertoire, in percentage points."""
        current = self.evaluate(Choice())
        if any(k not in s.values for s in (current, scenario) for k in starts):
            return None
        a, b = self.posterior(current), self.posterior(scenario)
        ga, gb = a.gradient(starts), b.gradient(starts)
        mean = gb['mean'][COMPLETED] - ga['mean'][COMPLETED]
        return difference_interval(mean, paired_variance(a, ga, b, gb))

    def metrics(self, scenario: Scenario, start: Position) -> JsonObject | None:
        """Preparation after one board: depth, own moves to know, gaps, reply variety and opponent ratings."""
        if start not in scenario.graph.nodes:
            return None
        evaluator = Evaluator(
            scenario.graph,
            self.color,
            self.evidence,
            chess_facts(scenario.graph, self.color, self.evidence),
            scenario.policy,
            sparse=self.sparse,
        )
        lines = position_lines(scenario.graph)
        result = scope_metrics(evaluator, {start: 1.0}, lines)
        value = evaluator.evaluate({start: 1.0})
        incoming = Context(evaluator, self.roots).local.get(start, unavailable('no preceding opponent move'))
        rating = Context(evaluator, {start: 1.0}, {start: incoming}).score_evidence()
        routes = evaluator.routes({start: 1.0}, replies=True)
        prefix = lines.get(start, '')
        number = len(prefix.split()) // 2 + 1 if prefix and prefix != '(PGN root)' else 1
        gaps = []
        for row in result['gap_coverage']['gaps'][:GAP_ROWS]:
            route = routes.get(row['position'])
            text = '' if route is None else route_text(start, prefix, number, route[2])
            gaps.append(
                dict(position=row['position'], reach=row['reach'], line=text, score=self.gap_score(start, route))
            )
        return dict(
            score=float(value[0]) if value[1] == 0 else None,
            expected_moves=float(value[2]),
            distinct_decisions=result['reuse']['reachable_distinct_decisions'],
            equivalent_gap_reach=result['gap_coverage']['equivalent_gap_reach'],
            largest_gap_reach=result['gap_coverage']['largest_gap_reach'],
            effective_replies=result['predictability']['effective_replies'],
            sparse_mass=sum(r['reach'] for r in result['stopping_outcomes'] if r.get('sparse')),
            opponent_rating=rating,
            gaps=gaps,
        )

    def gap_score(self, start: Position, route: Route | None) -> float | None:
        """Database score where preparation ends: the reply's row in its parent table, or the board's own table."""
        if route is None:
            return None
        position = start
        for move in route[2][:-1]:
            position = children(position)[move]
        if route[2] and turn(position) != self.color and position in self.evidence:
            row = next((r for r in self.evidence[position]['moves'] if r['uci'] == route[2][-1]), None)
            return score(counts(row), self.color) if row else None
        end = children(position)[route[2][-1]] if route[2] else start
        return score(counts(self.evidence[end]), self.color) if end in self.evidence else None

    def replies(self, scenario: Scenario, decision: Decision, move: str | tuple[str, ...] | None) -> list[JsonObject]:
        """The opponent's replies after a move at a decision board, with the score after each."""
        if not isinstance(move, str):
            return []
        after = children(decision.position)[move]
        if after not in scenario.model:
            return []
        rows: list[JsonObject] = []
        for b, (p, s) in zip(scenario.model[after].branches, scenario.raw[after]):
            if not b.move or not p:
                continue
            value = scenario.values[b.target] if b.target is not None else None
            rows.append(
                dict(
                    move=b.move,
                    san=san(after, b.move),
                    share=p,
                    prepared=b.target is not None,
                    score=(float(value[KNOWN]) if value[UNKNOWN] == 0 else None) if value is not None else s,
                    games=sum(b.counts),
                )
            )
        return sorted(rows, key=lambda r: -r['share'])[:REPLY_ROWS]


def route_text(start: Position, prefix: str, number: int, moves: Iterable[str]) -> str:
    from .board_cache import route_line

    text = route_line(start, number, moves)[0]
    return f'{prefix} {text}'.strip() if prefix and prefix != '(PGN root)' else text


def collect_evidence(comparison: Comparison, cache: str | Path, offline: bool, dry_run: bool) -> JsonObject | None:
    """Read every table the comparison may need, fetching missing ones unless offline. Returns fetched count."""
    positions = comparison.required()
    filters = comparison.analysis.manifest['filters']
    explorer = Explorer(cache, filters, offline=offline or dry_run)
    try:
        missing = survey(explorer, positions, 'comparison', fetching=not offline)
        if dry_run:
            return None
        if missing and offline:
            raise ValueError(
                f'{plural(len(missing), "Explorer table")} not cached; run without --offline to fetch '
                + ('it' if len(missing) == 1 else 'them')
            )
        from .explorer import Progress

        progress, pending = Progress(len(missing), 'comparison'), set(missing)
        for k in positions:
            try:
                comparison.evidence[k] = progress.fetch(explorer, k) if k in pending else explorer.get(k)
            except CacheMiss:
                raise ValueError(f'Explorer table not cached: {k}') from None
    finally:
        explorer.close()
    saved = comparison.analysis.manifest['evidence']
    if any(explorer.provenance.get(k) != v for k, v in saved.items()):
        raise ValueError('Cached evidence changed since your repertoire was scored; rebuild it first')
    return dict(tables=len(positions), fetched=len(missing), provenance=explorer.provenance)


def chapter_entries(comparison: Comparison, option: Option) -> list[JsonObject]:
    by_id = {c['id']: c for c in comparison.candidate.chapters}
    return [dict(id=cid, name=by_id[cid]['name'], url=by_id[cid]['url']) for cid in option.chapters]


def repertoire_chapters(
    comparison: Comparison, decision: Decision, move: str | tuple[str, ...] | None
) -> list[JsonObject]:
    """Your chapters whose recorded move at this board is the one you play."""
    node = comparison.repertoire.nodes.get(decision.position)
    if node is None or not isinstance(move, str) or move not in node.provenance:
        return []
    color = comparison.analysis.saved['color']
    pages = {c['id']: (i, c['name']) for i, c in enumerate(comparison.analysis.saved['chapters'], 1)}
    return [
        dict(id=cid, name=pages[cid][1], page=f'{color[0].upper()}{pages[cid][0]}')
        for cid in sorted(node.provenance[move], key=lambda cid: pages.get(cid, (len(pages) + 1,))[0])
        if cid in pages
    ]


def label(position: Position, move: str | tuple[str, ...] | None) -> str:
    if move is None:
        return 'No preparation'
    if isinstance(move, tuple):
        return 'Mixed: ' + ', '.join(san(position, m) for m in move)
    return san(position, move)


def analyze(
    analysis: AnalysisContext,
    games: list[chess.pgn.Game],
    sources: list[JsonObject],
    *,
    entry: str | None = None,
    cache: str | Path = DEFAULT_CACHE,
    offline: bool = False,
    dry_run: bool = False,
    name: str,
    title: str,
) -> JsonObject | None:
    comparison = Comparison(analysis, games, entry)
    decisions = comparison.decisions
    print(
        f'{plural(len(decisions), "decision point")} in {plural(len(comparison.candidate.chapters), "chapter")}',
        flush=True,
    )
    evidence = collect_evidence(comparison, cache, offline, dry_run)
    if evidence is None:
        return None
    current = comparison.evaluate(Choice())
    saved = analysis.saved['overall']['resolved_contribution']
    if not math.isclose(comparison.root(current), saved, abs_tol=1e-12):
        raise ValueError('Your repertoire does not reproduce its saved score; rebuild it first')
    if comparison.entry not in current.values:
        raise ValueError('The entry position is not part of your repertoire; pass --entry')
    reach = reaches(current.model, current.order, current.raw, comparison.roots)

    # Each option alone, with everything else as it is now.
    singles: dict[tuple[int, int], Scenario] = {}
    for d in decisions:
        for index in range(1, len(d.options)):
            singles[d.id, index] = comparison.evaluate(comparison.choice_for(d, index))
    _, superset, _ = comparison.scenarios.superset(comparison.config_policy, comparison.starts)
    footprints = {
        d.id: set().union(*(changed(singles[d.id, i], current) for i in range(1, len(d.options)))) for d in decisions
    }
    linked = groups(decisions, footprints, superset)

    def improving(d: Decision) -> list[int]:
        return list(range(len(d.options)))

    def candidate_only(d: Decision) -> list[int]:
        allowed = [i for i, o in enumerate(d.options) if o.source == 'candidate']
        return allowed or [0]

    plans: dict[str, JsonObject] = {}
    for label_, allowed in (('improving', improving), ('all', candidate_only)):
        searches: list[JsonObject]
        merged, searches = Choice(), []
        for members in linked:
            best, how = comparison.search(members, allowed, label_)
            merged.update(best)
            gain = comparison.root(comparison.evaluate(best)) - comparison.root(current)
            searches.append(dict(decisions=members, gain=gain, **how))
        scenario = comparison.evaluate(merged)
        total = comparison.root(scenario) - comparison.root(current)
        plans[label_] = dict(
            choice=merged,
            scenario=scenario,
            searches=searches,
            additive=math.isclose(total, sum(s['gain'] for s in searches), abs_tol=1e-9),
        )

    def summary(scenario: Scenario) -> JsonObject:
        root = comparison.root(scenario)
        entry = comparison.value(scenario, comparison.entry)
        return dict(
            score=root,
            unresolved_mass=float(sum(w * scenario.values[k][UNKNOWN] for k, w in comparison.roots.items())),
            entry_score=entry,
            change=root - comparison.root(current),
            entry_change=None if entry is None else entry - comparison.value(current, comparison.entry),  # type: ignore[operator]
            interval=comparison.interval(scenario, comparison.roots),
            entry_interval=comparison.interval(scenario, {comparison.entry: 1.0}),
            entry_metrics=comparison.metrics(scenario, comparison.entry),
            chosen=[[d, i] for d, i in sorted(scenario.choice.items()) if i],
            superseded_overrides=scenario.superseded,
        )

    scenarios = {
        'current': summary(current),
        'improving': summary(plans['improving']['scenario']),
        'all': summary(plans['all']['scenario']),
    }
    rows: list[JsonObject] = []
    for d in decisions:
        base = (
            current
            if d.parent is None
            else comparison.evaluate(comparison.choice_for(decisions[d.parent[0]], d.parent[1]))
        )
        table = comparison.evidence.get(d.position, {})
        moves = {r['uci']: r for r in table.get('moves', [])}
        options: list[JsonObject] = []
        option: Any
        for index, option in enumerate(d.options):
            scenario = base if index == 0 else singles[d.id, index]
            here = comparison.value(scenario, d.position)
            row = moves.get(option.move) if isinstance(option.move, str) else None
            database = score(counts(row), comparison.color) if row else None
            options.append(
                dict(
                    index=index,
                    move=option.move if isinstance(option.move, str) else None,
                    san=label(d.position, option.move),
                    source=option.source,
                    chapters=chapter_entries(comparison, option) if option.source == 'candidate' else [],
                    repertoire_chapters=repertoire_chapters(comparison, d, option.move) if index == 0 else [],
                    score=here,
                    change=None,
                    overall_change=comparison.root(scenario) - comparison.root(base),
                    entry_change=None,
                    interval=None if index == 0 else comparison.interval(scenario, {d.position: 1.0}),
                    overall_interval=None if index == 0 else comparison.interval(scenario, comparison.roots),
                    database_score=database,
                    database_games=sum(counts(row)) if row else None,
                    metrics=comparison.metrics(scenario, d.position),
                    replies=comparison.replies(scenario, d, option.move),
                    improving=plans['improving']['choice'].get(d.id, 0) == index,
                    all=plans['all']['choice'].get(d.id, 0) == index,
                )
            )
        reference = options[0]['score']
        entry_base = comparison.value(base, comparison.entry)
        for option, index in zip(options, range(len(options))):
            scenario = base if index == 0 else singles[d.id, index]
            option['change'] = None if option['score'] is None or reference is None else option['score'] - reference
            entry_value = comparison.value(scenario, comparison.entry)
            option['entry_change'] = None if entry_value is None or entry_base is None else entry_value - entry_base
        number = fen_number(comparison.candidate.nodes[d.position].fen)
        for option in options:
            option['label'] = move_text(d.position, number, option['move']) if option['move'] else option['san']
        path = comparison.candidate.nodes[d.position].path
        root_fen = next(
            c for c in comparison.candidate.chapters if c['id'] in comparison.candidate.nodes[d.position].chapters
        )['root']
        rows.append(
            dict(
                id=d.id,
                position=d.position,
                line=line_text(path, comparison.candidate.nodes[root_fen].fen),
                parent=list(d.parent) if d.parent else None,
                group=next(i for i, members in enumerate(linked) if d.id in members),
                reach=float(reach.get(d.position, 0.0)) if d.parent is None else None,
                games=sum(counts(table)) if table else None,
                database_score=score(counts(table), comparison.color) if table else None,
                options=options,
            )
        )
    interactions: list[JsonObject] = []
    for plan in ('improving', 'all'):
        for i, members in enumerate(linked):
            choice = Choice({d: plans[plan]['choice'].get(d, 0) for d in members})
            adopted = [d for d, index in choice.items() if index and decisions[d].parent is None]
            if len(adopted) < 2:
                continue
            together = comparison.root(comparison.evaluate(choice)) - comparison.root(current)
            separately = sum(comparison.root(singles[d, choice[d]]) - comparison.root(current) for d in adopted)
            interactions.append(dict(plan=plan, group=i, decisions=adopted, together=together, separately=separately))
    manifest = analysis.manifest
    black_or_white = analysis.saved['color']
    return dict(
        kind='comparison',
        comparison_version=COMPARISON_VERSION,
        schema_version=SCHEMA_VERSION,
        name=name,
        title=title,
        color=black_or_white,
        created_at=datetime.now(UTC).isoformat(),
        sources=sources,
        chapters=[dict(id=c['id'], name=c['name'], url=c['url']) for c in comparison.candidate.chapters],
        entry=dict(
            position=comparison.entry,
            line=line_text(current.graph.nodes[comparison.entry].path, current.graph.nodes[current.graph.roots[0]].fen)
            if comparison.entry != STARTING_POSITION
            else '',
            basis=comparison.entry_basis,
            reach=float(reach.get(comparison.entry, 0.0)),
            games=sum(counts(comparison.evidence[comparison.entry]))
            if comparison.entry in comparison.evidence
            else None,
            database_score=score(counts(comparison.evidence[comparison.entry]), comparison.color)
            if comparison.entry in comparison.evidence
            else None,
        ),
        starting_baseline=analysis.saved['starting_position_reference'].get('owner_score'),
        scenarios=scenarios,
        decisions=rows,
        groups=[
            dict(decisions=members, improving=plans['improving']['searches'][i], all=plans['all']['searches'][i])
            for i, members in enumerate(linked)
        ],
        interactions=interactions,
        evidence=dict(
            tables=evidence['tables'], fetched=evidence['fetched'], scenarios_evaluated=len(comparison.cache)
        ),
        validation=dict(
            current_score_reproduced=True,
            improving_additive=plans['improving']['additive'],
            all_additive=plans['all']['additive'],
        ),
        manifest=dict(
            report_path=str(analysis.path.resolve()),
            report_sha256=analysis.report_sha256,
            input_path=str(analysis.source),
            input_sha256=manifest['input_sha256'],
            filters=manifest['filters'],
            prior=manifest['prior'],
            sparse_threshold=manifest['sparse_threshold'],
            uncertainty_method=UNCERTAINTY_METHOD,
            evidence=evidence['provenance'],
            selection_rule='best option at every decision point by overall repertoire score; '
            'linked decision points searched together',
        ),
        _adopt=plans['improving']['choice'],
        _comparison=comparison,
    )


def write_outputs(result: JsonObject, data_dir: str | Path, report_dir: str | Path) -> Path:
    """Save the comparison JSON, the adopt PGN and the report page."""
    from .report.comparison import render

    comparison, choice = result.pop('_comparison'), result.pop('_adopt')
    name = result['name']
    adopt = Path(report_dir) / 'comparisons' / f'{name}.adopt.pgn'
    games = comparison.scenarios.candidate_games_for(choice, namespaced=False)
    if games:
        adopt.parent.mkdir(parents=True, exist_ok=True)
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        adopt.write_text('\n\n'.join(g.accept(exporter) for g in games) + '\n', encoding='utf-8', newline='\n')
        # The exported lines must reproduce the improving scenario when placed before your chapters.
        check = parse_games([*read_games(adopt), *comparison.repertoire_games], comparison.scenarios.exclusions)
        scenario = comparison.evaluate(choice)
        transitions = resolve(check, comparison.color, scenario.policy)
        result['validation']['adopt_pgn_reproduces_selection'] = all(
            transitions.get(k) == scenario.transitions[k] for k in scenario.order
        )
        result['outputs'] = dict(adopt_pgn=str(adopt.resolve()))
    else:
        adopt.unlink(missing_ok=True)
        result['outputs'] = dict(adopt_pgn=None)
    data = Path(data_dir) / 'comparisons' / f'{name}.json'
    data.parent.mkdir(parents=True, exist_ok=True)
    page = Path(report_dir) / 'comparisons' / f'{name}.md'
    result['outputs'].update(data=str(data.resolve()), report=str(page.resolve()))
    data.write_text(data_json(result), encoding='utf-8')
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(render(result, page), encoding='utf-8', newline='\n')
    return page


def run(
    inputs: Sequence[str],
    *,
    color: str | None = None,
    score: str | Path | None = None,
    entry: str | None = None,
    name: str | None = None,
    title: str | None = None,
    cache: str | Path = DEFAULT_CACHE,
    offline: bool = False,
    dry_run: bool = False,
    export: bool = True,
    directory: str | Path = CANDIDATES,
) -> JsonObject | None:
    games, name, sources = load_inputs(inputs, name, export and not offline, directory)
    color = color or infer_color(games)
    if color is None:
        raise ValueError('Pass --color white or --color black; the candidate has no Orientation tags')
    path = Path(score) if score else Path('reports/data') / f'{color}.json'
    analysis = AnalysisContext(path)
    if analysis.saved['color'] != color:
        raise ValueError(f'{path} scores {analysis.saved["color"]}, not {color}')
    title = title or next((g.headers['StudyName'] for g in games if g.headers.get('StudyName')), name)
    result = analyze(
        analysis, games, sources, entry=entry, cache=cache, offline=offline, dry_run=dry_run, name=name, title=title
    )
    if result is None:
        return None
    page = write_outputs(result, path.parent, report_directory(path))
    print(f'Comparison report: {page.resolve()}', flush=True)
    return dict(page=page, name=name, color=color)


def load_registry(path: str | Path = REGISTRY) -> list[JsonObject]:
    """The saved comparisons: name, color, sources and optional entry for each."""
    path = Path(path)
    if not path.is_file():
        return []
    entries = json.loads(path.read_text(encoding='utf-8')).get('comparisons', [])
    names = [e['name'] for e in entries]
    if len(set(names)) != len(names) or any(
        not e.get('sources') or e.get('color') not in ('white', 'black') for e in entries
    ):
        raise ValueError(f'{path}: each comparison needs a unique name, a color and its sources')
    return entries


def register(
    path: str | Path,
    name: str,
    color: str,
    inputs: Sequence[str],
    entry: str | None = None,
    directory: str | Path = CANDIDATES,
) -> None:
    """Save a comparison so `repertoire compare` without inputs and `repertoire build` rerun it.

    Files outside the repository are copied into `directory`, so the registry never names a local folder.
    """
    sources: list[str] = []
    for i, value in enumerate(inputs):
        if Path(value).is_file():
            source = Path(value).resolve()
            if not source.is_relative_to(Path.cwd()):
                target = Path(directory) / (f'{name}.pgn' if len(inputs) == 1 else f'{name}-{i + 1}.pgn')
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.read_bytes() != source.read_bytes():
                    target.write_bytes(source.read_bytes())
                source = target.resolve()
            sources.append(source.relative_to(Path.cwd()).as_posix())
        else:
            sources.append(value)
    entries = [e for e in load_registry(path) if e['name'] != name]
    entries.append(dict(name=name, color=color, sources=sources, **({'entry': entry} if entry else {})))
    Path(path).write_text(json.dumps({'comparisons': entries}, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {name} in {path}', flush=True)


def run_registered(path: str | Path = REGISTRY, **options: Any) -> None:
    entries = load_registry(path)
    if not entries:
        raise ValueError(f'No inputs given and no comparisons saved in {path}')
    for e in entries:
        run(e['sources'], color=e['color'], entry=e.get('entry'), name=e['name'], **options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'inputs',
        nargs='*',
        help='Candidate PGN files and Lichess study or chapter URLs, in priority order; '
        f'omit them to rerun every comparison saved in {REGISTRY}',
    )
    parser.add_argument(
        '--color', choices=['white', 'black'], help='Default: the Orientation tags of an exported study'
    )
    parser.add_argument('--entry', help="Report scores after this position: moves such as '1.e4 e5 2.Nc3', or a FEN")
    parser.add_argument('--name', help='Output name (default: from the study name)')
    parser.add_argument('--title', help='Report title (default: the study name)')
    parser.add_argument('--save', action='store_true', help=f'Save this comparison in {REGISTRY} for later builds')
    parser.add_argument('--registry', default=REGISTRY, help='The saved comparisons file')
    parser.add_argument('--score', help='Saved score of your repertoire (default: reports/data/<color>.json)')
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument('--offline', action='store_true', help='Use only cached tables and saved study exports')
    parser.add_argument('--no-export', action='store_true', help='Reuse the last export of study URLs')
    parser.add_argument('--dry-run', action='store_true', help='Count the tables to fetch and stop')
    add_token_option(parser)
    args = parser.parse_args()
    apply_token_file(parser, args)
    options = dict(cache=args.cache, offline=args.offline, dry_run=args.dry_run, export=not args.no_export)
    try:
        if not args.inputs:
            if args.save or args.name or args.entry or args.color:
                parser.error('--save, --name, --entry and --color need inputs')
            run_registered(args.registry, score=args.score, **options)
            if not args.dry_run:
                update_report_outputs(Path(args.score) if args.score else Path('reports/data/white.json'))
            return
        done = run(
            args.inputs,
            color=args.color,
            score=args.score,
            entry=args.entry,
            name=args.name,
            title=args.title,
            **options,
        )
        if args.save and done is not None:
            register(args.registry, done['name'], done['color'], args.inputs, args.entry)
        if done is not None:
            # The summary lists comparisons, so refresh it as the other standalone commands do.
            update_report_outputs(Path(args.score) if args.score else Path('reports/data') / f"{done['color']}.json")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f'Comparison failed: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())  # type: ignore[func-returns-value]
