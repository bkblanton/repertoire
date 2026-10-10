"""Engine evaluations where preparation ends, of your moves and of the opponent's replies.

Evaluations come from the local store that `repertoire evals import` fills from the Lichess evaluation export. Each
position uses its deepest evaluation's first line, converted to your side and then to an expected score with the
Lichess win-chance curve, the same curve the reports invert for centipawn equivalents. That puts the engine's view
on the same 0-100% scale as database scores. Nothing here changes a score, a ranking of the repertoire or a move.

Positions without an evaluation stay missing. Where preparation ends after a reply that has none, the position
before the reply bounds it from below: the engine evaluates that position with the opponent's best reply, and no
reply can be better for the opponent. These floors are kept apart from evaluated positions throughout.
"""

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from . import evals as store_module
from .board_cache import children
from .context import DEFAULT_CACHE, AnalysisContext, stage_main
from .evals import Evaluation, Store
from .schema import JsonObject, Position
from .status import Status

# The Lichess win-chance curve: expected score = 1 / (1 + exp(-k * centipawns)). report.format inverts it.
LICHESS_WIN_CHANCE_COEFFICIENT = 0.00368208
# Lichess judges a move by how much it lowers the mover's winning chances (-1 to 1): 0.1 is an inaccuracy, 0.2 a
# mistake and 0.3 a blunder. On the expected-score scale used here those are 5, 10 and 15 points.
JUDGEMENTS = ((15.0, 'blunder'), (10.0, 'mistake'), (5.0, 'inaccuracy'))


def expected_score(evaluation: Evaluation | None, color: bool) -> float | None:
    """Your expected score under the engine's evaluation of a position, or None without one."""
    if evaluation is None or not evaluation['pvs']:
        return None
    line = evaluation['pvs'][0]
    if line.get('mate') is not None:
        if line['mate'] == 0:
            return None
        white = 1.0 if line['mate'] > 0 else 0.0
    elif line.get('cp') is not None:
        white = 1 / (1 + math.exp(-LICHESS_WIN_CHANCE_COEFFICIENT * line['cp']))
    else:
        return None
    return white if color else 1 - white


def display(evaluation: Evaluation | None, color: bool) -> JsonObject | None:
    """The first line's evaluation from your side, as published: centipawns or moves to mate, with its depth."""
    if evaluation is None or not evaluation['pvs']:
        return None
    line, sign = evaluation['pvs'][0], 1 if color else -1
    return dict(
        cp=None if line.get('cp') is None else sign * line['cp'],
        mate=None if line.get('mate') is None else sign * line['mate'],
        depth=evaluation['depth'],
        expected_score=expected_score(evaluation, color),
    )


def judgement(loss_pp: float | None) -> str | None:
    """Lichess's name for a move that lowers the mover's expected score by `loss_pp` points."""
    if loss_pp is None:
        return None
    return next((name for threshold, name in JUDGEMENTS if loss_pp >= threshold), None)


def exits_summary(stops: Iterable[JsonObject], engine: Mapping[Position, float]) -> JsonObject:
    """The database and engine scores where preparation ends, over the exits with an evaluation, plus how much
    of the exit probability has one, a floor from the position before the reply, or neither."""
    evaluated = floored = missing = 0.0
    database = known = floor = 0.0
    for stop in stops:
        if stop.get('score') is None or not stop.get('reach'):
            continue
        reach = stop['reach']
        if stop['position'] in engine:
            evaluated += reach
            database += reach * stop['score']
            known += reach * engine[stop['position']]
        elif stop.get('parent_position') in engine:
            floored += reach
            floor += reach * engine[stop['parent_position']]
        else:
            missing += reach
    total = evaluated + floored + missing
    return dict(
        status=Status.RESOLVED if evaluated else Status.UNRESOLVED_EVIDENCE,
        reach=total,
        evaluated_reach=evaluated,
        floored_reach=floored,
        missing_reach=missing,
        database_score=database / evaluated if evaluated else None,
        engine_score=known / evaluated if evaluated else None,
        engine_floor=(known + floor) / (evaluated + floored) if evaluated + floored else None,
        practical_minus_engine_pp=100 * (database - known) / evaluated if evaluated else None,
    )


def wanted(vulnerabilities: JsonObject, preparation: JsonObject) -> set[Position]:
    """Every position the engine view looks up."""
    result: set[Position] = set()
    for scope in preparation['scopes']:
        for stop in scope.get('stops', []):
            result.update(k for k in (stop.get('position'), stop.get('parent_position')) if k)
    overall = vulnerabilities['overall']
    for row in overall['all_signed_rows']:
        result.add(row['position'])
        result.add(row['target'] or children(row['position'])[row['move']])
    for row in overall.get('free_transpositions', []):
        result.update((row['exit_position'], row['transposition_target']))
    return result


def analyze(path: str | Path, cache: str | Path = DEFAULT_CACHE, store: str | Path | None = None) -> JsonObject:
    analysis = AnalysisContext(path, companions=('vulnerabilities', 'preparation'))
    color, vulnerabilities, preparation = (
        analysis.color,
        analysis.companions['vulnerabilities'],
        analysis.companions['preparation'],
    )
    result: JsonObject = dict(color=analysis.saved['color'])
    # Looked up at call time, so tests can point every stage at their own store.
    store = Path(store_module.STORE if store is None else store)
    evaluations: dict[Position, Evaluation] = {}
    if store.exists():
        with Store(store, read_only=True) as evals:
            evaluations = evals.read(sorted(wanted(vulnerabilities, preparation)))
    engine = {k: e for k, ev in evaluations.items() if (e := expected_score(ev, color)) is not None}
    result['status'] = Status.RESOLVED if engine else Status.UNRESOLVED_EVIDENCE
    result['scopes'] = [
        dict(id=scope['id'], exits=exits_summary(scope.get('stops', []), engine)) for scope in preparation['scopes']
    ]
    overall_stops = next(s for s in preparation['scopes'] if s['id'] == 'overall').get('stops', [])
    exits: list[JsonObject] = []
    for stop in overall_stops:
        basis = (
            'evaluated' if stop['position'] in engine else 'floor' if stop.get('parent_position') in engine else None
        )
        if basis is None or stop.get('score') is None:
            continue
        exits.append(
            dict(
                position=stop['position'],
                parent_position=stop.get('parent_position'),
                move=stop.get('move'),
                basis=basis,
                engine=display(
                    evaluations[stop['position'] if basis == 'evaluated' else stop['parent_position']], color
                ),
                engine_score=engine[stop['position'] if basis == 'evaluated' else stop['parent_position']],
            )
        )
    moves: dict[str, JsonObject] = {}
    for row in vulnerabilities['overall']['all_signed_rows']:
        after_position = row['target'] or children(row['position'])[row['move']]
        before, after = engine.get(row['position']), engine.get(after_position)
        change = None if before is None or after is None else 100 * (after - before)
        entry: dict[str, Any] = dict(
            before=before, after=after, engine_after=display(evaluations.get(after_position), color)
        )
        if row['kind'] == 'own':
            loss = None if change is None else -change
            entry.update(loss_pp=loss, judgement=judgement(loss))
        else:
            # The opponent's loss is your gain.
            entry.update(swing_pp=change, judgement=judgement(change))
        moves[row['id']] = entry
    transpositions: dict[str, JsonObject] = {}
    for row in vulnerabilities['overall'].get('free_transpositions', []):
        before, after = engine.get(row['exit_position']), engine.get(row['transposition_target'])
        loss = None if before is None or after is None else 100 * (before - after)
        transpositions[row['id']] = dict(before=before, after=after, loss_pp=loss, judgement=judgement(loss))
    analysis.require_source('engine analysis')
    result.update(
        exits=exits,
        moves=moves,
        transpositions=transpositions,
        manifest=analysis.companion_manifest(
            supporting_sha256=analysis.companion_hashes,
            evaluations=dict(
                store=str(store.resolve()),
                positions=len(evaluations),
                conversion='expected score = 1 / (1 + exp(-0.00368208 * centipawns)) from your side; mate is 1 or 0',
                selection='deepest evaluation, first line',
            ),
        ),
    )
    return result


def main() -> None:
    stage_main('engine', analyze, __doc__)


if __name__ == '__main__':
    main()
