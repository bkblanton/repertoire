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

import chess

from . import evals as store_module
from .attribution import Attribution
from .board_cache import children, move_text, san, turn
from .context import DEFAULT_CACHE, AnalysisContext, stage_main
from .evals import Evaluation, Store
from .graph import Graph, Transitions, resolve
from .model import Evidence
from .ratings import reply_rating, response_rating
from .schema import JsonObject, Position
from .status import Status

# The Lichess win-chance curve: expected score = 1 / (1 + exp(-k * centipawns)). report.format inverts it.
LICHESS_WIN_CHANCE_COEFFICIENT = 0.00368208
# Lichess judges a move by how much it lowers the mover's winning chances (-1 to 1): 0.1 is an inaccuracy, 0.2 a
# mistake and 0.3 a blunder. On the expected-score scale used here those are 5, 10 and 15 points.
JUDGEMENTS = ((15.0, 'blunder'), (10.0, 'mistake'), (5.0, 'inaccuracy'))
# Lichess cloud PV lines encode castling in Chess960 UCI (king takes rook).
CASTLING_UCI = {'e1h1': 'e1g1', 'e1a1': 'e1c1', 'e8h8': 'e8g8', 'e8a8': 'e8c8'}
CASTLING_FLAG = {'e1h1': 'K', 'e1a1': 'Q', 'e8h8': 'k', 'e8a8': 'q'}
DEFAULT_AUDIT_GOAL = 2000
MIN_SHARPNESS_GAP_PP = 5.0
MIN_TRAP_GAP_PP = 8.0
MIN_TEST_THREAT_PP = 3.0
MIN_OWN_LOSS_PP = 5.0


def normalize_pv_uci(position: Position, uci: str) -> str:
    """Convert Lichess Chess960 castling UCI to standard UCI when `position` retains that castling right."""
    if uci in CASTLING_UCI and CASTLING_FLAG[uci] in position.split(' ')[2]:
        return CASTLING_UCI[uci]
    return uci


def pv_expected_score(pv: Mapping[str, Any] | None, color: bool) -> float | None:
    """Your expected score under a single PV entry, or None without a valid score."""
    if not pv:
        return None
    if pv.get('mate') is not None:
        if pv['mate'] == 0:
            return None
        white = 1.0 if pv['mate'] > 0 else 0.0
    elif pv.get('cp') is not None:
        white = 1 / (1 + math.exp(-LICHESS_WIN_CHANCE_COEFFICIENT * pv['cp']))
    else:
        return None
    return white if color else 1 - white


def expected_score(evaluation: Evaluation | None, color: bool) -> float | None:
    """Your expected score under the engine's evaluation of a position, or None without one."""
    if evaluation is None or not evaluation['pvs']:
        return None
    return pv_expected_score(evaluation['pvs'][0], color)


def pv_first_move(position: Position, evaluation: Evaluation | None) -> str | None:
    """The normalized legal first UCI move of an evaluation's top line, or None if unavailable."""
    if evaluation is None or not evaluation['pvs']:
        return None
    line = evaluation['pvs'][0].get('line', '')
    if not line:
        return None
    move = normalize_pv_uci(position, line.split()[0])
    return move if move in children(position) else None


def position_sharpness(
    position: Position,
    color: bool,
    evaluations: Mapping[Position, Evaluation],
    *,
    allow_children: bool = True,
) -> JsonObject | None:
    """Gap in expected score between the top two moves for `color` at `position`.

    Prefers multi-PV from `evaluations[position]` when at least two distinct legal lines exist; otherwise, when
    `allow_children` is true, compares 1-ply child evaluations in `evaluations`.
    """
    if turn(position) != color:
        return None
    legal = children(position)
    ev = evaluations.get(position)
    if ev and len(ev['pvs']) >= 2:
        by_move: dict[str, float] = {}
        for pv in ev['pvs']:
            line = pv.get('line', '')
            if not line:
                continue
            move = normalize_pv_uci(position, line.split()[0])
            sc = pv_expected_score(pv, color)
            if move in legal and sc is not None:
                prev = by_move.get(move)
                if prev is None or sc > prev:
                    by_move[move] = sc
        if len(by_move) >= 2:
            ranked = sorted(by_move.items(), key=lambda item: (-item[1], item[0]))
            (m0, s0), (m1, s1) = ranked[0], ranked[1]
            return dict(
                best_move=m0,
                best_score=s0,
                second_move=m1,
                second_score=s1,
                gap_pp=100 * (s0 - s1),
                source='multipv',
            )
    if not allow_children:
        return None
    scored: list[tuple[float, str]] = []
    for uci, child_pos in legal.items():
        cev = evaluations.get(child_pos)
        if cev is not None and (sc := expected_score(cev, color)) is not None:
            scored.append((sc, uci))
    if ev and ev['pvs'] and ev['pvs'][0].get('line'):
        pv_move = normalize_pv_uci(position, ev['pvs'][0]['line'].split()[0])
        pv_score = pv_expected_score(ev['pvs'][0], color)
        if pv_move in legal and pv_score is not None and all(u != pv_move for _, u in scored):
            scored.append((pv_score, pv_move))
    if len(scored) < 2:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    s0, m0 = scored[0]
    s1, m1 = scored[1]
    return dict(
        best_move=m0,
        best_score=s0,
        second_move=m1,
        second_score=s1,
        gap_pp=100 * (s0 - s1),
        source='children',
    )


def is_trivial_recapture(
    parent_pos: Position, opponent_move_uci: str, best_move_uci: str, second_move_uci: str | None
) -> bool:
    """Whether `best_move_uci` is an obvious single-piece recapture on the square the opponent just captured on."""
    board = chess.Board(parent_pos + ' 0 1')
    try:
        opp = chess.Move.from_uci(opponent_move_uci)
        best = chess.Move.from_uci(best_move_uci)
    except ValueError:
        return False
    if not board.is_capture(opp) or best.to_square != opp.to_square:
        return False
    if second_move_uci:
        try:
            if chess.Move.from_uci(second_move_uci).to_square == opp.to_square:
                return False
        except ValueError:
            pass
    return True


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


def move_rows(vulnerabilities: JsonObject) -> list[JsonObject]:
    """Every move row once, from the overall repertoire and every chapter; a chapter whose moves lost to an
    alternative has rows of its own."""
    seen: dict[str, JsonObject] = {}
    for scope in [vulnerabilities['overall'], *vulnerabilities.get('chapters', [])]:
        for row in scope.get('all_signed_rows', []):
            seen.setdefault(row['id'], row)
    return list(seen.values())


def after_move(row: JsonObject) -> Position:
    """The position after a vulnerability row's move."""
    return row['target'] or children(row['position'])[row['move']]


def wanted(vulnerabilities: JsonObject, preparation: JsonObject) -> set[Position]:
    """Every position the engine view looks up."""
    result: set[Position] = set()
    for scope in preparation['scopes']:
        for stop in scope.get('stops', []):
            result.update(k for k in (stop.get('position'), stop.get('parent_position')) if k)
    for row in move_rows(vulnerabilities):
        result.add(row['position'])
        result.add(after_move(row))
    for row in vulnerabilities['overall'].get('free_transpositions', []):
        result.update((row['exit_position'], row['transposition_target']))
    return result


def _move_rating_context(evidence: Evidence, parent: Position, move: str) -> JsonObject:
    """Opponent rating and difference vs parent for an opponent reply, matching `ratings.py` formatting."""
    table = evidence.get(parent)
    reply = reply_rating(table, move)
    parent_ctx = response_rating(table)
    result = dict(reply)
    if reply['mean'] is not None and parent_ctx['mean'] is not None:
        result['difference_vs_parent'] = reply['mean'] - parent_ctx['mean']
        result['parent_known_coverage'] = parent_ctx['known_coverage']
        result['comparison_coverage'] = reply['known_coverage']
    return result


def _needs_child_evals(position: Position, evaluations: Mapping[Position, Evaluation]) -> bool:
    ev = evaluations.get(position)
    if ev is None or len(ev['pvs']) < 2:
        return True
    p0, p1 = ev['pvs'][0], ev['pvs'][1]
    m0 = pv_first_move(
        position,
        {
            'depth': ev['depth'],
            'knodes': ev['knodes'],
            'pvs': [p0],
            'source': ev['source'],
            'retrieved_at': ev['retrieved_at'],
        },
    )
    m1 = pv_first_move(
        position,
        {
            'depth': ev['depth'],
            'knodes': ev['knodes'],
            'pvs': [p1],
            'source': ev['source'],
            'retrieved_at': ev['retrieved_at'],
        },
    )
    return not (m0 and m1 and m0 != m1)


def _route_last_move(line_text: str) -> tuple[Position, str] | None:
    """The `(parent_position, move_uci)` of the last move on a SAN line from the starting position."""
    import re

    tokens = [re.sub(r'^\d+\.(\.\.)?', '', t) for t in line_text.split()]
    if not tokens or line_text.strip() == '(PGN root)':
        return None
    board = chess.Board()
    try:
        for token in tokens[:-1]:
            board.push_san(token)
        parent = ' '.join(board.fen(en_passant='legal').split()[:4])
        move = board.parse_san(tokens[-1])
    except ValueError:
        return None
    return parent, move.uci()


def _scope_transitions(
    graph: Graph,
    color: bool,
    base_policy: Mapping[Position, str | Mapping[str, float]],
    vulnerabilities: JsonObject,
) -> dict[str, Transitions]:
    """Resolved repertoire transitions for overall and every chapter scope, sharing identical overrides."""
    import json

    overall = resolve(graph, color, base_policy)
    cache: dict[str, Transitions] = {'{}': overall}
    result: dict[str, Transitions] = {'overall': overall}
    for ch in vulnerabilities.get('chapters', []):
        overrides = ch.get('policy_overrides', {})
        key = json.dumps(overrides, sort_keys=True)
        if key not in cache:
            cache[key] = resolve(graph, color, dict(base_policy, **overrides))
        result[ch['id']] = cache[key]
    return result


def _scope_critical_audit(
    scope_id: str,
    vuln_scope: JsonObject,
    prep_scope: JsonObject,
    transitions: Transitions,
    evaluations: Mapping[Position, Evaluation],
    engine: Mapping[Position, float],
    moves_info: Mapping[str, JsonObject],
    evidence: Evidence,
    attribution: Attribution,
    color: bool,
) -> JsonObject:
    """Find unprepared opponent engine tests, sharp unprepared replies, and chapter endpoints before a cliff."""
    rows = vuln_scope.get('all_signed_rows', [])
    opp_by_parent: dict[Position, list[JsonObject]] = {}
    inaccurate_targets: set[Position] = set()
    for row in rows:
        if row['kind'] == 'opponent':
            opp_by_parent.setdefault(row['position'], []).append(row)
        elif row['kind'] == 'own' and row.get('target'):
            loss = (moves_info.get(row['id']) or {}).get('loss_pp')
            if loss is not None and loss >= MIN_OWN_LOSS_PP:
                inaccurate_targets.add(row['target'])

    audit_floor = 1 / DEFAULT_AUDIT_GOAL
    tests: list[JsonObject] = []
    for p, opp_rows in sorted(opp_by_parent.items()):
        if p not in engine or p not in evaluations:
            continue
        pv0 = pv_first_move(p, evaluations[p])
        if not pv0:
            continue
        prepared_at_p = transitions.get(p, {})
        if pv0 in prepared_at_p:
            continue
        row_by_move = {r['move']: r for r in opp_rows}
        matched = row_by_move.get(pv0)
        parent_reach = max(r['parent_reach'] for r in opp_rows)
        if parent_reach <= 0:
            continue
        rep_score = next((r['reference_score'] for r in opp_rows if r.get('reference_score') is not None), None)
        at_chapter_end = not any(r['prepared'] for r in opp_rows)
        target = children(p).get(pv0)
        if matched:
            move_reach = matched['branch_reach']
            db_score = matched['move_score']
            sample_count = matched['sample_count']
            sparse = matched['sparse']
            line_text = matched['line']
            attr = matched.get('chapter_attribution') or attribution.position_or_move(p, pv0)
        else:
            move_reach = 0.0
            db_score = None
            sample_count = 0
            sparse = True
            sample_row = opp_rows[0]
            parent_route = sample_row['line'].rsplit(' ', 1)[0] if ' ' in sample_row['line'] else ''
            number = len(parent_route.split()) // 2 + 1 if parent_route else 1
            token = move_text(p, number, pv0)
            line_text = f'{parent_route} {token}'.strip()
            attr = attribution.position_or_move(p, pv0)
        eng_score = engine[p]
        is_refutation = p in inaccurate_targets
        reply_sh = (
            position_sharpness(target, color, evaluations, allow_children=move_reach >= audit_floor)
            if target and target in evaluations
            else None
        )
        if reply_sh and is_trivial_recapture(p, pv0, reply_sh['best_move'], reply_sh['second_move']):
            reply_sh = None
        if rep_score is None:
            threat_pp = None
        else:
            ref_opp = min(eng_score, db_score) if (db_score is not None and not sparse) else eng_score
            threat_pp = 100 * (rep_score - ref_opp)
        reply_gap = reply_sh['gap_pp'] if reply_sh else None
        if (
            not is_refutation
            and (threat_pp is None or threat_pp < MIN_TEST_THREAT_PP)
            and (reply_gap is None or reply_gap < MIN_SHARPNESS_GAP_PP)
        ):
            continue
        assert target is not None
        reply_summary = (
            dict(
                best_move=reply_sh['best_move'],
                best_move_san=san(target, reply_sh['best_move']),
                second_move=reply_sh['second_move'],
                second_move_san=san(target, reply_sh['second_move']),
                gap_pp=reply_sh['gap_pp'],
                source=reply_sh['source'],
            )
            if reply_sh
            else None
        )
        threat_term = max(0.0, threat_pp or 0.0)
        gap_term = reply_gap if (reply_gap is not None and reply_gap >= MIN_SHARPNESS_GAP_PP) else 0.0
        base_score = threat_term + gap_term
        if is_refutation and base_score < MIN_OWN_LOSS_PP:
            base_score = MIN_OWN_LOSS_PP
        tests.append(
            dict(
                position=p,
                move=pv0,
                move_san=san(p, pv0),
                target=target,
                line=line_text,
                parent_reach=parent_reach,
                move_reach=move_reach,
                sample_count=sample_count,
                sparse=sparse,
                repertoire_score=rep_score,
                engine_score=eng_score,
                database_score=db_score,
                threat_pp=threat_pp,
                opponent_rating=_move_rating_context(evidence, p, pv0),
                is_refutation=is_refutation,
                at_chapter_end=at_chapter_end,
                reply_sharpness=reply_summary,
                chapter_attribution=attr,
                priority=parent_reach * base_score,
            )
        )
    tests.sort(key=lambda item: (-item['priority'], item['position'], item['move']))

    sharp_replies: list[JsonObject] = []
    for row in rows:
        if (
            row['kind'] != 'opponent'
            or row['prepared']
            or row['sparse']
            or row.get('move_score') is None
            or row['branch_reach'] <= 0
        ):
            continue
        target = after_move(row)
        if target not in evaluations:
            continue
        sh = position_sharpness(target, color, evaluations, allow_children=row['branch_reach'] >= audit_floor)
        if sh and is_trivial_recapture(row['position'], row['move'], sh['best_move'], sh['second_move']):
            sh = None
        target_score = engine.get(target)
        emd = None if target_score is None else 100 * (target_score - row['move_score'])
        gap = sh['gap_pp'] if sh else None
        if (gap is None or gap < MIN_SHARPNESS_GAP_PP) and (emd is None or emd < MIN_TRAP_GAP_PP):
            continue
        swing_pp = (moves_info.get(row['id']) or {}).get('swing_pp')
        sharpness_summary = (
            dict(
                best_move=sh['best_move'],
                best_move_san=san(target, sh['best_move']),
                second_move=sh['second_move'],
                second_move_san=san(target, sh['second_move']),
                gap_pp=sh['gap_pp'],
                source=sh['source'],
            )
            if sh
            else None
        )
        priority = row['branch_reach'] * max(gap or 0.0, max(0.0, emd or 0.0))
        sharp_replies.append(
            dict(
                position=row['position'],
                move=row['move'],
                move_san=row['move_san'],
                target=target,
                line=row['line'],
                move_reach=row['branch_reach'],
                database_score=row['move_score'],
                engine_score=target_score,
                engine_minus_db_pp=emd,
                opponent_swing_pp=swing_pp,
                at_chapter_end=not any(r['prepared'] for r in opp_by_parent.get(row['position'], [])),
                sharpness=sharpness_summary,
                chapter_attribution=row.get('chapter_attribution'),
                priority=priority,
            )
        )
    sharp_replies.sort(key=lambda item: (-item['priority'], item['position'], item['move']))

    cliffs: list[JsonObject] = []
    for stop in prep_scope.get('stops', []):
        if stop.get('type') != 'theory_leaf' or stop.get('sparse') or not stop.get('reach'):
            continue
        pos = stop['position']
        if pos not in evaluations:
            continue
        sh = position_sharpness(pos, color, evaluations, allow_children=True)
        if not sh or sh['gap_pp'] < MIN_SHARPNESS_GAP_PP:
            continue
        last = _route_last_move(stop.get('line', ''))
        if last and is_trivial_recapture(last[0], last[1], sh['best_move'], sh['second_move']):
            continue
        cliffs.append(
            dict(
                position=pos,
                line=stop['line'],
                reach=stop['reach'],
                database_score=stop.get('score'),
                engine_score=engine.get(pos),
                best_move=sh['best_move'],
                best_move_san=san(pos, sh['best_move']),
                second_move=sh['second_move'],
                second_move_san=san(pos, sh['second_move']),
                gap_pp=sh['gap_pp'],
                sharpness_source=sh['source'],
                chapter_attribution=stop.get('chapter_attribution'),
                priority=stop['reach'] * sh['gap_pp'],
            )
        )
    cliffs.sort(key=lambda item: (-item['priority'], item['position']))

    return dict(
        unprepared_engine_tests=tests,
        sharp_unprepared_replies=sharp_replies,
        endpoint_cliffs=cliffs,
    )


def analyze(path: str | Path, cache: str | Path = DEFAULT_CACHE, store: str | Path | None = None) -> JsonObject:
    analysis = AnalysisContext(path, companions=('vulnerabilities', 'preparation'))
    color, vulnerabilities, preparation = (
        analysis.color,
        analysis.companions['vulnerabilities'],
        analysis.companions['preparation'],
    )
    evidence = analysis.read_evidence(cache, list(analysis.manifest['evidence']), required=True)
    result: JsonObject = dict(color=analysis.saved['color'])
    # Looked up at call time, so tests can point every stage at their own store.
    store = Path(store_module.STORE if store is None else store)
    base_wanted = wanted(vulnerabilities, preparation)
    evaluations: dict[Position, Evaluation] = {}
    lookahead_evals: dict[Position, Evaluation] = {}
    if store.exists():
        with Store(store, read_only=True) as evals:
            evaluations = evals.read(sorted(base_wanted))
            lookahead_evals = dict(evaluations)
            # Load 1-ply lookahead evaluations for common exits and endpoints when multi-PV is absent.
            extra_wanted: set[Position] = set()
            audit_floor = 1 / DEFAULT_AUDIT_GOAL
            overall_rows = vulnerabilities['overall'].get('all_signed_rows', [])
            for row in overall_rows:
                if (
                    row['kind'] == 'opponent'
                    and not row['prepared']
                    and not row['sparse']
                    and row['branch_reach'] >= audit_floor
                ):
                    target = after_move(row)
                    if target in evaluations and _needs_child_evals(target, evaluations):
                        extra_wanted.update(children(target).values())
            for stop in preparation['scopes'][0].get('stops', []):
                if stop.get('type') == 'theory_leaf' and not stop.get('sparse') and stop.get('reach'):
                    pos = stop['position']
                    if pos in evaluations and _needs_child_evals(pos, evaluations):
                        extra_wanted.update(children(pos).values())
            for row in overall_rows:
                if row['kind'] == 'opponent' and row['position'] in evaluations and row['parent_reach'] >= audit_floor:
                    pv0 = pv_first_move(row['position'], evaluations[row['position']])
                    if pv0:
                        child_target = children(row['position']).get(pv0)
                        if child_target and child_target not in evaluations:
                            extra_wanted.add(child_target)
            if extra_wanted - set(evaluations):
                lookahead_evals.update(evals.read(sorted(extra_wanted - set(evaluations))))
                extra_children: set[Position] = set()
                for row in overall_rows:
                    if (
                        row['kind'] == 'opponent'
                        and not row['prepared']
                        and row['position'] in evaluations
                        and row['parent_reach'] >= audit_floor
                    ):
                        pv0 = pv_first_move(row['position'], evaluations[row['position']])
                        if pv0:
                            child_target = children(row['position']).get(pv0)
                            if (
                                child_target
                                and child_target in lookahead_evals
                                and _needs_child_evals(child_target, lookahead_evals)
                            ):
                                extra_children.update(children(child_target).values())
                if extra_children - set(lookahead_evals):
                    lookahead_evals.update(evals.read(sorted(extra_children - set(lookahead_evals))))

    engine = {k: e for k, ev in evaluations.items() if (e := expected_score(ev, color)) is not None}
    all_engine = {k: e for k, ev in lookahead_evals.items() if (e := expected_score(ev, color)) is not None}
    result['status'] = Status.RESOLVED if engine else Status.UNRESOLVED_EVIDENCE
    # Each evaluated position once, so every page can label the places its preparation ends.
    positions = {k: display(evaluations[k], color) for k in sorted(engine)}
    scope_trans = _scope_transitions(analysis.graph, color, analysis.policy, vulnerabilities)
    row_scope_trans: dict[str, list[Transitions]] = {}
    for scope_id, scope_vuln in [
        ('overall', vulnerabilities['overall']),
        *[(c['id'], c) for c in vulnerabilities.get('chapters', [])],
    ]:
        t = scope_trans[scope_id]
        for row in scope_vuln.get('all_signed_rows', []):
            if row['kind'] == 'own':
                row_scope_trans.setdefault(row['id'], []).append(t)

    moves: dict[str, JsonObject] = {}
    for row in move_rows(vulnerabilities):
        after_position = after_move(row)
        before, after = engine.get(row['position']), engine.get(after_position)
        change = None if before is None or after is None else 100 * (after - before)
        entry: dict[str, Any] = dict(
            before=before, after=after, engine_after=display(evaluations.get(after_position), color)
        )
        if row['kind'] == 'own':
            loss = None if change is None else -change
            judged_as = judgement(loss)
            refutation: JsonObject | None = None
            if judged_as and after_position in evaluations:
                refut = pv_first_move(after_position, evaluations[after_position])
                if refut:
                    prepared_refut = any(
                        refut in t.get(after_position, {})
                        for t in row_scope_trans.get(row['id'], [scope_trans['overall']])
                    )
                    refutation = dict(move=refut, move_san=san(after_position, refut), prepared=prepared_refut)
            entry.update(loss_pp=loss, judgement=judged_as, refutation=refutation)
        else:
            # The opponent's loss is your gain.
            entry.update(swing_pp=change, judgement=judgement(change))
        moves[row['id']] = entry

    vuln_scopes = {'overall': vulnerabilities['overall'], **{c['id']: c for c in vulnerabilities.get('chapters', [])}}
    attribution = Attribution(analysis.graph)
    result['scopes'] = [
        dict(
            id=scope['id'],
            exits=exits_summary(scope.get('stops', []), engine),
            critical=_scope_critical_audit(
                scope['id'],
                vuln_scopes.get(scope['id'], {}),
                scope,
                scope_trans.get(scope['id'], scope_trans['overall']),
                lookahead_evals,
                all_engine,
                moves,
                evidence,
                attribution,
                color,
            ),
        )
        for scope in preparation['scopes']
    ]

    transpositions: dict[str, JsonObject] = {}
    for row in vulnerabilities['overall'].get('free_transpositions', []):
        before, after = engine.get(row['exit_position']), engine.get(row['transposition_target'])
        loss = None if before is None or after is None else 100 * (before - after)
        transpositions[row['id']] = dict(before=before, after=after, loss_pp=loss, judgement=judgement(loss))
    analysis.require_source('engine analysis')
    result.update(
        positions=positions,
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
