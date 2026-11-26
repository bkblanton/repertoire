"""Render the build change report (`changes.md`) from saved `changes.json`."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from ..board_cache import STARTING_POSITION
from ..changes import EPSILON, summary_text
from ..schema import JsonObject, Position
from .format import (
    bounded,
    centipawn_equivalent,
    display,
    escape,
    line,
    number,
    pawns,
    percentage,
    score_points,
    span,
    weighted,
)
from .links import analysis_url, linked_line
from .markdown import Cell, section, table

LINE_LIMIT = 25


def _same_float(a: float | None, b: float | None, tol: float = EPSILON) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def _same_bounds(a: Sequence[float] | None, b: Sequence[float] | None, tol: float = EPSILON) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return len(a) == len(b) and all(_same_float(x, y, tol) for x, y in zip(a, b))


def _timestamp(value: str | None) -> str:
    if not value:
        return 'unknown time'
    try:
        return datetime.fromisoformat(value).astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')
    except ValueError:
        return value


def _plural(count: int, noun: str) -> str:
    return f'{count:,} {noun}' + ('' if count == 1 else 's')


def _score_text(score: float | None, bounds: Sequence[float] | None) -> str:
    return bounded(score, bounds)


def _delta_text(score: float | None, baseline: float | None, bounds: Sequence[float] | None) -> str:
    if score is not None and baseline is not None:
        return score_points(100 * (score - baseline), signed=True)
    if score is None and bounds and baseline is not None:
        return bounded(None, [100 * (b - baseline) for b in bounds], lambda v: score_points(v, signed=True))
    return 'unavailable'


def _bounded_delta_pp(
    old_value: float | None,
    old_bounds: Sequence[float] | None,
    new_value: float | None,
    new_bounds: Sequence[float] | None,
) -> str | None:
    if old_value is not None and new_value is not None:
        return weighted(100 * (new_value - old_value), signed=True)
    old_pair = [old_value, old_value] if old_value is not None else list(old_bounds or ())
    new_pair = [new_value, new_value] if new_value is not None else list(new_bounds or ())
    if len(old_pair) == 2 and len(new_pair) == 2:
        low = 100 * (new_pair[0] - old_pair[1])
        high = 100 * (new_pair[1] - old_pair[0])
        return span([low, high], lambda v: weighted(v, signed=True))
    return None


def _score_change(
    old_score: float | None,
    old_bounds: Sequence[float] | None,
    new_score: float | None,
    new_bounds: Sequence[float] | None,
) -> str:
    before, after = _score_text(old_score, old_bounds), _score_text(new_score, new_bounds)
    if _same_float(old_score, new_score) and _same_bounds(old_bounds, new_bounds):
        return after
    delta = _bounded_delta_pp(old_score, old_bounds, new_score, new_bounds)
    return f'{before} → {after} ({delta})' if delta else f'{before} → {after}'


def _delta_change(
    old_score: float | None,
    old_base: float | None,
    old_bounds: Sequence[float] | None,
    new_score: float | None,
    new_base: float | None,
    new_bounds: Sequence[float] | None,
) -> str:
    before = _delta_text(old_score, old_base, old_bounds)
    after = _delta_text(new_score, new_base, new_bounds)
    old_diff = None if old_score is None or old_base is None else old_score - old_base
    new_diff = None if new_score is None or new_base is None else new_score - new_base
    if _same_float(old_diff, new_diff) and before == after:
        return after
    return f'{before} → {after}'


def _int_change(old_value: int | None, new_value: int | None) -> str:
    if old_value is None and new_value is None:
        return 'unavailable'
    if old_value == new_value:
        return f'{new_value:,}' if new_value is not None else 'unavailable'
    if old_value is not None and new_value is not None:
        return f'{old_value:,} → {new_value:,} ({new_value - old_value:+,})'
    before = f'{old_value:,}' if old_value is not None else 'unavailable'
    after = f'{new_value:,}' if new_value is not None else 'unavailable'
    return f'{before} → {after}'


def _depth_change(old_value: float | None, new_value: float | None) -> str:
    if _same_float(old_value, new_value):
        return number(new_value, 1)
    if old_value is not None and new_value is not None:
        diff = new_value - old_value
        digits = 1 if abs(diff) >= 0.1 else 2
        return f'{number(old_value, 1)} → {number(new_value, 1)} ({number(diff, digits, signed=True)})'
    return f'{number(old_value, 1)} → {number(new_value, 1)}'


def _reach_change(old_value: float | None, new_value: float | None) -> str:
    if _same_float(old_value, new_value):
        return percentage(new_value)
    if old_value is not None and new_value is not None:
        delta = weighted(100 * (new_value - old_value), signed=True)
        return f'{percentage(old_value)} → {percentage(new_value)} ({delta})'
    return f'{percentage(old_value)} → {percentage(new_value)}'


def _gap_change(
    old_value: float | None,
    old_bounds: Sequence[float] | None,
    new_value: float | None,
    new_bounds: Sequence[float] | None,
) -> str:
    before, after = bounded(old_value, old_bounds), bounded(new_value, new_bounds)
    if _same_float(old_value, new_value) and _same_bounds(old_bounds, new_bounds):
        return after
    delta = _bounded_delta_pp(old_value, old_bounds, new_value, new_bounds)
    return f'{before} → {after} ({delta})' if delta else f'{before} → {after}'


def _chapter_link(
    cid: str,
    current_chapters: Mapping[str, JsonObject],
    previous_chapters: Mapping[str, JsonObject],
    *,
    full: bool = True,
    with_study: bool = False,
) -> str:
    if cid in current_chapters:
        ch = current_chapters[cid]
        label = f"{ch['code']}. {escape(ch['name'])}" if full else ch['code']
        text = f"[{label}](chapters/{ch['code']}.md)"
        if with_study and ch.get('url'):
            text += f"<br>[Lichess study]({ch['url']})"
        return text
    if cid in previous_chapters:
        ch = previous_chapters[cid]
        text = escape(ch['name']) if full else f"{ch['code']} (deleted)"
        if with_study and ch.get('url'):
            text += f"<br>[Lichess study]({ch['url']})"
        return text
    return escape(cid)


def _branch_cell(position: Position, route: str, color: str) -> str:
    if not route:
        label = 'Starting position' if position == STARTING_POSITION else 'Chapter root'
        return f'[{label}]({analysis_url(position, color=color)})'
    return linked_line(route, position, color)


def _continuation_cell(row: JsonObject, color: str) -> str:
    text = line(row['continuation'])
    return f'[{text}]({analysis_url(row["end_position"], row["full_line"], color)})'


def _chapter_sort_key(
    cid: str, current_chapters: Mapping[str, JsonObject], previous_chapters: Mapping[str, JsonObject]
) -> tuple[int, int, str]:
    if cid in current_chapters:
        code = str(current_chapters[cid].get('code', ''))
        return (0, int(code[1:]) if code[1:].isdigit() else 999_999, cid)
    if cid in previous_chapters:
        code = str(previous_chapters[cid].get('code', ''))
        return (1, int(code[1:]) if code[1:].isdigit() else 999_999, cid)
    return (2, 999_999, cid)


def _scope_text(
    row: JsonObject,
    *,
    is_addition: bool,
    current_chapters: Mapping[str, JsonObject],
    previous_chapters: Mapping[str, JsonObject],
) -> str:
    scope = row['scope']
    ordered_related = sorted(
        row.get('related_chapters', []),
        key=lambda cid: _chapter_sort_key(cid, current_chapters, previous_chapters),
    )
    related = [_chapter_link(cid, current_chapters, previous_chapters, full=False) for cid in ordered_related]
    joined = ', '.join(related)
    if is_addition:
        if scope == 'new':
            label = 'New to repertoire'
        elif scope == 'moved' and joined:
            label = f'Moved from {joined}'
        elif scope == 'existing' and joined:
            label = f'Already in {joined}'
        elif scope == 'existing_in_chapter':
            label = 'Already in this chapter'
        else:
            label = 'Already in repertoire'
        if row.get('replaced_mainline'):
            label += '; new main line'
        return label
    if scope == 'removed':
        label = 'Removed from repertoire'
    elif scope == 'moved' and joined:
        label = f'Moved to {joined}'
    elif scope == 'kept_elsewhere' and joined:
        label = f'Still in {joined}'
    elif scope == 'kept_in_chapter':
        label = 'Still in this chapter'
    else:
        label = 'Still in repertoire'
    if row.get('is_mainline') and row.get('replaced_mainline'):
        label += '; was main line'
    return label


def _summary_table(data: JsonObject) -> list[str]:
    rows: list[list[Cell]] = []
    for color in ('white', 'black'):
        item = data['colors'].get(color)
        if not item:
            continue
        before, after = item['overall']['before'], item['overall']['after']
        rows.append(
            [
                f'[{color.title()}](#{color})' if item['has_changes'] else color.title(),
                _int_change(before.get('chapters_count'), after.get('chapters_count')),
                _int_change(before.get('distinct_decisions'), after.get('distinct_decisions')),
                _score_change(
                    before.get('score'),
                    before.get('score_bounds'),
                    after.get('score'),
                    after.get('score_bounds'),
                ),
                _delta_change(
                    before.get('score'),
                    before.get('baseline'),
                    before.get('score_bounds'),
                    after.get('score'),
                    after.get('baseline'),
                    after.get('score_bounds'),
                ),
                _depth_change(before.get('prepared_depth'), after.get('prepared_depth')),
                _gap_change(
                    before.get('equivalent_gap_reach'),
                    before.get('equivalent_gap_reach_bounds'),
                    after.get('equivalent_gap_reach'),
                    after.get('equivalent_gap_reach_bounds'),
                ),
            ]
        )
    combined = data.get('combined')
    if combined and combined.get('before') and combined.get('after'):
        before, after = combined['before'], combined['after']
        rows.append(
            [
                '**Combined**',
                _int_change(before.get('chapters_count'), after.get('chapters_count')),
                _int_change(before.get('distinct_decisions'), after.get('distinct_decisions')),
                _score_change(
                    before.get('score'),
                    before.get('score_bounds'),
                    after.get('score'),
                    after.get('score_bounds'),
                ),
                _delta_change(
                    before.get('score'),
                    before.get('baseline'),
                    before.get('score_bounds'),
                    after.get('score'),
                    after.get('baseline'),
                    after.get('score_bounds'),
                ),
                _depth_change(before.get('prepared_depth'), after.get('prepared_depth')),
                '-',
            ]
        )
    return table(
        [
            'Repertoire',
            'Chapters',
            'Own moves',
            'Repertoire score',
            'Delta',
            'Prepared depth',
            'Equivalent gap reach',
        ],
        rows,
    )


def _config_and_evidence_section(item: JsonObject) -> list[str]:
    color = item['color']
    current_chapters = item['current_chapters']
    previous_chapters = item['previous_chapters']
    bullets: list[str] = []

    for change in item.get('config', []):
        key = change['key']
        if key == 'chapter_order':
            bullets.append(f"**Chapter order:** `{change['old']}` → `{change['new']}`.")
        elif key == 'coverage_goal':
            old_g, new_g = change['old'], change['new']
            old_cov = item['overall']['before'].get('coverage') or {}
            new_cov = item['overall']['after'].get('coverage') or {}
            detail = ''
            if old_cov and new_cov:
                moves = _int_change(old_cov.get('decisions'), new_cov.get('decisions'))
                score = _score_change(
                    old_cov.get('score'), old_cov.get('score_bounds'), new_cov.get('score'), new_cov.get('score_bounds')
                )
                detail = f' (kept moves: {moves}; trimmed score: {score})'
            bullets.append(f'**Coverage goal:** 1 in {old_g:,} games → 1 in {new_g:,} games{detail}.')
        elif key == 'policy':
            parts = []
            for p in change['changes']:
                where = _branch_cell(p['position'], p['line'], color)
                if p['action'] == 'added':
                    parts.append(f"added {where}: `{escape(p['new'])}`")
                elif p['action'] == 'removed':
                    parts.append(f"removed {where} (was `{escape(p['old'])}`)")
                else:
                    parts.append(f"changed {where}: `{escape(p['old'])}` → `{escape(p['new'])}`")
            bullets.append('**Policy overrides:** ' + '; '.join(parts) + '.')
        elif key == 'exclude':
            parts = []
            if change['added']:
                parts.append('added ' + ', '.join(f'`{escape(e)}`' for e in change['added']))
            if change['removed']:
                parts.append('removed ' + ', '.join(f'`{escape(e)}`' for e in change['removed']))
            bullets.append('**Exclusions:** ' + '; '.join(parts) + '.')
        elif key == 'entries':
            parts = []
            for verb, field in (('added', 'added'), ('removed', 'removed'), ('updated', 'changed')):
                cids = change.get(field, [])
                if cids:
                    links = ', '.join(_chapter_link(c, current_chapters, previous_chapters, full=False) for c in cids)
                    parts.append(f'{verb} for {links}')
            bullets.append('**Configured chapter entries:** ' + '; '.join(parts) + '.')
        elif key == 'filters':
            parts = [f"`{k}`: `{escape(v['old'])}` → `{escape(v['new'])}`" for k, v in change['changes'].items()]
            bullets.append('**Explorer filters:** ' + '; '.join(parts) + '.')
        elif key == 'root_weights':
            bullets.append('**Root weights:** updated.')
        elif key == 'prior':
            bullets.append(f"**Prior:** `{change['old']}` → `{change['new']}`.")
        elif key == 'sparse_threshold':
            bullets.append(f"**Sparse threshold:** {change['old']} → {change['new']} games.")

    # Coverage summary when coverage goal itself did not change, but trimmed moves or trimmed score changed.
    if not any(c['key'] == 'coverage_goal' for c in item.get('config', [])):
        old_cov = item['overall']['before'].get('coverage')
        new_cov = item['overall']['after'].get('coverage')
        if (
            old_cov
            and new_cov
            and (
                old_cov.get('decisions') != new_cov.get('decisions')
                or not _same_float(old_cov.get('score'), new_cov.get('score'))
                or not _same_bounds(old_cov.get('score_bounds'), new_cov.get('score_bounds'))
            )
        ):
            moves = _int_change(old_cov.get('decisions'), new_cov.get('decisions'))
            score = _score_change(
                old_cov.get('score'), old_cov.get('score_bounds'), new_cov.get('score'), new_cov.get('score_bounds')
            )
            bullets.append(
                f"**Trimmed repertoire (1 in {new_cov['goal']:,} games):** kept moves {moves}; trimmed score {score}."
            )

    ev = item.get('evidence') or {}
    if ev.get('changed'):
        bullets.append(
            f"**Explorer evidence:** refreshed for {_plural(ev['refreshed_positions'], 'position')} "
            f"({ev['tables_before']:,} → {ev['tables_after']:,} tables)."
        )

    eng = item.get('engine') or {}
    if eng.get('changed'):
        before_e, after_e = eng.get('before') or {}, eng.get('after') or {}
        if after_e.get('resolved') and not before_e.get('resolved'):
            exits = after_e.get('exits_score')
            score_part = (
                f"; where prep ends: {pawns(centipawn_equivalent(exits))} ({percentage(exits)})"
                if exits is not None
                else ''
            )
            bullets.append(
                f"**Engine evaluations:** imported ({after_e.get('evaluated_positions', 0):,} positions{score_part})."
            )
        elif before_e.get('resolved') and not after_e.get('resolved'):
            bullets.append('**Engine evaluations:** no longer available.')
        else:
            pos_text = _int_change(before_e.get('evaluated_positions'), after_e.get('evaluated_positions'))
            old_s, new_s = before_e.get('exits_score'), after_e.get('exits_score')
            score_part = ''
            if not _same_float(old_s, new_s):
                old_lbl = f'{pawns(centipawn_equivalent(old_s))} ({percentage(old_s)})' if old_s is not None else 'n/a'
                new_lbl = f'{pawns(centipawn_equivalent(new_s))} ({percentage(new_s)})' if new_s is not None else 'n/a'
                score_part = f'; where prep ends: {old_lbl} → {new_lbl}'
            bullets.append(f'**Engine evaluations:** {pos_text} evaluated positions{score_part}.')

    if not bullets:
        return []
    return [*section('### Configuration and evidence', f'{color}-config'), *[f'- {b}' for b in bullets], '']


def _order_block(
    title: str,
    order_diff: JsonObject,
    current_chapters: Mapping[str, JsonObject],
    previous_chapters: Mapping[str, JsonObject],
    *,
    note: str = '',
) -> list[str]:
    moved = order_diff['moved']
    shifted = order_diff['shifted']
    if not moved:
        return []
    rows: list[list[Cell]] = []
    for entry in moved:
        diff = entry['old_index'] - entry['new_index']
        direction = f'Up {diff}' if diff > 0 else f'Down {-diff}' if diff < 0 else 'Same position'
        rows.append(
            [
                _chapter_link(entry['id'], current_chapters, previous_chapters),
                entry['old_code'],
                entry['new_code'],
                direction,
            ]
        )
    text = [f'**{title} ({len(moved)} moved)**', '']
    if note:
        text += [note, '']
    text += table(['Chapter', 'Previous position', 'Current position', 'Shift'], rows)
    if shifted:
        shifted_rows: list[list[Cell]] = [
            [
                _chapter_link(entry['id'], current_chapters, previous_chapters),
                entry['old_code'],
                entry['new_code'],
            ]
            for entry in shifted
        ]
        text += [
            '<details>',
            f'<summary>{_plural(len(shifted), "other chapter")} shifted position</summary>',
            '',
            *table(['Chapter', 'Previous position', 'Current position'], shifted_rows),
            '</details>',
            '',
        ]
    return text


def _chapters_section(item: JsonObject) -> list[str]:
    color = item['color']
    chapters = item['chapters']
    current_chapters = item['current_chapters']
    previous_chapters = item['previous_chapters']
    added = chapters['added']
    deleted = chapters['deleted']
    renamed = chapters['renamed']
    report_order = chapters['report_order']
    study_order = chapters['study_order']
    changed = chapters['changed']
    mode_before = item['chapter_order_mode']['before']
    mode_after = item['chapter_order_mode']['after']

    if not any((added, deleted, renamed, report_order['reordered'], study_order['reordered'], changed)):
        return []

    text = section('### Chapters', f'{color}-chapters')

    if added:
        text += [f'**Added chapters ({len(added)})**', '']
        rows: list[list[Cell]] = [
            [
                _chapter_link(c['id'], current_chapters, previous_chapters, with_study=True),
                percentage(c.get('entry_probability')),
                _score_text(c.get('score'), c.get('score_bounds')),
                _delta_text(c.get('score'), c.get('baseline'), c.get('score_bounds')),
                number(c.get('prepared_depth'), 1),
                f"{c['line_count']:,}",
            ]
            for c in added
        ]
        text += table(['Chapter', 'Chapter reach', 'Repertoire score', 'Delta', 'Prepared depth', 'Lines'], rows)

    if deleted:
        text += [f'**Deleted chapters ({len(deleted)})**', '']
        rows = [
            [
                _chapter_link(c['id'], current_chapters, previous_chapters, with_study=True),
                c['code'],
                percentage(c.get('entry_probability')),
                _score_text(c.get('score'), c.get('score_bounds')),
                number(c.get('prepared_depth'), 1),
                f"{c['line_count']:,}",
            ]
            for c in deleted
        ]
        text += table(
            ['Chapter', 'Previous position', 'Previous reach', 'Previous score', 'Previous depth', 'Lines'],
            rows,
        )

    if renamed:
        text += [f'**Renamed chapters ({len(renamed)})**', '']
        rows = [
            [
                _chapter_link(c['id'], current_chapters, previous_chapters, full=False),
                escape(c['old_name']),
                escape(c['new_name']),
            ]
            for c in renamed
        ]
        text += table(['Chapter', 'Previous name', 'New name'], rows)

    if report_order['reordered']:
        note = ''
        if mode_before != mode_after:
            note = f'Chapter order switched from `{mode_before}` to `{mode_after}`.'
        elif mode_after == 'opening':
            note = 'Chapters are ordered by opening line and group reach (`chapter_order: "opening"`).'
        text += _order_block('Reordered chapters', report_order, current_chapters, previous_chapters, note=note)
    elif (added or deleted) and report_order['shifted']:
        shifted = report_order['shifted']
        text += [
            f'{_plural(len(shifted), "existing chapter")} kept their relative order and shifted numbering '
            f'due to chapter additions or deletions.',
            '',
        ]

    # When using opening order, also report study/PGN order changes if the study itself was reordered.
    if study_order['reordered'] and (mode_after != 'study' or mode_before != 'study'):
        same_moves = report_order['reordered'] and [m['id'] for m in report_order['moved']] == [
            m['id'] for m in study_order['moved']
        ]
        if not same_moves:
            note = (
                'Study/PGN chapter order changed; report chapter numbering (`opening` order) was unaffected.'
                if not report_order['reordered']
                else 'Study/PGN chapter order also changed.'
            )
            text += _order_block(
                'Study chapter order changed', study_order, current_chapters, previous_chapters, note=note
            )

    if changed:
        text += [f'**Changed chapters ({len(changed)})**', '']
        rows = []
        for ch in changed:
            before, after = ch['before'], ch['after']
            notes: list[str] = []
            if ch['lines_added']:
                notes.append(f"+{_plural(ch['lines_added'], 'line')}")
            if ch['lines_deleted']:
                notes.append(f"-{_plural(ch['lines_deleted'], 'line')}")
            if ch['lines_reordered']:
                notes.append(f"{_plural(ch['lines_reordered'], 'variation')} reordered")
            if ch['entries_changed']:
                notes.append('entries changed')
            if ch['alternative_changed']:
                notes.append('now alternative' if after.get('is_alternative') else 'now played')
            rows.append(
                [
                    _chapter_link(ch['id'], current_chapters, previous_chapters),
                    _reach_change(before.get('entry_probability'), after.get('entry_probability')),
                    _score_change(
                        before.get('score'),
                        before.get('score_bounds'),
                        after.get('score'),
                        after.get('score_bounds'),
                    ),
                    _depth_change(before.get('prepared_depth'), after.get('prepared_depth')),
                    _int_change(before.get('distinct_decisions'), after.get('distinct_decisions')),
                    ', '.join(notes) or 'Metrics updated',
                ]
            )
        text += table(
            ['Chapter', 'Chapter reach', 'Repertoire score', 'Prepared depth', 'Own moves', 'Notes'],
            rows,
        )

    return text


def _played_moves_section(item: JsonObject) -> list[str]:
    color = item['color']
    played = item.get('played_moves', [])
    alts = item.get('alternatives', {})
    has_alts = bool(alts.get('added') or alts.get('removed') or alts.get('changed'))
    if not played and not has_alts:
        return []
    text = section('### Move selection and alternatives', f'{color}-moves')
    if played:
        text += [
            'Positions present in both builds where the repertoire plays a different move.',
            '',
            *table(
                ['Position', 'Position reach', 'Previous move', 'Current move', 'Reason'],
                [
                    [
                        _branch_cell(row['position'], row['line'], color),
                        percentage(row.get('reach')),
                        escape(row['old_label']),
                        escape(row['new_label']),
                        escape(row['reason']),
                    ]
                    for row in played
                ],
            ),
        ]
    if has_alts:
        bullets: list[str] = []
        for alt in alts.get('added', []):
            opts = ', '.join(escape(o['san']) for o in alt['options'])
            bullets.append(
                f"New competing moves at {_branch_cell(alt['position'], alt['line'], color)}: "
                f"{opts} (plays `{escape(alt['selected_san'])}`)."
            )
        for alt in alts.get('removed', []):
            opts = ', '.join(escape(o['san']) for o in alt['options'])
            bullets.append(f"No longer competing at {_branch_cell(alt['position'], alt['line'], color)} (was {opts}).")
        for alt in alts.get('changed', []):
            old_opts = ', '.join(escape(s) for s in alt['old_options'])
            new_opts = ', '.join(escape(s) for s in alt['new_options'])
            detail = (
                f"winner `{escape(alt['old_selected'])}` → `{escape(alt['new_selected'])}`"
                if alt['old_selected'] != alt['new_selected']
                else f"options {old_opts} → {new_opts}"
            )
            bullets.append(f"Competing moves updated at {_branch_cell(alt['position'], alt['line'], color)}: {detail}.")
        text += ['**Competing alternatives**', '', *[f'- {b}' for b in bullets], '']
    return text


def _chunked_table(headers: Sequence[str], rows: Sequence[Sequence[Cell]], noun: str) -> list[str]:
    if len(rows) <= LINE_LIMIT:
        return table(headers, rows)
    text = table(headers, rows[:LINE_LIMIT])
    remaining = rows[LINE_LIMIT:]
    text += [
        '<details>',
        f'<summary>Show remaining {_plural(len(remaining), noun)}</summary>',
        '',
        *table(headers, remaining),
        '</details>',
        '',
    ]
    return text


def _lines_section(item: JsonObject) -> list[str]:
    color = item['color']
    line_changes = item.get('lines', [])
    if not line_changes:
        return []
    current_chapters = item['current_chapters']
    previous_chapters = item['previous_chapters']
    text = section('### Line changes', f'{color}-lines')
    text += [
        'Line additions, deletions and variation reorderings within existing chapters. '
        'Lines in newly added or deleted chapters are summarized under Chapters above.',
        '',
    ]
    for ch in line_changes:
        cid = ch['id']
        heading = _chapter_link(cid, current_chapters, previous_chapters)
        text += [f'#### {heading}', '']
        if ch.get('root_changed'):
            text += ['Chapter starting position (custom FEN) changed.', '']

        added = ch.get('added', [])
        if added:
            text += [f"**Added lines ({len(added)}, {_plural(ch['added_moves'], 'move')})**", '']
            rows: list[list[Cell]] = [
                [
                    _branch_cell(row['branch_position'], row['branch_line'], color),
                    _continuation_cell(row, color),
                    f"{row['plies']} ({row['own_moves']} own)",
                    _scope_text(
                        row,
                        is_addition=True,
                        current_chapters=current_chapters,
                        previous_chapters=previous_chapters,
                    ),
                ]
                for row in added
            ]
            text += _chunked_table(['Branch point', 'Added continuation', 'Plies', 'Scope'], rows, 'added line')

        deleted = ch.get('deleted', [])
        if deleted:
            text += [f"**Deleted lines ({len(deleted)}, {_plural(ch['deleted_moves'], 'move')})**", '']
            rows = [
                [
                    _branch_cell(row['branch_position'], row['branch_line'], color),
                    _continuation_cell(row, color),
                    f"{row['plies']} ({row['own_moves']} own)",
                    _scope_text(
                        row,
                        is_addition=False,
                        current_chapters=current_chapters,
                        previous_chapters=previous_chapters,
                    ),
                ]
                for row in deleted
            ]
            text += _chunked_table(['Branch point', 'Deleted continuation', 'Plies', 'Scope'], rows, 'deleted line')

        reordered = ch.get('reordered', [])
        if reordered:
            text += [f'**Reordered variations ({len(reordered)})**', '']
            rows = []
            for row in reordered:
                effect = (
                    f"Main line changed ({line(row['old_mainline'])} → {line(row['new_mainline'])})"
                    if row.get('mainline_changed')
                    else 'Side variation order changed'
                )
                rows.append(
                    [
                        _branch_cell(row['position'], row['line'], color),
                        ', '.join(line(m) for m in row['old_order']),
                        ', '.join(line(m) for m in row['new_order']),
                        effect,
                    ]
                )
            text += _chunked_table(
                ['Branch point', 'Previous order', 'Current order', 'Notes'], rows, 'reordered variation'
            )

    return text


def _color_section(item: JsonObject) -> list[str]:
    color = item['color']
    if not item['has_changes']:
        return []
    text = section(f'## {color.title()} changes', color)
    text += _config_and_evidence_section(item)
    text += _chapters_section(item)
    text += _played_moves_section(item)
    text += _lines_section(item)
    return text


def render(data: JsonObject) -> str:
    """Render a structured change diff (`changes.json`) to Markdown (`changes.md`)."""
    with display(digits=1):
        links = ['[Summary](summary.md)', '[Full report](report.md)']
        for color in ('white', 'black'):
            if (data.get('colors') or {}).get(color, {}).get('has_changes'):
                links.append(f'[{color.title()} changes](#{color})')
        text = [
            '# Repertoire changes',
            '',
            ' · '.join(links),
            '',
            f"Changes between the previous build ({_timestamp(data.get('previous_built_at'))}) and the current build "
            f"({_timestamp(data.get('current_built_at'))}). Only the most recent build with changes is kept; "
            'unchanged builds preserve this report.',
            '',
            summary_text(data),
            '',
            *section('## Summary', 'summary'),
            *_summary_table(data),
        ]
        for color in ('white', 'black'):
            item = (data.get('colors') or {}).get(color)
            if item:
                text += _color_section(item)
        return '\n'.join(row.rstrip() for row in text) + '\n'
