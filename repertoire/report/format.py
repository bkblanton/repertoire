"""Number, score, centipawn, rating and label formatting shared by every page."""

import html
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType
from typing import cast

from ..explorer import DEFAULT_FILTERS
from ..layout import report_directory
from ..schema import JsonObject

LICHESS_WIN_CHANCE_COEFFICIENT = 0.00368208


# Every page shows one decimal and compact game counts; scores are rarely known more precisely than that.
_display: ContextVar[Mapping[str, int | bool]] = ContextVar(
    'display', default=MappingProxyType({'digits': 1, 'compact_counts': True})
)


@contextmanager
def display(**options: int | bool) -> Iterator[None]:
    token = _display.set(dict(_display.get(), **options))
    try:
        yield
    finally:
        _display.reset(token)


def percentage(value: float | None, digits: int | None = None) -> str:
    if value is None:
        return 'unresolved'
    if digits is None:
        digits = _display.get()['digits']
        # Rare positions and moves keep a second significant digit, so 0.43% and 0.38% stay distinct.
        if 0 < abs(100 * value) < 1:
            digits += 1
    smallest = 10**-digits
    return f'<{smallest:.{digits}f}%' if 0 < 100 * value < smallest / 2 else f'{100 * value:.{digits}f}%'


def number(value: float | None, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return 'unresolved'
    if signed and 0 < abs(value) < 0.5 * 10**-digits:
        return ('+<' if value > 0 else '-<') + f'{10**-digits:.{digits}f}'
    return format(value, f'{"+" if signed else ""}.{digits}f')


def score_points(value: float | None, digits: int | None = None, signed: bool = False) -> str:
    """Display an already percentage-scaled score difference or contribution."""
    digits = _display.get()['digits'] if digits is None else digits
    return 'unresolved' if value is None else number(value, digits, signed) + '%'


def per_thousand(value_pp: float | None, signed: bool = False) -> str:
    """Reach-weighted percentage points as score points per 1,000 games with that color or scope."""
    if value_pp is None:
        return 'unresolved'
    value = 10 * value_pp
    digits = 0 if abs(value) >= 100 else 1 if abs(value) >= 1 else 2
    return number(value, digits, signed)


def count(value: int | None) -> str:
    if value is None:
        return 'unavailable'
    if not _display.get()['compact_counts'] or value < 10_000:
        return f'{value:,}'
    for size, suffix in ((1e9, 'B'), (1e6, 'M')):
        if value >= size:
            return f'{value / size:.1f}{suffix}'
    return f'{value / 1e3:.0f}k'


def plies(route: str | None) -> int:
    route = (route or '').strip()
    return 0 if route in ('', '(PGN root)') else len(route.split())


def games_per_encounter(probability: float | None) -> str:
    """Mean waiting interval in independent games with the displayed reach basis."""
    if probability is None:
        return 'unavailable'
    if probability <= 0:
        return 'never'
    value = 1 / probability
    return f'{value:,.0f}' if value >= 10 else f'{value:.1f}'


def encounters(probability: float | None, color: str = '') -> str:
    """How often a reach comes up, in words: 1 in 43 games, or 1 in 43 White games."""
    noun = f'{color} game'.strip()
    if probability is not None and probability > 0.995:
        return f'every {noun}'
    games = games_per_encounter(probability)
    return games if games in ('unavailable', 'never') else f'1 in {games} {noun}s'


def reach_cell(probability: float | None) -> str:
    return percentage(probability) + '<br>' + encounters(probability)


def escape(value: object) -> str:
    return html.escape(str(value), quote=False).replace('|', '&#124;').replace('\n', ' ')


def line(value: str) -> str:
    """Display repeated PGN move numbers as conventional paired moves."""
    value = ' '.join(value.split())
    value = re.sub(r'(?<!\S)(\d+)\.(?!\.)(\S+)\s+\1\.\.\.(\S+)', r'\1. \2 \3', value)
    value = re.sub(r'(?<!\S)(\d+)\.\.\.(?=\S)', r'\1... ', value)
    value = re.sub(r'(?<!\S)(\d+)\.(?!\.)', r'\1. ', value)
    return escape(' '.join(value.split()) or '(starting position)')


def elo_equivalent(score: float | None, baseline: float | None) -> float | None:
    """Difference of logistic Elo score equivalents; boundaries are undefined."""
    if score is None or baseline is None or not (0 < score < 1 and 0 < baseline < 1):
        return None
    return 400 * (math.log10(score / (1 - score)) - math.log10(baseline / (1 - baseline)))


def centipawn_equivalent(score: float | None) -> float | None:
    """Direct inverse Lichess score equivalent, from the repertoire owner's view."""
    if score is None or not 0 < score < 1:
        return None
    return math.log(score / (1 - score)) / LICHESS_WIN_CHANCE_COEFFICIENT


def centipawn_delta(after: float | None, before: float | None) -> float | None:
    """Convert both scores first, then subtract before from after."""
    converted = [centipawn_equivalent(p) for p in (after, before)]
    return None if any(p is None for p in converted) else cast(float, converted[0]) - cast(float, converted[1])


def cp(score: float | None) -> str:
    value = centipawn_equivalent(score)
    return 'unavailable' if value is None else number(value, signed=True)


def cp_change(after: float | None, before: float | None) -> str:
    value = centipawn_delta(after, before)
    return 'unavailable' if value is None else number(value, signed=True)


def cp_interval(bounds: Sequence[float | None] | None) -> str:
    return ' to '.join(map(cp, bounds)) if bounds else 'unavailable'


def spread_display(spread: JsonObject | None, immediate: bool = True) -> str:
    if not spread:
        return 'unavailable'
    result = percentage(spread.get('standard_deviation'))
    if immediate and spread.get('immediate_standard_deviation') is not None:
        result += '<br>Reply ' + percentage(spread['immediate_standard_deviation'])
    elif immediate and spread.get('basis') == 'stopping score; no modeled future replies':
        result += '<br>Prep ends'
    return result


def score_cell(value: float | None) -> str:
    return percentage(value) + ' (' + cp(value) + ' cp)'


def delta_cell(after: float | None, before: float | None, *, drag: bool = False) -> str:
    delta = None if after is None or before is None else 100 * (before - after if drag else after - before)
    return score_points(delta, signed=True) + ' (' + cp_change(after, before) + ' cp)'


def delta_points(after: float | None, before: float | None, *, drag: bool = False) -> str:
    delta = None if after is None or before is None else 100 * (before - after if drag else after - before)
    return score_points(delta, signed=True)


def headline_delta(after: float | None, before: float | None) -> str:
    """Headline deltas alone keep a centipawn-scale translation; other tables show percentages only."""
    value = centipawn_delta(after, before)
    return delta_points(after, before) + ('' if value is None else f' ({number(value, 0, signed=True)} cp)')


def interval_cell(bounds: Sequence[float | None] | None) -> str:
    return ' to '.join(score_points(v, signed=True) for v in bounds) if bounds else 'unavailable'


def gain_split(row: JsonObject) -> str:
    if row.get('database_move_gain_pp') is None:
        return 'unavailable'
    return (
        'Move ' + score_points(row['database_move_gain_pp'], signed=True) + '<br>'
        'Prep ' + score_points(row['continuation_gain_pp'], signed=True)
    )


def population_text(report: JsonObject) -> str:
    filters = report['manifest']['filters']
    speeds = filters.get('speeds', 'configured speeds').replace(',', ', ')
    ratings = filters.get('ratings', '')
    bands = 'all ratings' if ratings == DEFAULT_FILTERS['ratings'] else f"rating groups {ratings or 'as configured'}"
    since, until = filters.get('since'), filters.get('until')
    dates = (
        'all available dates'
        if (since, until) == (DEFAULT_FILTERS['since'], DEFAULT_FILTERS['until'])
        else f"{since or 'start'} to {until or 'end'}"
    )
    return f"Lichess rated {speeds}; {bands}; {dates}."


def evidence_date(report: JsonObject) -> str:
    dates = sorted(
        {
            p['retrieved_at'][:10]
            for p in report['manifest'].get('evidence', {}).values()
            if p.get('retrieved_at', '')[:4].isdigit()
        }
    )
    return (
        (dates[0] if dates[0] == dates[-1] else f'{dates[0]} to {dates[-1]}')
        if dates
        else report['manifest'].get('created_at', '')[:10]
    )


def display_source(source: str | Path, result_path: str | Path) -> str:
    """Show the PGN relative to the report folder, so published reports carry no local absolute path."""
    try:
        return Path(os.path.relpath(source, report_directory(result_path))).as_posix()
    except ValueError:
        # Different Windows drives have no relative path.
        return Path(source).name


def opponent_rating(context: JsonObject | None) -> str:
    if not context or context.get('mean') is None:
        if context and context.get('reason') == 'no_preceding_opponent_move':
            return 'n/a (no preceding opponent move)'
        return 'unavailable'
    result = f"{context['mean']:,.0f}"
    if context['known_coverage'] < 1 - 1e-9:
        result += f" ({100 * context['known_coverage']:.1f}% rated)"
    return result


def rating_difference(context: JsonObject | None) -> str:
    if context and context.get('reason') == 'no_preceding_opponent_move':
        return 'n/a (no preceding opponent move)'
    if context and context.get('basis') == 'current opponent response rows' and 'difference_vs_parent' not in context:
        return 'n/a'
    if not context or context.get('difference_vs_parent') is None:
        return 'unavailable'
    result = f"{context['difference_vs_parent']:+,.0f}"
    notes = []
    if context.get('comparison_coverage', 1.0) < 1 - 1e-9:
        notes.append(f"{100 * context['comparison_coverage']:.1f}% compared")
    if context.get('parent_known_coverage', 1.0) < 1 - 1e-9:
        notes.append(f"parent {100 * context['parent_known_coverage']:.1f}% rated")
    return result + (' (' + '; '.join(notes) + ')' if notes else '')


def rating_cell(context: JsonObject | None) -> str:
    """Average opponent rating, with its difference from the parent position when there is one."""
    rating = opponent_rating(context)
    difference = rating_difference(context)
    if difference.startswith(('n/a', 'unavailable')):
        return rating
    value, _, notes = difference.partition(' (')
    return f'{rating}<br>{value} vs parent' + (f' ({notes}' if notes else '')


def gap_percentage(metrics: JsonObject | None, weighted: bool = False) -> str:
    metrics = metrics or {}
    prefix = 'weighted_' if weighted else ''
    value = metrics.get(prefix + 'equivalent_gap_reach')
    if value is not None:
        return percentage(value)
    bounds = metrics.get(prefix + 'equivalent_gap_reach_bounds')
    return ' to '.join(map(percentage, bounds)) + ' (bounds)' if bounds is not None else 'unavailable'


def position_reach_label(scope: JsonObject) -> str:
    """Board reach combines every transposed arrival."""
    return 'Position reach' if scope.get('id', 'overall') == 'overall' else 'Position reach after chapter entry'


def move_reach_label(scope: JsonObject) -> str:
    """Move reach counts only arrivals at the parent that continue with this move."""
    return 'Move reach' if scope.get('id', 'overall') == 'overall' else 'Move reach after chapter entry'
