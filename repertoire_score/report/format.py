"""Number, score, centipawn, rating and label formatting shared by every page."""

import html
import math
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType

from ..explorer import DEFAULT_FILTERS
from ..layout import report_directory

LICHESS_WIN_CHANCE_COEFFICIENT = 0.00368208


# The summary uses one decimal and compact game counts; the full report keeps two decimals.
_display = ContextVar('display', default=MappingProxyType({'digits': 2, 'compact_counts': False}))


@contextmanager
def display(**options):
    token = _display.set(dict(_display.get(), **options))
    try:
        yield
    finally:
        _display.reset(token)


def percentage(value, digits=None):
    if value is None:
        return 'unresolved'
    digits = _display.get()['digits'] if digits is None else digits
    smallest = 10**-digits
    return f'<{smallest:.{digits}f}%' if 0 < 100 * value < smallest / 2 else f'{100 * value:.{digits}f}%'


def number(value, digits=2, signed=False):
    if value is None:
        return 'unresolved'
    if signed and 0 < abs(value) < 0.5 * 10**-digits:
        return ('+<' if value > 0 else '-<') + f'{10**-digits:.{digits}f}'
    return format(value, f'{"+" if signed else ""}.{digits}f')


def score_points(value, digits=None, signed=False):
    """Display an already percentage-scaled score difference or contribution."""
    digits = _display.get()['digits'] if digits is None else digits
    return 'unresolved' if value is None else number(value, digits, signed) + '%'


def per_thousand(value_pp, signed=False):
    """Reach-weighted percentage points as score points per 1,000 games with that color or scope."""
    if value_pp is None:
        return 'unresolved'
    value = 10 * value_pp
    digits = 0 if abs(value) >= 100 else 1 if abs(value) >= 1 else 2
    return number(value, digits, signed)


def count(value):
    if value is None:
        return 'unavailable'
    if not _display.get()['compact_counts'] or value < 10_000:
        return f'{value:,}'
    for size, suffix in ((1e9, 'B'), (1e6, 'M')):
        if value >= size:
            return f'{value / size:.1f}{suffix}'
    return f'{value / 1e3:.0f}k'


def plies(route):
    route = (route or '').strip()
    return 0 if route in ('', '(PGN root)') else len(route.split())


def games_per_encounter(probability):
    """Mean waiting interval in independent games with the displayed reach basis."""
    if probability is None:
        return 'unavailable'
    if probability <= 0:
        return 'never'
    value = 1 / probability
    return f'{value:,.0f}' if value >= 10 else f'{value:.1f}'


def escape(value):
    return html.escape(str(value), quote=False).replace('|', '&#124;').replace('\n', ' ')


def line(value):
    """Display repeated PGN move numbers as conventional paired moves."""
    value = ' '.join(value.split())
    value = re.sub(r'(?<!\S)(\d+)\.(?!\.)(\S+)\s+\1\.\.\.(\S+)', r'\1. \2 \3', value)
    value = re.sub(r'(?<!\S)(\d+)\.\.\.(?=\S)', r'\1... ', value)
    value = re.sub(r'(?<!\S)(\d+)\.(?!\.)', r'\1. ', value)
    return escape(' '.join(value.split()) or '(starting position)')


def elo_equivalent(score, baseline):
    """Difference of logistic Elo score equivalents; boundaries are undefined."""
    if score is None or baseline is None or not (0 < score < 1 and 0 < baseline < 1):
        return None
    return 400 * (math.log10(score / (1 - score)) - math.log10(baseline / (1 - baseline)))


def centipawn_equivalent(score):
    """Direct inverse Lichess score equivalent, from the repertoire owner's view."""
    if score is None or not 0 < score < 1:
        return None
    return math.log(score / (1 - score)) / LICHESS_WIN_CHANCE_COEFFICIENT


def centipawn_delta(after, before):
    """Convert both scores first, then subtract before from after."""
    converted = [centipawn_equivalent(p) for p in (after, before)]
    return None if any(p is None for p in converted) else converted[0] - converted[1]


def cp(score):
    value = centipawn_equivalent(score)
    return 'unavailable' if value is None else number(value, signed=True)


def cp_change(after, before):
    value = centipawn_delta(after, before)
    return 'unavailable' if value is None else number(value, signed=True)


def cp_interval(bounds):
    return ' to '.join(map(cp, bounds)) if bounds else 'unavailable'


def spread_display(spread, immediate=True):
    if not spread:
        return 'unavailable'
    result = percentage(spread.get('standard_deviation'))
    if immediate and spread.get('immediate_standard_deviation') is not None:
        result += '<br>Reply ' + percentage(spread['immediate_standard_deviation'])
    elif immediate and spread.get('basis') == 'stopping score; no modeled future replies':
        result += '<br>Prep ends'
    return result


def score_cell(value):
    return percentage(value) + ' (' + cp(value) + ' cp)'


def delta_cell(after, before, *, drag=False):
    delta = None if after is None or before is None else 100 * (before - after if drag else after - before)
    return score_points(delta, signed=True) + ' (' + cp_change(after, before) + ' cp)'


def delta_points(after, before, *, drag=False):
    delta = None if after is None or before is None else 100 * (before - after if drag else after - before)
    return score_points(delta, signed=True)


def headline_delta(after, before):
    """Headline deltas alone keep a centipawn-scale translation; other tables show percentages only."""
    value = centipawn_delta(after, before)
    return delta_points(after, before) + ('' if value is None else f' ({number(value, 0, signed=True)} cp)')


def interval_cell(bounds):
    return ' to '.join(score_points(v, signed=True) for v in bounds) if bounds else 'unavailable'


def gain_split(row):
    if row.get('database_move_gain_pp') is None:
        return 'unavailable'
    return (
        'Move ' + score_points(row['database_move_gain_pp'], signed=True) + '<br>'
        'Prep ' + score_points(row['continuation_gain_pp'], signed=True)
    )


def population_text(report):
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


def evidence_date(report):
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


def display_source(source, result_path):
    """Show the PGN relative to the report folder, so published reports carry no local absolute path."""
    try:
        return Path(os.path.relpath(source, report_directory(result_path))).as_posix()
    except ValueError:
        # Different Windows drives have no relative path.
        return Path(source).name


def opponent_rating(context):
    if not context or context.get('mean') is None:
        if context and context.get('reason') == 'no_preceding_opponent_move':
            return 'n/a (no preceding opponent move)'
        return 'unavailable'
    result = f"{context['mean']:,.0f}"
    if context['known_coverage'] < 1 - 1e-9:
        result += f" ({100 * context['known_coverage']:.1f}% rated)"
    return result


def rating_difference(context):
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


def gap_percentage(metrics, weighted=False):
    metrics = metrics or {}
    prefix = 'weighted_' if weighted else ''
    value = metrics.get(prefix + 'equivalent_gap_reach')
    if value is not None:
        return percentage(value)
    bounds = metrics.get(prefix + 'equivalent_gap_reach_bounds')
    return ' to '.join(map(percentage, bounds)) + ' (bounds)' if bounds is not None else 'unavailable'


def position_reach_label(scope):
    """Board reach combines every transposed arrival."""
    return 'Position reach' if scope.get('id', 'overall') == 'overall' else 'Position reach after chapter entry'


def move_reach_label(scope):
    """Move reach counts only arrivals at the parent that continue with this move."""
    return 'Move reach' if scope.get('id', 'overall') == 'overall' else 'Move reach after chapter entry'
