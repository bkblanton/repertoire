"""Markdown building blocks: tables, sections, glossary links and the table of contents."""

import html
import re
from collections.abc import Callable, Container, Iterable, Sequence
from typing import Self, cast

# One table cell as passed in; `table` stringifies each, and SourceCell is a str carrying extra context.
Cell = str | int | float


def about(key: str, text: str = 'Definitions') -> str:
    """Caveats live once in the glossary; tables link to the relevant entry."""
    return f'[{text}](#def-{key})'


class SourceCell(str):
    """Chapter links that carry the opening name shown with them, and an optional further note."""

    opening: str
    note: str

    def __new__(cls, text: str, opening: str, note: str = '') -> Self:
        result = super().__new__(cls, text)
        result.opening = opening
        result.note = note
        return result

    def context(self) -> str:
        """The line under a table's board label: the opening name, then the chapters."""
        parts = [part for part in (self.opening, str(self)) if part and part not in ('unavailable', 'None')]
        return '<br>'.join(([' · '.join(parts)] if parts else []) + ([self.note] if self.note else []))


def compressed_columns(headers: Sequence[str], rows: Iterable[Sequence[Cell]]) -> tuple[list[str], list[list[Cell]]]:
    """Keep related values together without discarding any table evidence."""
    headers, rows = list(headers), cast('list[list[Cell]]', [list(row) for row in rows])
    if any(len(row) != len(headers) for row in rows):
        raise ValueError('Table header and row lengths differ')

    def combine(primary: str, secondary: str, title: str, format_cell: Callable[[Cell, Cell], Cell]) -> None:
        if primary not in headers or secondary not in headers:
            return
        first, second = headers.index(primary), headers.index(secondary)
        headers[first] = title
        for row in rows:
            row[first] = format_cell(row[first], row[second])
            del row[second]
        del headers[second]

    def rating(value: Cell, difference: Cell) -> Cell:
        if str(difference).startswith(('n/a', 'unavailable')):
            return value
        number, _, notes = str(difference).partition(' (')
        return f'{value}<br>{number} vs parent' + (f' ({notes}' if notes else '')

    combine('Opening', 'ECO', 'Opening / ECO', lambda opening, eco: f'{opening}<br>{eco}')
    combine('Opponent rating', 'Rating Δ vs parent', 'Opponent rating', rating)
    return headers, rows


# Headers of columns that hold words rather than numbers, so they stay left-aligned.
TEXT_COLUMNS = (
    'Line',
    'Position',
    'Chapter',
    'Source',
    'Your move',
    'Option',
    'Verdict',
    'Color',
    'Common reply',
    'Observed alternative',
    'Opening',
    'Most common opening',
    'ECO',
    'Current names',
    'Name source',
    'Evidence source',
    'Feature',
    'Stopping type',
    'Measure',
    'Destination',
    'Frequent pawn',
    'Preparation ends',
    'Most common unprepared',
    'Most common reply',
    'Example route',
    'Comparison',
    'Part',
    'Arrivals',
)

# Numeric headers that share a prefix with a text column.
NUMBER_COLUMNS = ('Position reach', 'Chapter reach')


def table(headers: Sequence[str], rows: Sequence[Sequence[Cell]]) -> list[str]:
    """A Markdown table; chapter and opening context moves under the board label in the column before it."""
    if not rows:
        return []
    headers, rows = compressed_columns(headers, rows)
    sources = [
        i
        for i in range(1, len(headers))
        if headers[i] != 'Chapters' and any(isinstance(row[i], SourceCell) for row in rows)
    ]
    for i in reversed(sources):
        headers = headers[:i] + headers[i + 1 :]
        rows = [[*row[: i - 1], _with_context(row[i - 1], row[i]), *row[i + 1 :]] for row in rows]
    return [
        '| ' + ' | '.join(headers) + ' |',
        '| '
        + ' | '.join(
            '---' if i == 0 or (h.startswith(TEXT_COLUMNS) and not h.startswith(NUMBER_COLUMNS)) else '---:'
            for i, h in enumerate(headers)
        )
        + ' |',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
        '',
    ]


def _with_context(label: Cell, source: Cell) -> str:
    context = source.context() if isinstance(source, SourceCell) else str(source)
    return f'{label}<br>{context}' if context and context != 'None' else str(label)


def drop_uniform(
    headers: Sequence[str], rows: Sequence[Sequence[Cell]], optional: Container[str]
) -> tuple[list[str], list[list[Cell]]]:
    """Remove optional columns that carry the same value in every row, such as all-zero endings."""
    keep = [
        i for i, h in enumerate(headers) if h not in optional or len(rows) < 2 or len({str(row[i]) for row in rows}) > 1
    ]
    return [headers[i] for i in keep], [[row[i] for i in keep] for row in rows]


def section(title: str, anchor: str | None = None) -> list[str]:
    return ([f'<a id="{anchor}"></a>', ''] if anchor else []) + [title, '']


def report_navigation(text: Sequence[str]) -> list[str]:
    """Build a nested contents list from actual second- and third-level headings."""
    existing = set(re.findall(r'<a id="([^"]+)"', '\n'.join(text)))
    body: list[str]
    headings: list[tuple[str, str, str, str | None]]
    parent: str | None
    body, headings, parent = [], [], None
    for item in text:
        heading = re.fullmatch(r'(##|###) (.+)', item)
        if heading:
            level, title = heading.groups()
            previous = next((line for line in reversed(body) if line.strip()), '')
            explicit = re.fullmatch(r'<a id="([^"]+)"></a>', previous)
            if explicit:
                anchor = explicit[1]
            else:
                slug = re.sub(r'[^a-z0-9]+', '-', html.unescape(title).lower()).strip('-')
                stem = f'{parent}-{slug}' if level == '###' and parent else slug
                anchor, suffix = stem, 2
                while anchor in existing:
                    anchor, suffix = f'{stem}-{suffix}', suffix + 1
                existing.add(anchor)
                body += [f'<a id="{anchor}"></a>', '']
            if level == '##':
                parent = anchor
            headings.append((level, title, anchor, parent))
        body.append(item)
    navigation = ['**Table of contents**', '', '- [Summary](summary.md)']
    for level, title, anchor, parent in headings:
        label = 'Chapter-by-chapter table' if anchor in ('white-chapters', 'black-chapters') else title
        indent = '    ' if level == '###' and parent else ''
        navigation.append(f'{indent}- [{label}](#{anchor})')
    navigation.append('')
    index = body.index('<!-- report-navigation -->')
    body[index : index + 1] = navigation
    return body
