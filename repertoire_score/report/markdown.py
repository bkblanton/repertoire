"""Markdown building blocks: tables, sections, glossary links and the table of contents."""

import html
import re


def about(key, text='Definitions'):
    """Caveats live once in the glossary; tables link to the relevant entry."""
    return f'[{text}](#def-{key})'


class SourceCell(str):
    """Chapter links carry their neighboring opening cell through table rendering."""

    def __new__(cls, text, opening):
        result = super().__new__(cls, text)
        result.opening = opening
        return result


def compressed_columns(headers, rows):
    """Keep related values together without discarding any table evidence."""
    headers, rows = list(headers), [list(row) for row in rows]
    if any(len(row) != len(headers) for row in rows):
        raise ValueError('Table header and row lengths differ')

    def combine(primary, secondary, title, format_cell):
        if primary not in headers or secondary not in headers:
            return
        first, second = headers.index(primary), headers.index(secondary)
        headers[first] = title
        for row in rows:
            row[first] = format_cell(row[first], row[second])
            del row[second]
        del headers[second]

    combine('Opening', 'ECO', 'Opening / ECO', lambda opening, eco: f'{opening}<br>{eco}')
    for reach in (
        'Position reach',
        'Position reach after chapter entry',
        'Move reach',
        'Move reach after chapter entry',
        'Games leaving prep here',
        'Games leaving prep here after chapter entry',
    ):
        combine(
            reach,
            'Avg games per encounter',
            reach + '<br>Avg games per encounter',
            lambda probability, games: f'{probability}<br>1 per {games} games',
        )
    combine(
        'Avg opponent rating',
        'Rating Δ vs parent',
        'Avg opponent rating<br>Rating Δ vs parent',
        lambda rating, difference: f'{rating}<br>Δ {difference}',
    )
    return headers, rows


def table(headers, rows):
    if not rows:
        return []
    headers, rows = compressed_columns(headers, rows)
    if 'Most common opening source' not in headers:
        source_columns = [
            i
            for i in range(len(headers))
            if any(isinstance(row[i], SourceCell) for row in rows) and headers[i] != 'Chapters'
        ]
        for i in reversed(source_columns):
            headers = [*headers[: i + 1], 'Most common opening source', *headers[i + 1 :]]
            rows = [
                [*row[: i + 1], row[i].opening if isinstance(row[i], SourceCell) else 'unavailable', *row[i + 1 :]]
                for row in rows
            ]
    text_columns = (
        'Line',
        'Position',
        'Chapter',
        'Source',
        'Your move',
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
    )
    return [
        '| ' + ' | '.join(headers) + ' |',
        '| '
        + ' | '.join(
            '---' if i == 0 or (h.startswith(text_columns) and not h.startswith('Position reach')) else '---:'
            for i, h in enumerate(headers)
        )
        + ' |',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
        '',
    ]


def drop_uniform(headers, rows, optional):
    """Remove optional columns that carry the same value in every row, such as all-zero endings."""
    keep = [
        i for i, h in enumerate(headers) if h not in optional or len(rows) < 2 or len({str(row[i]) for row in rows}) > 1
    ]
    return [headers[i] for i in keep], [[row[i] for i in keep] for row in rows]


def section(title, anchor=None):
    return ([f'<a id="{anchor}"></a>', ''] if anchor else []) + [title, '']


def report_navigation(text):
    """Build a nested contents list from actual second- and third-level headings."""
    existing = set(re.findall(r'<a id="([^"]+)"', '\n'.join(text)))
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
