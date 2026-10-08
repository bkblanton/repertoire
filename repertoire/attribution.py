"""Chapter provenance for displayed moves and positions."""

import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict

from .board_cache import children
from .graph import Graph
from .schema import JsonObject, Position

ATTRIBUTION_NOTE = (
    'Chapter attribution follows the exact position or final recorded '
    'move, so a representative route can combine chapters. '
    'Shared sources list every contributing chapter. Unprepared replies name their parent context; '
    'transpositions name the chapters containing the reached position.'
)


class ChapterAttribution(TypedDict):
    source_ids: list[str]
    context_ids: list[str]
    transposition_ids: list[str]
    basis: str


def write_text(path: str | Path, text: str) -> None:
    """Replace complete outputs atomically, tolerating brief Windows sync locks."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent, prefix='.chapter-attribution-', suffix='.tmp', delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except OSError:
                if attempt == 7:
                    raise
                time.sleep(min(0.1 * 2**attempt, 1.0))
    finally:
        temporary.unlink(missing_ok=True)


class Attribution:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.catalog: list[JsonObject] = [{k: c.get(k) for k in ('id', 'name', 'url')} for c in graph.chapters]
        self.order = {c['id']: i for i, c in enumerate(graph.chapters)}
        self.cache: dict[tuple[Position, str | None], ChapterAttribution] = {}

    def ordered(self, ids: Iterable[str]) -> list[str]:
        return sorted(set(ids), key=lambda cid: (self.order.get(cid, len(self.order)), cid))

    def position_or_move(self, position: str, move: str | None = None) -> ChapterAttribution:
        position = ' '.join(position.split()[:4])
        identity = position, move
        if identity in self.cache:
            return self.cache[identity]
        node = self.graph.nodes.get(position)
        context = self.ordered(node.chapters) if node else []
        sources: list[str]
        transpositions: list[str]
        sources, transpositions = [], []
        if move is None:
            sources, basis = context, 'position'
        elif node and node.provenance.get(move):
            sources, basis = self.ordered(node.provenance[move]), 'recorded move'
        else:
            basis = 'unprepared move'
            if node:
                target = self.graph.nodes.get(children(position)[move])
                if target:
                    transpositions, basis = self.ordered(target.chapters), 'transposition'
        result: ChapterAttribution = dict(
            source_ids=sources, context_ids=context, transposition_ids=transpositions, basis=basis
        )
        self.cache[identity] = result
        return result


def enrich(report: JsonObject, graph: Graph) -> JsonObject:
    """Add provenance only; leave every score, probability and ordering unchanged."""
    attribution = Attribution(graph)
    report['chapter_catalog'] = attribution.catalog
    if 'events' in report:
        for row in report['events']:
            row['chapter_attribution'] = attribution.position_or_move(row['parent_position'], row.get('move'))
        for chapter in report['chapters']:
            for entry in chapter['entries']:
                entry['chapter_attribution'] = attribution.position_or_move(entry['position'])
    elif 'scopes' in report:
        for scope in report['scopes']:
            for row in scope.get('stops', []):
                row['chapter_attribution'] = attribution.position_or_move(row['parent_position'], row.get('move'))
            for row in scope.get('entry_routes', {}).get('positions', []):
                row['chapter_attribution'] = attribution.position_or_move(row['position'])
            if 'reuse' not in scope:
                continue
            for row in scope.get('positions', []):
                origins = row.get('unprepared_origins', [])
                if origins:
                    contexts = [attribution.position_or_move(o['parent_position'], o['move']) for o in origins]
                    row['chapter_attribution'] = dict(
                        source_ids=[],
                        transposition_ids=[],
                        context_ids=attribution.ordered(cid for context in contexts for cid in context['context_ids']),
                        basis='unprepared move',
                    )
                else:
                    row['chapter_attribution'] = attribution.position_or_move(row['position'])
            for row in scope['reuse']['decisions']:
                row['chapter_attribution'] = attribution.position_or_move(row['position'], row['move'])
            for row in scope['predictability']['positions']:
                row['chapter_attribution'] = attribution.position_or_move(row['position'])
                for reply in row['replies']:
                    reply['chapter_attribution'] = attribution.position_or_move(row['position'], reply['move'])
            examples = {}
            for row in scope['stopping_outcomes']:
                row['chapter_attribution'] = attribution.position_or_move(row['parent_position'], row.get('move'))
                examples[row['position'], row['line']] = row['chapter_attribution']
            for profile in scope['position_profiles'].values():
                for distribution in profile['distributions'].values():
                    for row in distribution:
                        row['example_chapter_attribution'] = examples[row['example_position'], row['example_line']]
    else:
        # Rankings reuse the row objects of all_signed_rows; visit each object once.
        seen: set[int] = set()

        def visit(item: Any) -> None:
            if id(item) in seen:
                return
            seen.add(id(item))
            if isinstance(item, list):
                for child in item:
                    visit(child)
            elif isinstance(item, dict):
                if item.get('kind') in ('own', 'opponent') and 'position' in item and 'move' in item:
                    item['chapter_attribution'] = attribution.position_or_move(item['position'], item['move'])
                    if item.get('alternative'):
                        alt = item['alternative']
                        alt['chapter_attribution'] = attribution.position_or_move(item['position'], alt['move'])
                for field, child in list(item.items()):
                    if field != 'chapter_attribution':
                        visit(child)

        visit(report.get('overall', {}))
        visit(report.get('chapters', []))
    return report


def chapter_text(row: Mapping[str, Any], catalog: Iterable[Mapping[str, Any]]) -> str:
    attribution = row.get('chapter_attribution')
    if attribution is None:
        return 'Not attributed'
    lookup = {c['id']: c for c in catalog}

    def names(ids: Iterable[str]) -> str:
        result = []
        for cid in ids:
            chapter = lookup.get(cid, {'name': cid})
            label = (
                chapter['name'].replace('|', '&#124;').replace('[', '&#91;').replace(']', '&#93;').replace('\n', ' ')
            )
            result.append(f"[{label}]({chapter['url']})" if chapter.get('url') else label)
        return '; '.join(result) or 'None'

    if attribution['source_ids']:
        return names(attribution['source_ids'])
    if attribution['transposition_ids']:
        return 'Transposition into: ' + names(attribution['transposition_ids'])
    if attribution['basis'] == 'unprepared move':
        return 'Unprepared' + ('; parent: ' + names(attribution['context_ids']) if attribution['context_ids'] else '')
    return 'None'
