"""Render every page from matching saved results and resolve links between pages."""

import os
import re
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

from ..attribution import write_text
from ..layout import report_directory
from ..schema import JsonObject
from .bundle import FAMILIES, Bundle, load, load_correlations, load_rating_correlations
from .pages import full_report, summary_report

CHAPTER_PAGE = re.compile(r'[WB]\d+\.md')


def resolve_links(pages: Mapping[Path, str], aliases: Mapping[str, Path]) -> dict[Path, str]:
    """Point in-page anchors and @page placeholders at whichever generated file holds them."""
    owners: dict[str, Path] = {}
    homes: dict[Path, str] = {}
    for path, text in pages.items():
        found = re.findall(r'<a id="([^"]+)"></a>', text)
        for anchor in found:
            owners.setdefault(anchor, path)
        if path.parent.name == 'chapters' and found:
            homes[path] = found[0]

    def relative(target: Path, source: Path) -> str:
        return Path(os.path.relpath(target, source.parent)).as_posix()

    def anchor_link(path: Path, match: re.Match[str]) -> str:
        anchor = match[1]
        target = owners.get(anchor)
        if target is None or target == path:
            return match[0]
        return (
            f']({relative(target, path)})' if homes.get(target) == anchor else f']({relative(target, path)}#{anchor})'
        )

    def page_link(path: Path, match: re.Match[str]) -> str:
        target = aliases.get(match[1])
        return match[0] if target is None else f']({relative(target, path)}{match[2] or ""})'

    resolved: dict[Path, str] = {}
    for path, text in pages.items():
        text = re.sub(r'\]\(#([^)\s]+)\)', partial(anchor_link, path), text)
        resolved[path] = re.sub(r'\]\(@([\w/.-]+)(#[^)\s]*)?\)', partial(page_link, path), text)
    return resolved


def generate(
    paths: Sequence[str | Path],
    full_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    *,
    strict: bool = True,
    require_complete: bool = False,
    top: int = 10,
    chapter_top: int = 5,
    position_top: int = 20,
) -> list[Bundle]:
    if not paths or min(top, chapter_top, position_top) < 1:
        raise ValueError('Supply reports and positive table lengths')
    bundles = load(paths, strict, require_complete)
    correlations, reason = load_correlations(bundles, strict, require_complete)
    rating_correlations, rating_reason = load_rating_correlations(bundles, strict)
    full_path = Path(full_path) if full_path else report_directory(bundles[0]['path']) / 'report.md'
    summary_path = Path(summary_path) if summary_path else full_path.parent / 'summary.md'
    if full_path.resolve() == summary_path.resolve():
        raise ValueError('Report and summary must be different files')
    if any(p.suffix.lower() != '.md' for p in (full_path, summary_path)):
        raise ValueError('Report and summary destinations must be Markdown (.md) files')
    rendered = full_report(
        bundles, correlations, reason, top, chapter_top, position_top, rating_correlations, rating_reason
    )
    folder = full_path.parent
    pages = {
        full_path: rendered.pop('report.md'),
        summary_path: summary_report(bundles) + '\n',
        **{folder / name: content for name, content in rendered.items()},
    }
    aliases = {'report': full_path, 'summary': summary_path, **{name[:-3]: folder / name for name in rendered}}
    # Comparison pages are written by `repertoire compare` beside the report.
    aliases.update({f'comparisons/{p.stem}': p for p in sorted((folder / 'comparisons').glob('*.md'))})
    pages[full_path] = pages[full_path].replace('](summary.md)', '](@summary)')
    # Use relative paths even when callers place the outputs in different folders.
    for b in bundles:
        for data in [b['path']] + [b['path'].with_suffix(f'.{family}.json') for family in FAMILIES if family in b]:
            pages[full_path] = pages[full_path].replace(
                f']({data.name})', f']({Path(os.path.relpath(data, folder)).as_posix()})'
            )
    pages = resolve_links(pages, aliases)
    for target, content in pages.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text(target, content)
    # Remove pages for chapters or colors that no longer exist; only generator-owned names are touched.
    for directory, pattern in (
        (folder / 'chapters', CHAPTER_PAGE),
        (folder / 'openings', re.compile(r'(white|black)\.md')),
    ):
        if directory.is_dir():
            for stale in directory.iterdir():
                if pattern.fullmatch(stale.name) and stale not in pages:
                    stale.unlink()
    return bundles


def page_names(report: JsonObject) -> list[str]:
    """Relative chapter and opening page paths that a render of this score result writes."""
    color = report['color']
    return [f'chapters/{color[0].upper()}{i}.md' for i in range(1, len(report['chapters']) + 1)] + [
        f'openings/{color}.md'
    ]
