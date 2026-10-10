"""Load matching saved analyses for rendering, refusing companions from a different snapshot."""

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from .. import SCHEMA_VERSION
from ..schema import JsonObject
from ..sharpness import stopping_wdl
from ..sharpness import summarize as summarize_outcomes

Family = Literal['vulnerabilities', 'preparation', 'character', 'ratings', 'openings', 'insights', 'engine']
FAMILIES: tuple[Family, ...] = (
    'vulnerabilities',
    'preparation',
    'character',
    'ratings',
    'openings',
    'insights',
    'engine',
)
# Companions built from other companions, which must still match the files they were built from.
SUPPORTED: dict[str, set[str]] = {
    'insights': {'preparation', 'character', 'vulnerabilities', 'openings'},
    'engine': {'preparation', 'vulnerabilities'},
}


class Bundle(TypedDict):
    """One color's saved score and the companion analyses that match it, as loaded for rendering.

    `report` is the score JSON (see schema.ScoreResult) with display fields attached after loading.
    A companion is present only when it matches the score snapshot; otherwise `unavailable` says why.
    """

    path: Path
    report: JsonObject
    digest: str
    current_source: bool
    unavailable: dict[str, str]
    vulnerabilities: NotRequired[JsonObject]
    preparation: NotRequired[JsonObject]
    character: NotRequired[JsonObject]
    ratings: NotRequired[JsonObject]
    openings: NotRequired[JsonObject]
    insights: NotRequired[JsonObject]
    engine: NotRequired[JsonObject]


def load(paths: Iterable[str | Path], strict: bool = True, require_complete: bool = False) -> list[Bundle]:
    bundles: list[Bundle]
    seen: set[str]
    bundles, seen = [], set()
    for value in paths:
        path = Path(value)
        report = json.loads(path.read_text(encoding='utf-8'))
        color = report.get('color')
        if color not in ('white', 'black') or not all(
            k in report for k in ('overall', 'chapters', 'manifest', 'events')
        ):
            raise ValueError(f'Not a repertoire result file: {path}')
        if color in seen:
            raise ValueError('Supply one score report per color')
        if report['manifest'].get('schema_version') != SCHEMA_VERSION:
            raise ValueError(f'{path} was written by a different program version; rebuild it')
        seen.add(color)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = report['manifest']
        source = Path(manifest['input_path'])
        current = source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() == manifest['input_sha256']
        bundle: Bundle = dict(path=path, report=report, digest=digest, current_source=current, unavailable={})
        for family in FAMILIES:
            companion = path.with_suffix(f'.{family}.json')
            reason: str | None = None
            if not companion.exists():
                reason = 'not generated'
            else:
                data = json.loads(companion.read_text(encoding='utf-8'))
                m = data.get('manifest', {})
                if data.get('color') != color or any(
                    m.get(key) != expected
                    for key, expected in (
                        ('report_sha256', digest),
                        ('input_sha256', manifest['input_sha256']),
                        ('filters', manifest['filters']),
                    )
                ):
                    reason = 'belongs to a different score snapshot'
                    if strict:
                        raise ValueError(f'{companion}: {reason}; regenerate this analysis before combining')
                elif m.get('schema_version') != SCHEMA_VERSION:
                    reason = 'written by a different program version'
                    if strict:
                        raise ValueError(f'{companion}: {reason}; rebuild the analyses')
                else:
                    if family == 'ratings':
                        for supporting, expected_hash in m.get('supporting_sha256', {}).items():
                            supporting_path = path.with_suffix(f'.{supporting}.json')
                            if (
                                not supporting_path.exists()
                                or hashlib.sha256(supporting_path.read_bytes()).hexdigest() != expected_hash
                                or supporting not in bundle
                            ):
                                reason = 'supporting analysis changed since rating generation'
                                if strict:
                                    raise ValueError(f'{companion}: {reason}; regenerate ratings before combining')
                                break
                        if set(m.get('supporting_sha256', {})) != {'preparation', 'character', 'vulnerabilities'}:
                            reason = 'missing rating provenance'
                            if strict:
                                raise ValueError(f'{companion}: {reason}')
                    if family in SUPPORTED:
                        if set(m.get('supporting_sha256', {})) != SUPPORTED[family]:
                            reason = f'missing {family} provenance'
                        elif any(
                            name not in bundle
                            or not path.with_suffix(f'.{name}.json').exists()
                            or hashlib.sha256(path.with_suffix(f'.{name}.json').read_bytes()).hexdigest() != digest
                            for name, digest in m['supporting_sha256'].items()
                        ):
                            reason = f'supporting analysis changed since the {family} analysis'
                        if reason and strict:
                            raise ValueError(f'{companion}: {reason}; regenerate the {family} analysis')
                    if reason is None:
                        bundle[family] = data
            if reason:
                bundle['unavailable'][family] = reason
        if require_complete and bundle['unavailable']:
            raise ValueError(f'{color.title()} analyses incomplete: {bundle["unavailable"]}')
        if 'ratings' in bundle:
            from ..ratings import attach

            attach(bundle)
        gap_scopes = scope_by_id(bundle.get('character'))
        report['gap_coverage'] = gap_scopes.get('overall', {}).get('gap_coverage')
        report['overall']['outcomes'] = gap_scopes.get('overall', {}).get('outcomes')
        for chapter in report['chapters']:
            chapter['gap_coverage'] = gap_scopes.get(chapter['id'], {}).get('gap_coverage')
            chapter['score']['outcomes'] = gap_scopes.get(chapter['id'], {}).get('outcomes')
        attach_outcomes(bundle)
        attach_insights(bundle)
        attach_engine(bundle)
        bundles.append(bundle)
    return sorted(bundles, key=lambda b: b['report']['color'] != 'white')


def load_correlations(
    bundles: Sequence[Bundle], strict: bool = True, require_complete: bool = False
) -> tuple[JsonObject | None, str | None]:
    path = bundles[0]['path'].parent / 'prepared-depth-gain-correlation.json'
    if not path.exists():
        if require_complete:
            raise ValueError('Generate future preparation gain correlations before combining the reports')
        return None, 'not generated'
    result = json.loads(path.read_text(encoding='utf-8'))
    expected = {b['report']['color']: b for b in bundles}
    provenance = result.get('provenance', {})
    matches = result.get('schema_version') == SCHEMA_VERSION and set(result.get('results', {})) == set(expected)
    matches &= set(provenance) == set(expected) and all(
        p.get('report_sha256') == expected[c]['digest']
        and p.get('input_sha256') == expected[c]['report']['manifest']['input_sha256']
        and p.get('filters') == expected[c]['report']['manifest']['filters']
        and p.get('prior') == expected[c]['report']['manifest']['prior']
        and p.get('sparse_threshold') == expected[c]['report']['manifest']['sparse_threshold']
        and expected[c]['path'].with_suffix('.vulnerabilities.json').exists()
        and p.get('vulnerabilities_sha256')
        == hashlib.sha256(expected[c]['path'].with_suffix('.vulnerabilities.json').read_bytes()).hexdigest()
        for c, p in provenance.items()
    )
    if not matches:
        if strict:
            raise ValueError(
                f'{path}: correlations belong to different score snapshots; regenerate them before combining'
            )
        return None, 'belongs to different score snapshots'
    return result, None


def load_rating_correlations(bundles: Sequence[Bundle], strict: bool = True) -> tuple[JsonObject | None, str | None]:
    path = bundles[0]['path'].parent / 'opponent-rating-score-correlation.json'
    if not path.exists():
        return None, 'not generated'
    result = json.loads(path.read_text(encoding='utf-8'))
    expected = {b['report']['color']: b for b in bundles}
    matches = set(result.get('results', {})) == set(expected)
    for color, bundle in expected.items():
        provenance = result.get('results', {}).get(color, {}).get('provenance', {})
        hashes = provenance.get('hashes', {})
        matches &= (
            hashes.get('score') == bundle['digest']
            and provenance.get('input_sha256') == bundle['report']['manifest']['input_sha256']
            and provenance.get('filters') == bundle['report']['manifest']['filters']
            and provenance.get('sparse_threshold') == bundle['report']['manifest']['sparse_threshold']
        )
        for family in ('ratings', 'vulnerabilities'):
            companion = bundle['path'].with_suffix(f'.{family}.json')
            matches &= (
                family in bundle
                and companion.exists()
                and hashes.get(family) == hashlib.sha256(companion.read_bytes()).hexdigest()
            )
    if not matches:
        if strict:
            raise ValueError(
                f'{path}: rating correlations belong to different analysis snapshots; regenerate them before combining'
            )
        return None, 'belongs to different analysis snapshots'
    return result, None


def attach_outcomes(bundle: Bundle) -> None:
    """Join saved recursive position WDL to comparison rows without new queries."""
    character = scope_by_id(bundle.get('character'))
    vulnerabilities = bundle.get('vulnerabilities', {})
    scopes = [vulnerabilities.get('overall', {}), *vulnerabilities.get('chapters', [])]
    color = bundle['report']['color'] == 'white'
    for scope in scopes:
        positions = {r['position']: r for r in character.get(scope.get('id', 'overall'), {}).get('positions', [])}
        rows = list(scope.get('all_signed_rows', [])) + list(scope.get('strengths', []))
        for ranking in scope.get('rankings', {}).values():
            rows.extend(ranking)
        for row in rows:
            row['reference_outcomes'] = (
                positions.get(row['position'], {}).get('outcomes') if row['kind'] == 'opponent' else None
            )
            if row['prepared']:
                row['move_outcomes'] = positions.get(row['target'], {}).get('outcomes')
            else:
                fixed = row['move_score'] if row.get('score_basis') == 'terminal result' else None
                row['move_outcomes'] = summarize_outcomes(stopping_wdl(row['counts_white_draw_black'], color, fixed))


def attach_insights(bundle: Bundle) -> None:
    insights = scope_by_id(bundle.get('insights'))
    characters = scope_by_id(bundle.get('character'))
    bundle['report']['overall']['branch_score_spread'] = insights.get('overall', {}).get('branch_score_spread')
    for chapter in bundle['report']['chapters']:
        chapter['score']['branch_score_spread'] = insights.get(chapter['id'], {}).get('branch_score_spread')
        chapter['entry_opening_sources'] = insights.get(chapter['id'], {}).get('entry_opening_sources', {})
    for sid, scope in characters.items():
        if sid in insights:
            scope['gap_priorities'] = insights[sid]['gaps']
            scope['branch_score_spread'] = insights[sid]['branch_score_spread']
            for row in scope.get('positions', []):
                row['branch_score_spread'] = insights[sid].get('position_spreads', {}).get(row['position'])
    for row in bundle.get('openings', {}).get('openings', []):
        added = bundle.get('insights', {}).get('opening_spreads', {}).get(row['id'], {})
        row['branch_score_spread'] = added.get('branch_score_spread')
        for entry in row['entries']:
            entry['branch_score_spread'] = added.get('entries', {}).get(entry['position'])
    for scope in [
        bundle.get('vulnerabilities', {}).get('overall', {}),
        *bundle.get('vulnerabilities', {}).get('chapters', []),
    ]:
        added = insights.get(scope.get('id', 'overall'), {}).get('moves', {})
        rows = list(scope.get('all_signed_rows', [])) + list(scope.get('strengths', []))
        for values in scope.get('rankings', {}).values():
            rows.extend(values)
        for row in rows:
            row.update(added.get(row['id'], {}))


def attach_engine(bundle: Bundle) -> None:
    """Each move row gets its engine evaluations before and after, keyed like the vulnerability rows."""
    engine = bundle.get('engine')
    if not engine:
        return
    moves, transpositions = engine.get('moves', {}), engine.get('transpositions', {})
    vulnerabilities = bundle.get('vulnerabilities', {})
    for scope in [vulnerabilities.get('overall', {}), *vulnerabilities.get('chapters', [])]:
        rows = list(scope.get('all_signed_rows', [])) + list(scope.get('strengths', []))
        for values in scope.get('rankings', {}).values():
            rows.extend(values)
        for row in rows:
            row['engine'] = moves.get(row['id'])
        for row in scope.get('free_transpositions', []):
            row['engine'] = transpositions.get(row['id'])


def scope_by_id(data: JsonObject | None, key: str = 'scopes') -> dict[str, JsonObject]:
    return {s['id']: s for s in data.get(key, [])} if data else {}
