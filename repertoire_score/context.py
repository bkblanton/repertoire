"""Shared loading and saving for the cache-only analysis stages.

Each stage reads a saved score, checks that its source PGN and cached evidence still match, rebuilds the
repertoire graph, and writes a companion JSON whose manifest ties it to that exact snapshot.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from . import SCHEMA_VERSION
from .explorer import CacheMiss, Explorer
from .graph import parse
from .layout import data_json
from .schema import CompanionManifest

DEFAULT_CACHE = '.cache/explorer'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    return sha256(Path(path).read_bytes())


class AnalysisContext:
    """A saved score result, the companion analyses it needs, and the cached evidence behind it."""

    def __init__(self, path, companions=()):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.report_sha256 = sha256(raw)
        self.saved = json.loads(raw)
        self.manifest = self.saved['manifest']
        self.source = Path(self.manifest['input_path'])
        self.color = self.saved['color'] == 'white'
        self.require_source()
        self.companions, self.companion_hashes = {}, {}
        for family in companions:
            data = self.path.with_suffix(f'.{family}.json').read_bytes()
            value = json.loads(data)
            if not self.matches(value):
                raise ValueError(f'{family} belongs to a different score snapshot; regenerate it first')
            self.companions[family], self.companion_hashes[family] = value, sha256(data)
        self.evidence, self.missing, self.provenance = {}, [], {}
        self._graph = None

    @property
    def policy(self):
        return self.manifest['configuration'].get('policy', {})

    @property
    def roots(self):
        return self.manifest['root_weights']

    @property
    def sparse_threshold(self):
        return self.manifest['sparse_threshold']

    @property
    def graph(self):
        if self._graph is None:
            self._graph = parse(self.source, self.manifest['configuration'].get('exclude', []))
        return self._graph

    def matches(self, companion):
        """Whether a companion analysis was made from this exact score result."""
        m = companion.get('manifest', {})
        return (
            companion.get('color') == self.saved['color']
            and m.get('report_sha256') == self.report_sha256
            and m.get('input_sha256') == self.manifest['input_sha256']
            and m.get('filters') == self.manifest['filters']
        )

    def require_source(self, during=None):
        if file_sha256(self.source) != self.manifest['input_sha256']:
            raise ValueError(f'PGN changed during {during}' if during else 'PGN changed since scoring; rescore first')

    def read_evidence(self, cache, positions=None, required=False):
        """Read cached tables offline; tables the score used must be unchanged.

        Misses are recorded in `missing`, or raise CacheMiss when `required`.
        """
        saved = self.manifest['evidence']
        explorer = Explorer(cache, self.manifest['filters'], offline=True)
        try:
            for k in self.graph.nodes if positions is None else positions:
                if k in self.evidence or k in self.missing:
                    continue
                try:
                    self.evidence[k] = explorer.get(k)
                except CacheMiss:
                    if required:
                        raise
                    self.missing.append(k)
                if k in saved and explorer.provenance.get(k) != saved[k]:
                    raise ValueError('Cached evidence changed since scoring; rescore first')
        finally:
            explorer.close()
        self.provenance.update(explorer.provenance)
        return self.evidence

    def scopes(self):
        """The overall repertoire and every chapter, each with its first-entry starting weights."""
        result = [
            dict(
                id='overall',
                name='Overall repertoire',
                starts=self.roots,
                policy_basis='overall policy',
                chapter=None,
                entry_probability=1.0,
                overall_policy_entry_probability=1.0,
                score=self.saved['overall'],
            )
        ]
        for c in self.saved['chapters']:
            weights = c['score'].get('first_entry_weights', {})
            if not weights and len(c['entries']) == 1:
                weights = {c['entries'][0]['position']: 1.0}
            result.append(
                dict(
                    id=c['id'],
                    name=c['name'],
                    starts={k: w for k, w in weights.items() if w},
                    policy_basis=c.get('policy_basis', 'overall policy'),
                    chapter=c['id'],
                    entry_probability=c['score'].get('entry_probability'),
                    overall_policy_entry_probability=c['score'].get('overall_policy_entry_probability'),
                    score=c['score'],
                )
            )
        return result

    def companion_manifest(self, **fields) -> CompanionManifest:
        """Provenance shared by every companion: the score snapshot, its source and the cache reads."""
        manifest = dict(
            created_at=datetime.now(UTC).isoformat(),
            schema_version=SCHEMA_VERSION,
            report_path=str(self.path.resolve()),
            report_sha256=self.report_sha256,
            input_path=str(self.source),
            input_sha256=self.manifest['input_sha256'],
            filters=self.manifest['filters'],
            cache_only=True,
            network_requests=0,
            evidence=self.provenance,
            uncached_positions=self.missing,
        )
        manifest.update(fields)
        return manifest


def write_companion(path, family, value):
    Path(path).with_suffix(f'.{family}.json').write_text(data_json(value), encoding='utf-8')


def stage_main(family, analyze, description, configure=None, options=None):
    """Command-line entry point shared by the cache-only stages.

    `configure(parser)` adds stage options; `options(parser, args)` validates them and returns extra
    keyword arguments for `analyze(path, cache, **kwargs)`.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('reports', nargs='+', type=Path, help='Saved score JSON files, with unchanged source PGNs')
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    if configure:
        configure(parser)
    args = parser.parse_args()
    kwargs = options(parser, args) if options else {}
    try:
        for path in args.reports:
            value = analyze(path, args.cache, **kwargs)
            write_companion(path, family, value)
            print(f"{value['color']}: {family} analysis saved", flush=True)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'{family.title()} analysis failed: {exc}\n')
    from .render import update_report_outputs

    update_report_outputs(args.reports[-1])
