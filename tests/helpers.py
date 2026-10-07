"""Fixtures shared by several test modules: small PGN graphs, Explorer tables and cached runs."""

import hashlib
import json
import re
from types import SimpleNamespace

import chess
import httpx
import pytest

from repertoire import score as cli
from repertoire.evaluate import backward
from repertoire.explorer import DEFAULT_FILTERS, ENDPOINT
from repertoire.graph import key, parse, resolve, topology
from repertoire.model import empirical, prepare
from repertoire.report.format import centipawn_delta
from repertoire.uncertainty import Posterior


def graph(tmp_path, text):
    path = tmp_path / 'fixture.pgn'
    path.write_text(text, encoding='utf-8')
    return parse(path)


def position(moves):
    b = chess.Board()
    for m in moves.split():
        b.push_san(m)
    return key(b)


def data(w, d, b, moves=()):
    return dict(white=w, draws=d, black=b, moves=[dict(uci=m, white=a, draws=c, black=e) for m, a, c, e in moves])


def setup(g, color, evidence, policy=None):
    """Model, order, empirical sample, Posterior, empirical values and posterior-mean values."""
    t = resolve(g, color, policy or {})
    order = topology(t, g.roots)
    model = prepare(g, t, order, color, evidence)
    raw = empirical(model, color)
    values = backward(model, order, raw, 30)
    posterior = Posterior(model, order, color, [0.5] * 3, 30)
    return model, order, raw, posterior, values, posterior.values


def cache_row(cache, k, data):
    query = dict(
        DEFAULT_FILTERS, fen=k + ' 0 1', moves=chess.Board(k + ' 0 1').legal_moves.count(), topGames=0, recentGames=0
    )
    identity = {'endpoint': ENDPOINT, 'query': query}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (cache / (digest + '.json')).write_text(json.dumps(dict(identity=identity, retrieved_at='2026-09-28', data=data)))
    return dict(cache_key=digest, retrieved_at='2026-09-28')


ADVANCE = (
    '[ChapterURL "https://lichess.org/study/test/advance"]\n[ChapterName '
    '"Advance"]\n\n1. e4 e6 2. d4 d5 3. e5 c5 4. c3 *'
)


TARRASCH = (
    '[ChapterURL "https://lichess.org/study/test/tarrasch"]\n[ChapterName '
    '"Tarrasch Nf6"]\n\n1. e4 e6 2. d4 d5 3. Nd2 Nf6 4. e5 *'
)


SPLIT = (
    '[ChapterURL "https://lichess.org/study/test/split"]\n[ChapterName '
    '"Tarrasch c5"]\n\n1. e4 e6 2. d4 d5 3. Nd2 c5 4. exd5 *'
)


# The Tarrasch chapters score better than the Advance, so they would win 3rd move on score. Fixtures force 3.e5
# by default to keep them as unselected alternative chapters.
FORCE_ADVANCE = {position('e4 e6 d4 d5'): 'e4e5'}


def run_fixture(
    tmp_path, monkeypatch, common_entry=False, alternative_first=False, multiple_entries=False, policy=FORCE_ADVANCE
):
    def forbidden(*args, **kwargs):
        raise AssertionError('A fully cached comparison must not use the network')

    monkeypatch.setattr(httpx.Client, 'request', forbidden)
    parts = [TARRASCH, SPLIT, ADVANCE] if alternative_first else [ADVANCE, TARRASCH, SPLIT]
    g = graph(tmp_path, '\n\n'.join(parts))
    cache = tmp_path / 'cache'
    cache.mkdir(exist_ok=True)
    leaves = {
        position('e4 e6 d4 d5 e5 c5 c3'): 40,
        position('e4 e6 d4 d5 Nd2 Nf6 e5'): 90,
        position('e4 e6 d4 d5 Nd2 c5 exd5'): 80,
    }
    for k, n in g.nodes.items():
        if k in leaves:
            evidence = data(leaves[k], 0, 100 - leaves[k])
        else:
            moves = list(n.edges)
            weights = [60, 40] if k == position('e4 e6 d4 d5 Nd2') else [100 // len(moves)] * len(moves)
            evidence = data(50, 0, 50, [(move, weight // 2, 0, weight // 2) for move, weight in zip(moves, weights)])
        cache_row(cache, k, evidence)
    config = tmp_path / 'config.json'
    configuration = {'entries': {c['id']: [{'path': ['e4', 'e6']}] for c in g.chapters}} if common_entry else {}
    if multiple_entries:
        configuration.setdefault('entries', {})['tarrasch'] = [
            {'path': ['e4', 'e6', 'd4', 'd5', 'Nd2', move]} for move in ('Nf6', 'c5')
        ]
    if policy:
        configuration['policy'] = policy
    config.write_text(json.dumps(configuration))
    args = SimpleNamespace(
        config=str(config),
        pgn=str(tmp_path / 'fixture.pgn'),
        color='white',
        output=str(tmp_path / 'white'),
        command='run',
        cache=str(cache),
        offline=True,
        refresh=False,
        prior=[0.5] * 3,
        sparse_threshold=30,
        tolerance=1,
    )
    cli.analyze(args)
    return json.loads((tmp_path / 'white.json').read_text()), cache


def sample(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nf3 Nc6 3. Bb5 *')
    evidence = {
        position(''): data(50, 0, 50),
        position('e4'): data(50, 0, 50, [('e7e5', 50, 0, 50)]),
        position('e4 e5'): data(60, 0, 40),
        position('e4 e5 Nf3'): data(38, 0, 62, [('b8c6', 32, 0, 48), ('d7d6', 6, 0, 14)]),
        position('e4 e5 Nf3 Nc6'): data(80, 0, 20),
        position('e4 e5 Nf3 Nc6 Bb5'): data(20, 0, 80),
    }
    return g, evidence


def check_score_tables(text):
    """CP appears only beside headline deltas, and each one reproduces C(score) - C(baseline)."""
    checked = 0
    for block in re.findall(r'(?m)(?:^\|.*\|\n)+', text):
        rows = [[cell.strip() for cell in row.strip().strip('|').split('|')] for row in block.splitlines()]
        headers = rows[0]
        for row in rows[2:]:
            assert len(row) == len(headers)
            for header, value in zip(headers, row):
                if ' cp)' not in value:
                    continue
                assert header == 'Delta' and {'Starting baseline', 'Repertoire score'} <= set(headers), (header, value)
                base, score = (
                    float(row[headers.index(h)].rstrip('%')) / 100 for h in ('Starting baseline', 'Repertoire score')
                )
                shown = float(re.search(r'\(([+-]?\d+) cp\)', value)[1])
                # Displayed scores are rounded, so allow for their rounding as well as the whole-number CP.
                assert shown == pytest.approx(centipawn_delta(score, base), abs=2.0)
                checked += 1
    return checked
