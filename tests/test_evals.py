import json
from types import SimpleNamespace

import httpx
import pytest
import zstandard
from helpers import cache_row, data, position

from repertoire import evals, score
from repertoire.context import AnalysisContext
from repertoire.evals import Evaluation, Store, frequencies, import_export, neighborhood, scan

E4 = position('e4')
E4_E5 = position('e4 e5')
D4 = position('d4')


def line(fen, *evals_):
    # The export is compact JSON, which the line matcher relies on.
    evaluations = [dict(pvs=pvs, knodes=knodes, depth=depth) for depth, knodes, pvs in evals_]
    return json.dumps({'fen': fen, 'evals': evaluations}, separators=(',', ':'))


def export(tmp_path, lines, newline_at_end=False):
    path = tmp_path / 'lichess_db_eval.jsonl.zst'
    text = '\n'.join(lines) + ('\n' if newline_at_end else '')
    path.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
    return path


def evaluation(depth, cp=20):
    return Evaluation(depth=depth, knodes=100, pvs=[{'cp': cp, 'line': 'e7e5'}], source='export', retrieved_at='now')


EXPORT_LINES = [
    line('8/8/8/8/8/5k2/8/4K3 w - -', (30, 10, [{'cp': 0, 'line': 'e1e2'}])),
    # The export may name an en passant square no pawn can use; the position is the same as without it.
    line(
        'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3',
        (20, 50, [{'cp': 30, 'line': 'e7e5'}]),
        (34, 900, [{'cp': 25, 'line': 'c7c5'}, {'cp': 31, 'line': 'e7e5'}]),
    ),
    line('rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -', (25, 70, [{'mate': 12, 'line': 'g1f3'}])),
]


@pytest.mark.parametrize('size', [64, 1 << 20])
@pytest.mark.parametrize('newline_at_end', [False, True])
def test_scan_finds_wanted_positions_across_blocks(tmp_path, size, newline_at_end):
    path = export(tmp_path, EXPORT_LINES, newline_at_end)
    found = dict(scan(path, {E4, E4_E5, D4}, size))
    assert set(found) == {E4, E4_E5}
    assert found[E4]['depth'] == 34
    assert found[E4]['pvs'][0] == {'cp': 25, 'line': 'c7c5'}
    assert found[E4_E5]['pvs'] == [{'mate': 12, 'line': 'g1f3'}]
    assert found[E4]['source'] == 'export'


def test_import_searches_each_export_once_per_position(tmp_path, monkeypatch):
    path = export(tmp_path, EXPORT_LINES)
    with Store(tmp_path / 'evals.sqlite') as store:
        assert import_export(store, path, {E4, D4}) == 1
        searched = []
        real_scan = evals.scan
        monkeypatch.setattr(
            evals, 'scan', lambda p, wanted, size=0: searched.append(set(wanted)) or real_scan(p, wanted)
        )
        assert import_export(store, path, {E4, D4}) == 0
        assert searched == []
        assert import_export(store, path, {E4, D4, E4_E5}) == 1
        assert searched == [{E4_E5}]
        assert set(store.read([E4, E4_E5, D4])) == {E4, E4_E5}


def test_store_keeps_the_deepest_evaluation(tmp_path):
    with Store(tmp_path / 'evals.sqlite') as store:
        assert store.put(E4, evaluation(30, cp=7))
        assert not store.put(E4, evaluation(25))
        assert store.get(E4)['pvs'][0]['cp'] == 7
        assert store.put(E4, evaluation(40, cp=-5))
        assert store.get(E4)['pvs'][0]['cp'] == -5
    with Store(tmp_path / 'evals.sqlite') as store:
        assert store.get(E4)['depth'] == 40


def scored(tmp_path, monkeypatch):
    """A cached White score where preparation ends after unprepared replies and at a theory leaf."""
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('no network'))
    pgn = tmp_path / 'white.pgn'
    pgn.write_text('1. e4 e5 (1... c5) 2. Nf3 *', encoding='utf-8')
    cache = tmp_path / 'cache'
    cache.mkdir()
    tables = {
        position(''): data(50, 0, 50, [('e2e4', 50, 0, 50)]),
        position('e4'): data(50, 0, 50, [('e7e5', 30, 0, 30), ('c7c5', 15, 0, 15), ('d7d5', 5, 0, 5)]),
        position('e4 c5'): data(60, 0, 40),
        position('e4 e5 Nf3'): data(40, 0, 40, [('b8c6', 25, 0, 25), ('g8f6', 15, 0, 15)]),
    }
    for k, table in tables.items():
        cache_row(cache, k, table)
    args = SimpleNamespace(
        config=None,
        pgn=str(pgn),
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
    score.analyze(args)
    return tmp_path / 'white.json', cache


def test_frequencies_cover_every_place_preparation_ends(tmp_path, monkeypatch):
    path, cache = scored(tmp_path, monkeypatch)
    analysis = AnalysisContext(path)
    reach, exits = frequencies(analysis, cache)
    assert exits == pytest.approx(
        {
            position('e4 c5'): 0.3,
            position('e4 d5'): 0.1,
            position('e4 e5 Nf3 Nc6'): 0.375,
            position('e4 e5 Nf3 Nf6'): 0.225,
        }
    )
    assert reach[position('e4 e5 Nf3')] == pytest.approx(0.6)
    assert set(exits) <= neighborhood(analysis.graph.nodes, 1)


def test_neighborhood_adds_every_position_one_move_away():
    assert neighborhood([position('')], 0) == {position('')}
    assert len(neighborhood([position('')], 1)) == 21
