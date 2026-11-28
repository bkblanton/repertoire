import json
import struct

import chess
import pytest
import zstandard
from helpers import position

from repertoire import eval_index, evals
from repertoire.eval_index import (
    PZSTD_SKIPPABLE_MAGIC,
    EvalIndex,
    build_index,
    index_path_for,
    pzstd_frames,
)
from repertoire.evals import Store, import_export, pending_in_store, resolve_source


def _eval_line(fen: str, *evals_: tuple[int, int, list[dict]]) -> str:
    return json.dumps(
        {'fen': fen, 'evals': [dict(depth=d, knodes=k, pvs=pvs) for d, k, pvs in evals_]},
        separators=(',', ':'),
    )


def _write_plain_zst(path, lines: list[str], newline_at_end: bool = True):
    text = '\n'.join(lines) + ('\n' if newline_at_end else '')
    path.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
    return path


def _write_pzstd(path, raw_bytes: bytes, frame_bytes: int):
    """Compress `raw_bytes` into independent pzstd frames of `frame_bytes` uncompressed bytes each,
    intentionally slicing across line boundaries."""
    cctx = zstandard.ZstdCompressor(level=1)
    out = bytearray()
    for i in range(0, len(raw_bytes), frame_bytes):
        comp = cctx.compress(raw_bytes[i : i + frame_bytes])
        out += struct.pack('<III', PZSTD_SKIPPABLE_MAGIC, 4, len(comp))
        out += comp
    path.write_bytes(bytes(out))
    return path


def _sample_positions(count: int) -> list[tuple[str, str]]:
    """Generate `count` distinct legal chess positions (canonical key, full 4-field FEN)."""
    board = chess.Board()
    found: dict[str, str] = {}
    queue = [board.copy()]
    while queue and len(found) < count:
        cur = queue.pop(0)
        for move in cur.legal_moves:
            cur.push(move)
            fen4 = ' '.join(cur.fen().split()[:4])
            key = evals.position_of(fen4)
            if key not in found:
                found[key] = fen4
                if len(found) < count:
                    queue.append(cur.copy())
            cur.pop()
            if len(found) >= count:
                break
    return list(found.items())


@pytest.mark.parametrize('newline_at_end', [False, True])
def test_build_index_and_lookup_handles_en_passant_and_deepest(tmp_path, newline_at_end):
    e4 = position('e4')
    e4_e5 = position('e4 e5')
    d4 = position('d4')
    lines = [
        _eval_line(
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3',
            (20, 50, [{'cp': 30, 'line': 'e7e5'}]),
            (36, 900, [{'cp': 24, 'line': 'c7c5'}]),
        ),
        _eval_line('rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -', (28, 80, [{'mate': 9, 'line': 'g1f3'}])),
    ]
    zst = _write_plain_zst(tmp_path / 'lichess_db_eval.jsonl.zst', lines, newline_at_end=newline_at_end)
    meta = build_index(zst, num_buckets=16, workers=1)
    idx_path = index_path_for(zst)
    assert idx_path.is_file() and meta['lines'] == 2
    index = EvalIndex(idx_path)
    assert index.matches_export(zst)
    found = dict(index.lookup({e4, e4_e5, d4}))
    assert set(found) == {e4, e4_e5}
    assert found[e4]['depth'] == 36 and found[e4]['pvs'][0]['cp'] == 24
    assert found[e4_e5]['pvs'] == [{'mate': 9, 'line': 'g1f3'}]


def test_pzstd_multiframe_indexing_preserves_lines_crossing_frame_boundaries(tmp_path):
    pairs = _sample_positions(60)
    lines = [_eval_line(fen4, (30, 100 + i, [{'cp': i, 'line': 'e2e4'}])) for i, (_, fen4) in enumerate(pairs)]
    raw = ('\n'.join(lines) + '\n').encode()
    # Slice every 350 bytes so almost every frame boundary splits a JSON line in half.
    zst = _write_pzstd(tmp_path / 'lichess_db_eval.jsonl.zst', raw, frame_bytes=350)
    frames = pzstd_frames(zst)
    assert frames is not None and len(frames) >= 10

    idx_path = tmp_path / 'lichess_db_eval.idx'
    meta = build_index(zst, idx_path, num_buckets=16, workers=2)
    assert meta['lines'] == len(pairs)
    found = dict(EvalIndex(idx_path).lookup([k for k, _ in pairs]))
    assert len(found) == len(pairs)
    for i, (k, _) in enumerate(pairs):
        assert found[k]['pvs'][0]['cp'] == i


def test_large_multi_bucket_lookup_exercises_parallel_reader(tmp_path):
    pairs = _sample_positions(220)
    lines = [_eval_line(fen4, (25, 10, [{'cp': i - 100, 'line': 'e2e4'}])) for i, (_, fen4) in enumerate(pairs)]
    zst = _write_plain_zst(tmp_path / 'lichess_db_eval.jsonl.zst', lines)
    idx_path = tmp_path / 'lichess_db_eval.idx'
    # 256 buckets with 220 positions guarantees >=128 populated buckets hit, exercising ThreadPoolExecutor.
    build_index(zst, idx_path, num_buckets=256, workers=1)
    found = dict(EvalIndex(idx_path).lookup([k for k, _ in pairs]))
    assert len(found) == len(pairs)
    for i, (k, _) in enumerate(pairs):
        assert found[k]['pvs'][0]['cp'] == i - 100


def test_index_shares_export_identity_works_standalone_and_rebuilds_when_stale(tmp_path, monkeypatch):
    e4 = position('e4')
    e4_e5 = position('e4 e5')
    zst = tmp_path / 'lichess_db_eval.jsonl.zst'
    idx = tmp_path / 'lichess_db_eval.idx'
    monkeypatch.setattr(evals, 'EXPORT', zst)
    monkeypatch.setattr(evals, 'INDEX', idx)

    _write_plain_zst(
        zst,
        [
            _eval_line(
                'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -', (30, 10, [{'cp': 20, 'line': 'e7e5'}])
            ),
            _eval_line(
                'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -', (30, 10, [{'cp': 22, 'line': 'g1f3'}])
            ),
        ],
    )
    store_path = tmp_path / 'evals.sqlite'
    # First import from raw .jsonl.zst for {e4}.
    with Store(store_path) as store:
        assert import_export(store, zst, {e4}) == 1
    # Build index from the same .jsonl.zst: {e4} is already known searched via shared source identity!
    build_index(zst, idx, num_buckets=16, workers=1)
    assert resolve_source() == idx
    assert pending_in_store(store_path, idx, {e4}) == set()
    assert pending_in_store(store_path, idx, {e4, e4_e5}) == {e4_e5}

    # Deleting the original .jsonl.zst leaves .idx fully usable standalone.
    zst.unlink()
    assert resolve_source() == idx
    with Store(store_path) as store:
        assert import_export(store, idx, {e4, e4_e5}) == 1
        assert store.get(e4_e5)['pvs'][0]['cp'] == 22

    # Dropping a newer .jsonl.zst makes resolve_source rebuild the stale .idx automatically.
    _write_plain_zst(
        zst,
        [
            _eval_line(
                'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -', (45, 500, [{'cp': 33, 'line': 'c7c5'}])
            ),
        ],
    )
    monkeypatch.setattr(
        evals, 'build_index', lambda src, dst: eval_index.build_index(src, dst, num_buckets=16, workers=1)
    )
    assert resolve_source(rebuild_stale_index=True) == idx
    assert EvalIndex(idx).matches_export(zst)
    with Store(store_path) as store:
        assert import_export(store, idx, {e4}) == 1
        assert store.get(e4)['depth'] == 45 and store.get(e4)['pvs'][0]['cp'] == 33
