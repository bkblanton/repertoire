import json
from types import SimpleNamespace

import chess
import httpx
import pytest
from helpers import cache_row, data, position

from repertoire import score
from repertoire.graph import parse
from repertoire.report.links import Chapters
from repertoire.report.sections import free_transpositions_section
from repertoire.vulnerabilities import analyze

# After 2...Qf6 3.Nd5 the line ends, so 3...Qd8 is unprepared; 4.Nc3 then reaches the prepared 2.Nc3 position.
PGN = '[ChapterURL "https://lichess.org/study/test/vienna"]\n[ChapterName "Vienna"]\n\n1. e4 e5 2. Nc3 Qf6 3. Nd5 *'


def first_legal(k, besides):
    board = chess.Board(k + ' 0 1')
    return next(m.uci() for m in sorted(board.legal_moves, key=lambda m: m.uci()) if m.uci() not in besides)


@pytest.fixture
def scored(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('no network'))
    pgn = tmp_path / 'white.pgn'
    pgn.write_text(PGN, encoding='utf-8')
    cache = tmp_path / 'cache'
    cache.mkdir()
    for k, node in parse(pgn).nodes.items():
        if chess.Board(k + ' 0 1').turn == chess.WHITE:
            cache_row(cache, k, data(52, 6, 42))
            continue
        rows = [(m, 300, 40, 260) for m in node.edges]
        rows.append((first_legal(k, node.edges), 100, 20, 80))
        cache_row(cache, k, data(*(sum(r[i] for r in rows) for i in (1, 2, 3)), rows))
    # The reply that allows the transposition scores poorly for you in the database.
    after = position('e4 e5 Nc3 Qf6 Nd5')
    cache_row(cache, after, data(150, 30, 320, [('f6d8', 50, 10, 140), ('c7c6', 100, 20, 180)]))
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
    path = tmp_path / 'white.json'
    return path, cache, json.loads(path.read_text())


def test_a_move_back_into_preparation_is_scored_against_the_reply(scored):
    path, cache, saved = scored
    rows = analyze(path, cache)['overall']['free_transpositions']
    row = next(r for r in rows if r['exit_position'] == position('e4 e5 Nc3 Qf6 Nd5 Qd8'))
    assert row['transposing_san'] == 'Nc3'
    assert row['transposition_target'] == position('e4 e5 Nc3')
    assert row['database_score'] == pytest.approx((50 + 5) / 200)
    assert row['change_pp'] == pytest.approx(100 * (row['repertoire_score'] - row['database_score']))
    assert row['move_change_pp'] + row['preparation_change_pp'] == pytest.approx(row['change_pp'])
    low, high = row['change_interval_pp']
    assert low < row['change_pp'] < high
    assert row['chapter_attribution']['context_ids'] == ['vienna']
    assert not row['sparse']


def test_section_lists_the_transposing_move_and_its_chapter(scored):
    path, cache, saved = scored
    scope = analyze(path, cache)['overall']
    refs = Chapters(saved)
    text = '\n'.join(free_transpositions_section(scope, refs, 5))
    assert '### Free transpositions' in text
    assert '**Transposing raises your score**' in text
    assert '4. Nc3' in text and 'to [W1]' in text
    assert free_transpositions_section(scope, refs, 5, chapter='elsewhere') == []
