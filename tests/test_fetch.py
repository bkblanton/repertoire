import chess
import httpx
import pytest
from helpers import data

from repertoire_score import build, fetch
from repertoire_score.graph import parse

TEXT = (
    '[ChapterURL "https://lichess.org/study/test/a"]\n[ChapterName "King pawn"]\n\n'
    '1. e4 e5 2. Nf3 Nc6 (2... d6 3. d4) *\n\n'
    '[ChapterURL "https://lichess.org/study/test/b"]\n[ChapterName "Queen pawn"]\n\n'
    '1. d4 d5 2. Nf3 Nf6 *'
)
BLACK = '[ChapterURL "https://lichess.org/study/test/c"]\n[ChapterName "Queen pawn"]\n\n1. d4 d5 2. Nf3 Nf6 *'


@pytest.fixture
def lichess(tmp_path, monkeypatch):
    """Both PGNs and a mock Explorer that answers every legal position and records each request."""
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    monkeypatch.setattr('repertoire_score.explorer.time.sleep', lambda _: None)
    white, black = tmp_path / 'white.pgn', tmp_path / 'black.pgn'
    white.write_text(TEXT)
    # A different Black repertoire, so its scoring tables cannot stand in for White's own-move parents.
    black.write_text(BLACK)
    nodes = parse(white).nodes
    requests = []

    def respond(request):
        fen = request.url.params['fen']
        requests.append(fen)
        k = chess.Board(fen).fen().rsplit(' ', 2)[0]
        edges = nodes[k].edges if k in nodes else {}
        size = 100 // max(1, len(edges))
        return httpx.Response(200, json=data(50, 0, 50, [(m, size // 2, 0, size // 2) for m in edges]))

    client = httpx.Client
    monkeypatch.setattr(
        'repertoire_score.explorer.httpx.Client',
        lambda **options: client(transport=httpx.MockTransport(respond), **options),
    )
    return tmp_path, white, black, requests


def test_build_fetches_every_table_first_then_runs_offline(lichess, monkeypatch, capsys):
    tmp_path, white, black, requests = lichess
    original = build.Builder.step

    def step(self, name, inputs, outputs, action):
        # No stage may contact Lichess: every table was fetched before the first stage started.
        fetched = len(requests)
        original(self, name, inputs, outputs, action)
        assert len(requests) == fetched, name

    monkeypatch.setattr(build.Builder, 'step', step)
    options = dict(white_config=None, black_config=None, directory=tmp_path / 'data', cache=tmp_path / 'cache')
    result = build.build(white, black, **options)
    assert result['built'] == 17
    assert len(requests) == len(set(requests))
    out = capsys.readouterr().out
    assert f'white and black: {len(requests)} Explorer tables: 0 cached, {len(requests)} to fetch' in out
    assert out.index('to fetch') < out.index('Building white.score')
    requests.clear()
    assert build.build(white, black, **options)['reused'] == 17
    assert requests == []


def test_fetch_includes_own_move_parents_and_dry_run_makes_no_requests(lichess, monkeypatch, capsys):
    tmp_path, white, black, requests = lichess
    pgns, configs, cache = dict(white=white, black=black), {}, tmp_path / 'cache'
    monkeypatch.delenv('LICHESS_TOKEN')
    fetch.fetch(pgns, configs, cache, dry_run=True)
    assert requests == []
    assert 'to fetch (at least' in capsys.readouterr().out
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    fetch.fetch(pgns, configs, cache)
    fetched = {chess.Board(fen).fen().rsplit(' ', 2)[0] for fen in requests}
    # White's own-move parent after 1. e4 e5 is not a scoring table but is still fetched.
    board = chess.Board()
    for move in ('e4', 'e5'):
        board.push_san(move)
    assert board.fen().rsplit(' ', 2)[0] in fetched
    requests.clear()
    fetch.fetch(pgns, configs, cache)
    assert requests == []
    assert capsys.readouterr().out.endswith(f'white and black: {len(fetched)} Explorer tables, all cached\n')
