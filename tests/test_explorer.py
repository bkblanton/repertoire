import chess
import httpx
import pytest

from repertoire_score.explorer import DEFAULT_FILTERS, Explorer, Progress, collect, duration, fetch_missing
from repertoire_score.graph import key


def test_cache_auth_identity_and_offline_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-secret-never-persist')
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers['Authorization'] == 'Bearer test-secret-never-persist'
        assert request.url.params['moves'] == '20'
        assert request.url.params['topGames'] == '0'
        return httpx.Response(200, json={'white': 1, 'draws': 0, 'black': 0, 'moves': []})

    client = Explorer(tmp_path, delay=0)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(respond))
    # Mock transport still uses the exact auth header configured by Explorer.
    client.client.headers['Authorization'] = 'Bearer test-secret-never-persist'
    position = key(chess.Board())
    client.get(position)
    client.get(position)
    assert len(requests) == 1
    assert 'test-secret' not in next(tmp_path.glob('*.json')).read_text()
    client.close()
    monkeypatch.delenv('LICHESS_TOKEN')
    offline = Explorer(tmp_path, offline=True)
    assert offline.get(position)['white'] == 1
    offline.close()
    other = Explorer(tmp_path, dict(DEFAULT_FILTERS, speeds='rapid'), offline=True)
    with pytest.raises(ValueError, match='Offline cache miss'):
        other.get(position)
    other.close()


def test_api_failure_is_not_zero_data(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    client = Explorer(tmp_path, delay=0)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(401)))
    with pytest.raises(RuntimeError, match='HTTP 401'):
        client.get(key(chess.Board()))
    assert not list(tmp_path.glob('*.json'))
    client.close()


def test_rate_limit_wait_and_retry(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    sleeps = []
    monkeypatch.setattr('repertoire_score.explorer.time.sleep', sleeps.append)
    responses = iter(
        [
            httpx.Response(429, headers={'Retry-After': '65'}),
            httpx.Response(200, json={'white': 0, 'draws': 0, 'black': 0, 'moves': []}),
        ]
    )
    client = mock_client(tmp_path, lambda _: next(responses))
    assert client.get(key(chess.Board()))['white'] == 0
    assert 65 in sleeps
    client.close()


EMPTY = {'white': 0, 'draws': 0, 'black': 0, 'moves': []}


def mock_client(tmp_path, handler, **options):
    client = Explorer(tmp_path, delay=0, **options)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_sustained_rate_limit_is_waited_out(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    sleeps = []
    monkeypatch.setattr('repertoire_score.explorer.time.sleep', sleeps.append)
    # An hour of HTTP 429 is still a temporary condition, never a reason to abandon the run.
    responses = iter([httpx.Response(429)] * 60 + [httpx.Response(200, json=EMPTY)])
    client = mock_client(tmp_path, lambda _: next(responses), patience=0)
    assert client.get(key(chess.Board()))['white'] == 0
    assert sleeps.count(60) == 60
    assert len(list(tmp_path.glob('*.json'))) == 1
    client.close()


def test_outages_are_retried_with_growing_waits(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    sleeps = []
    monkeypatch.setattr('repertoire_score.explorer.time.sleep', sleeps.append)
    outcomes = iter([None] * 6 + [httpx.Response(503)] * 2 + [httpx.Response(200, json=EMPTY)])

    def respond(request):
        outcome = next(outcomes)
        if outcome is None:
            raise httpx.ConnectError('offline', request=request)
        return outcome

    client = mock_client(tmp_path, respond)
    assert client.get(key(chess.Board()))['white'] == 0
    assert [s for s in sleeps if s] == [1, 2, 4, 8, 16, 32, 60, 60]
    client.close()


def test_outage_longer_than_patience_fails_without_caching(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    sleeps = []
    monkeypatch.setattr('repertoire_score.explorer.time.sleep', sleeps.append)
    client = mock_client(tmp_path, lambda _: httpx.Response(502), patience=600)
    with pytest.raises(RuntimeError, match='still unavailable after 10 minutes.*HTTP 502'):
        client.get(key(chess.Board()))
    assert sum(sleeps) == 600
    assert not list(tmp_path.glob('*.json'))
    client.close()


def test_client_errors_fail_immediately(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    sleeps = []
    monkeypatch.setattr('repertoire_score.explorer.time.sleep', sleeps.append)
    client = mock_client(tmp_path, lambda _: httpx.Response(400))
    with pytest.raises(RuntimeError, match='HTTP 400'):
        client.get(key(chess.Board()))
    assert not any(sleeps)
    client.close()


def positions(n):
    board, result = chess.Board(), []
    for move in ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'd2d3', 'f8c5'][:n]:
        board.push_uci(move)
        result.append(key(board))
    return result


def test_collect_counts_cached_and_missing_and_keeps_order(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    monkeypatch.setattr('repertoire_score.explorer.time.sleep', lambda _: None)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=EMPTY)

    boards = positions(6)
    client = mock_client(tmp_path, respond)
    collect(client, boards[:2], 'warm')
    capsys.readouterr()
    assert not client.cached(boards[2]) and client.cached(boards[0])
    evidence = collect(client, [boards[3], boards[0], boards[2], boards[1], boards[3]], 'white scores')
    assert list(evidence) == [boards[3], boards[0], boards[2], boards[1]]
    assert len(requests) == 4
    out = capsys.readouterr().out
    assert 'white scores: 4 Explorer tables: 2 cached, 2 to fetch' in out
    assert 'white scores: 2/2 fetched' in out
    collect(client, boards[:4], 'again')
    assert capsys.readouterr().out.splitlines() == ['again: 4 Explorer tables, all cached']
    client.close()


def test_progress_estimates_from_recent_rate(monkeypatch, capsys):
    clock = iter([0, 10, 20, 30, 40])
    monkeypatch.setattr('repertoire_score.explorer.time.monotonic', lambda: next(clock))
    progress = Progress(100, 'black scores')
    progress.INTERVAL = 0
    progress.advance()
    progress.advance()
    progress.advance()
    lines = capsys.readouterr().out.splitlines()
    # Three tables in 30s: ten seconds each, so 97 left take about 16 minutes.
    assert lines[-1] == 'black scores: 3/100 fetched, ~16m left'
    assert lines[0] == 'black scores: 1/100 fetched'


def test_backoff_messages_say_how_far_the_run_has_got(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    monkeypatch.setattr('repertoire_score.explorer.time.sleep', lambda _: None)
    responses = iter([httpx.Response(200, json=EMPTY), httpx.Response(429), httpx.Response(200, json=EMPTY)])
    client = mock_client(tmp_path, lambda _: next(responses))
    collect(client, positions(2), 'white scores')
    assert 'rate-limited (HTTP 429); waiting 60s (1/2 fetched)' in capsys.readouterr().out
    assert client.note == ''
    client.close()


@pytest.mark.parametrize(('seconds', 'text'), [(0, '0s'), (45, '45s'), (90, '2m'), (3599, '1h 00m'), (11100, '3h 05m')])
def test_duration(seconds, text):
    assert duration(seconds) == text


def test_interrupted_fetch_reports_progress_and_keeps_fetched_tables(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    monkeypatch.setattr('repertoire_score.explorer.time.sleep', lambda _: None)
    responses = iter([httpx.Response(200, json=EMPTY), KeyboardInterrupt])

    def respond(_):
        response = next(responses)
        if response is KeyboardInterrupt:
            raise KeyboardInterrupt
        return response

    client = mock_client(tmp_path, respond)
    with pytest.raises(KeyboardInterrupt):
        fetch_missing(client, positions(3), 'white and black')
    assert 'white and black: stopped at 1/3 fetched; tables fetched so far stay cached' in capsys.readouterr().out
    assert len(list(tmp_path.glob('*.json'))) == 1
    assert client.note == ''
    client.close()


def test_online_explorer_removes_partial_writes(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-token')
    (tmp_path / 'partial.tmp').write_text('{"trunc')
    Explorer(tmp_path, offline=True).close()
    assert (tmp_path / 'partial.tmp').exists()
    Explorer(tmp_path).close()
    assert not list(tmp_path.glob('*.tmp'))
