import chess
import httpx
import pytest

from repertoire_score.explorer import DEFAULT_FILTERS, Explorer
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
