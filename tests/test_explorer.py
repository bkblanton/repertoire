import chess
import httpx
import pytest
from repertoire_score.graph import key
from repertoire_score.explorer import Explorer, DEFAULT_FILTERS


def test_cache_auth_identity_and_offline_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test-secret-never-persist')
    requests = []
    def respond(request):
        requests.append(request)
        assert request.headers['Authorization'] == 'Bearer test-secret-never-persist'
        assert request.url.params['moves'] == '20'
        assert request.url.params['topGames'] == '0'
        return httpx.Response(200,json={'white':1,'draws':0,'black':0,'moves':[]})
    client = Explorer(tmp_path,delay=0)
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
    offline = Explorer(tmp_path,offline=True)
    assert offline.get(position)['white'] == 1
    offline.close()
    other = Explorer(tmp_path,dict(DEFAULT_FILTERS,speeds='rapid'),offline=True)
    with pytest.raises(ValueError,match='Offline cache miss'):
        other.get(position)
    other.close()


def test_api_failure_is_not_zero_data(tmp_path,monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN','test-token')
    client = Explorer(tmp_path,delay=0)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(lambda _:httpx.Response(401)))
    with pytest.raises(RuntimeError,match='HTTP 401'):
        client.get(key(chess.Board()))
    assert not list(tmp_path.glob('*.json'))
    client.close()


def test_rate_limit_wait_and_retry(tmp_path,monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN','test-token')
    sleeps = []
    monkeypatch.setattr('repertoire_score.explorer.time.sleep',sleeps.append)
    responses = iter([httpx.Response(429,headers={'Retry-After':'65'}),httpx.Response(200,json={'white':0,'draws':0,'black':0,'moves':[]})])
    client = Explorer(tmp_path,delay=0,retries=1)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(lambda _:next(responses)))
    assert client.get(key(chess.Board()))['white'] == 0
    assert 65 in sleeps
    client.close()
