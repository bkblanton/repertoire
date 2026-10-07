import httpx
import pytest

from repertoire_score import build, studies
from repertoire_score.graph import parse

CHAPTERS = '''[Event "Study: One"]
[Date "{date}"]
[ChapterURL "https://lichess.org/study/abcdEFGH/chap0001"]

1. e4 e5 2. Nc3 (2. Nf3) *

[Event "Study: Two"]
[Date "{date}"]
[SetUp "1"]
[FEN "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"]
[ChapterURL "https://lichess.org/study/abcdEFGH/chap0002"]

2. Nc3 Nf6 *
'''


def client(responses, seen):
    def handler(request):
        seen.append(request)
        return responses.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    'value',
    [
        'abcd1234',
        'https://lichess.org/study/abcd1234',
        'lichess.org/study/abcd1234/',
        'https://lichess.org/study/abcd1234/efgh5678#3',
    ],
)
def test_study_id_accepts_ids_and_study_or_chapter_urls(value):
    assert studies.study_id(value) == 'abcd1234'


def test_study_id_rejects_other_urls():
    with pytest.raises(ValueError):
        studies.study_id('https://lichess.org/broadcast/abcd1234')


def test_fetch_exports_whole_studies_and_ignores_export_date_changes(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test')
    sources = dict(white='https://lichess.org/study/abcdEFGH', black='https://lichess.org/study/ijklMNOP/chap0001')
    seen = []
    first = [httpx.Response(200, text=CHAPTERS.format(date='2026.10.05')) for _ in range(2)]
    paths = studies.fetch(sources, tmp_path, client=client(first, seen))
    assert [r.url.path for r in seen] == ['/api/study/abcdEFGH.pgn', '/api/study/ijklMNOP.pgn']
    assert seen[0].url.params['variations'] == 'true'
    assert [c['id'] for c in parse(paths['white']).chapters] == ['chap0001', 'chap0002']
    original = paths['white'].read_text(encoding='utf-8')
    later = [httpx.Response(200, text=CHAPTERS.format(date='2026.10.06')) for _ in range(2)]
    studies.fetch(sources, tmp_path, client=client(later, seen))
    assert paths['white'].read_text(encoding='utf-8') == original
    changed = [httpx.Response(200, text=CHAPTERS.format(date='2026.10.06').replace('Nf6', 'Nc6')) for _ in range(2)]
    studies.fetch(sources, tmp_path, client=client(changed, seen))
    assert 'Nc6' in paths['white'].read_text(encoding='utf-8')
    assert not list(tmp_path.glob('*.tmp'))


def test_failed_or_malformed_export_keeps_previous_file(tmp_path, monkeypatch):
    monkeypatch.setenv('LICHESS_TOKEN', 'test')
    sources = dict(white='abcdEFGH', black='ijklMNOP')
    path = tmp_path / 'white.pgn'
    path.write_text('previous', encoding='utf-8')
    with pytest.raises(RuntimeError, match='study:read'):
        studies.fetch(sources, tmp_path, client=client([httpx.Response(403)], []))
    with pytest.raises(ValueError):
        studies.fetch(sources, tmp_path, client=client([httpx.Response(200, text='1. e4 Ke7?? 2. Qxx *')], []))
    assert path.read_text(encoding='utf-8') == 'previous'
    assert not list(tmp_path.glob('*.tmp'))


def test_fetch_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv('LICHESS_TOKEN', raising=False)
    with pytest.raises(ValueError, match='LICHESS_TOKEN'):
        studies.fetch(dict(white='abcdEFGH', black='ijklMNOP'), tmp_path)


def test_export_code_does_not_invalidate_analyses():
    assert not any(path.endswith('studies.py') for path in build.code_inputs(presentation=True))
