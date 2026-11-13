import io
import json
from types import SimpleNamespace

import chess.pgn
import httpx
import pytest
from helpers import cache_row, data, graph, position

from repertoire import build, coverage
from repertoire import score as cli
from repertoire.board_cache import position_of
from repertoire.context import AnalysisContext
from repertoire.graph import chapter_ids, parse_games

# 1...d6 and 1...Nf6 both reach the Pirc position after 2.d4, which only the Pirc chapter answers (3.Nc3); the
# Alekhine chapter ends there. 2.Bc4 is a side variation that loses to 2.Nf3.
CHAPTERS = (
    '[ChapterURL "https://lichess.org/study/test/pirc"]\n[ChapterName "Pirc"]\n\n'
    '1. e4 d6 2. d4 Nf6 3. Nc3 *\n\n'
    '[ChapterURL "https://lichess.org/study/test/alekhine"]\n[ChapterName "Alekhine"]\n\n'
    '1. e4 Nf6 2. d4 d6 *\n\n'
    '[ChapterURL "https://lichess.org/study/test/open"]\n[ChapterName "Open games"]\n\n'
    '1. e4 { [%cal Ge2e4] main idea } e5 2. Nf3 $1 { develops } (2. Bc4 { rare idea }) *'
)
PIRC = position('e4 d6 d4 Nf6')


def score(tmp_path, monkeypatch, replies):
    """Score the chapters offline, with the given reply counts after 1.e4 and even tables elsewhere."""
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('Coverage must stay offline'))
    g = graph(tmp_path, CHAPTERS)
    cache = tmp_path / 'cache'
    cache.mkdir(exist_ok=True)
    tables = {
        position('e4'): [(m, n // 2, 0, n - n // 2) for m, n in replies.items()],
        position('e4 d6 d4'): [('g8f6', 50, 0, 50)],
        position('e4 Nf6 d4'): [('d7d6', 40, 0, 60)],
        position('e4 e5 Nf3'): [('b8c6', 60, 0, 40), ('d7d6', 30, 0, 10)],
        position('e4 e5 Bc4'): [('g8f6', 20, 0, 80)],
        position('e4 d6 d4 Nf6 Nc3'): [('g7g6', 70, 0, 30)],
    }
    for k in [*g.nodes, position('e4 e5 Bc4')]:
        rows = tables.get(k, [])
        totals = [sum(r[i] for r in rows) for i in (1, 2, 3)] if rows else [50, 0, 50]
        cache_row(cache, k, data(*totals, rows))
    config = tmp_path / 'config.json'
    config.write_text('{}')
    cli.analyze(
        SimpleNamespace(
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
    )
    return tmp_path / 'white.json', cache


def test_reach_counts_only_routes_through_kept_positions(tmp_path, monkeypatch):
    path, cache = score(tmp_path, monkeypatch, {'e7e5': 900, 'd7d6': 50, 'g8f6': 50})
    result = coverage.Coverage(AnalysisContext(path), cache)
    assert result.reach[PIRC] == pytest.approx(0.1)
    # Each route arrives through a position met in 1 in 20 games, below 1 in 12, so the Pirc position is cut
    # although its two routes add up to 1 in 10.
    level = result.level(12)
    assert PIRC in level.cut and position('e4 d6') in level.cut and position('e4 Nf6') in level.cut
    row = result.row(level)
    assert (row['decisions'], row['dropped']) == (2, 3)
    assert row['leaving_earlier'] == pytest.approx(0.1)
    details = result.details(level)
    assert {c['position']: c['decisions_dropped'] for c in details['cuts']} == {
        position('e4 d6'): 2,
        position('e4 Nf6'): 1,
    }
    # The cuts' costs add up exactly to the change in score.
    assert sum(c['cost_pp'] for c in details['cuts']) == pytest.approx(-row['change_pp'])
    # With 1 in 25 both routes are kept, and so is the Pirc position.
    assert PIRC not in result.level(25).cut
    # Goals keep fewer moves as they get stricter.
    kept = [result.row(result.level(goal))['decisions'] for goal in (2, 12, 25, None)]
    assert kept == sorted(kept)


def test_trimmed_pgn_keeps_comments_drops_lost_alternatives_and_leaves_the_source(tmp_path, monkeypatch):
    path, cache = score(tmp_path, monkeypatch, {'e7e5': 900, 'd7d6': 50, 'g8f6': 50})
    source = tmp_path / 'fixture.pgn'
    original = source.read_bytes()
    value = coverage.run(path, cache, 12)
    text = (tmp_path / 'trimmed' / 'white-1-in-12.pgn').read_text(encoding='utf-8')
    assert source.read_bytes() == original
    assert '{ [%cal Ge2e4] main idea }' in text and '$1' in text and '{ develops }' in text
    assert '[ChapterName "Open games"]' in text
    # The lost side variation and the cut lines are gone, at every goal.
    assert 'Bc4' not in text and 'rare idea' not in text and 'Nc3' not in text and 'd6' not in text
    assert value['at_goal']['never_reached'] == 1
    assert value['validation']['trimmed_pgn_reproduces_repertoire']
    # The Alekhine chapter keeps only 1.e4, which the Pirc chapter before it already has.
    assert value['at_goal']['pgn']['chapters_left_out'] == ['alekhine']
    assert [g.headers['ChapterName'] for g in chess_games(text)] == ['Pirc', 'Open games']
    full = coverage.run(path, cache, 1000)
    assert not (tmp_path / 'trimmed' / 'white-1-in-12.pgn').exists()
    every = (tmp_path / 'trimmed' / 'white-1-in-1000.pgn').read_text(encoding='utf-8')
    assert 'Nc3' in every and 'Bc4' not in every
    assert full['levels'][-1]['dropped'] == 0


def chess_games(text):
    stream, games = io.StringIO(text), []
    while (game := chess.pgn.read_game(stream)) is not None:
        games.append(game)
    return games


def test_kept_line_recorded_only_below_a_cut_moves_to_a_route_that_reaches_it(tmp_path, monkeypatch):
    # 1...d6 is rare and 1...Nf6 common: the Pirc position stays, but 3.Nc3 is recorded only after 1...d6.
    path, cache = score(tmp_path, monkeypatch, {'e7e5': 700, 'd7d6': 20, 'g8f6': 280})
    value, text = coverage.analyze(path, cache, 20)
    assert [m['position'] for m in value['at_goal']['pgn']['moved']] == [PIRC]
    games = {g.headers['ChapterName']: g for g in chess_games(text)}
    alekhine = games['Alekhine'].accept(chess.pgn.StringExporter(headers=False, comments=False))
    assert alekhine.startswith('1. e4 Nf6 2. d4 d6 3. Nc3')
    assert 'd6 2. d4' not in games['Pirc'].accept(chess.pgn.StringExporter(headers=False))
    # Read back and scored from scratch, the trimmed PGN gives the trimmed repertoire's score.
    trimmed = tmp_path / 'rescored'
    trimmed.mkdir()
    (trimmed / 'fixture.pgn').write_text(text, encoding='utf-8')
    (trimmed / 'config.json').write_text('{}')
    cli.analyze(
        SimpleNamespace(
            config=str(trimmed / 'config.json'),
            pgn=str(trimmed / 'fixture.pgn'),
            color='white',
            output=str(trimmed / 'white'),
            command='run',
            cache=str(cache),
            offline=True,
            refresh=False,
            prior=[0.5] * 3,
            sparse_threshold=30,
            tolerance=1,
        )
    )
    rescored = json.loads((trimmed / 'white.json').read_text())['overall']
    row = next(r for r in value['levels'] if r['goal'] == 20)
    assert rescored['raw_empirical_score'] == pytest.approx(row['score']['raw_empirical_score'], abs=1e-12)


def test_goal_validation_and_default():
    assert coverage.goal_from_config({}) == coverage.DEFAULT_GOAL == 300
    assert coverage.goal_from_config({'coverage_goal': 500}) == 500
    for bad in (1, 0, 'often', True, 2.5):
        with pytest.raises(ValueError, match='coverage_goal'):
            coverage.goal_from_config({'coverage_goal': bad})
    assert coverage.goals(250) == [100, 150, 200, 250, 300, 400, 600, 800, 1000, None]


def test_chapter_ids_match_the_graph(tmp_path):
    text = '1. e4 *\n\n1. d4 *\n\n[ChapterURL "https://lichess.org/study/x/same"]\n\n1. c4 *\n\n'
    text += '[ChapterURL "https://lichess.org/study/x/same"]\n\n1. Nf3 *'
    games = chess_games(text)
    assert chapter_ids(games, {'2'}) == ['1', None, 'same', 'same-copy-3']
    assert [c['id'] for c in parse_games(chess_games(text), {'2'}).chapters] == ['1', 'same', 'same-copy-3']


def test_changing_the_goal_reruns_only_coverage_and_render(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('Batch must stay offline'))
    text = '1. e4 e5 2. Nf3 Nc6 *\n\n1. d4 d5 2. Nf3 Nf6 *'
    white, black = tmp_path / 'white.pgn', tmp_path / 'black.pgn'
    white.write_text(text)
    black.write_text(text)
    cache = tmp_path / 'cache'
    cache.mkdir()
    for k, n in graph(tmp_path, text).nodes.items():
        size = 100 // max(1, len(n.edges))
        cache_row(cache, k, data(50, 0, 50, [(m, size // 2, 0, size // 2) for m in n.edges]))
    configs = {color: tmp_path / f'{color}.config.json' for color in ('white', 'black')}
    for path in configs.values():
        path.write_text('{"policy": {}}')

    def run():
        return build.build(
            white,
            black,
            white_config=configs['white'],
            black_config=configs['black'],
            directory=tmp_path / 'data',
            cache=cache,
            offline=True,
        )

    run()
    assert (tmp_path / 'trimmed' / 'white-1-in-300.pgn').is_file()
    score = (tmp_path / 'data' / 'white.json').read_bytes()
    configs['white'].write_text('{"policy": {}, "coverage_goal": 2}')
    changed = run()
    assert {s['step'] for s in changed['steps'] if s['status'] == 'built'} == {'white.coverage', 'render'}
    assert (tmp_path / 'data' / 'white.json').read_bytes() == score
    assert (tmp_path / 'trimmed' / 'white-1-in-2.pgn').is_file()
    assert not (tmp_path / 'trimmed' / 'white-1-in-300.pgn').exists()
    assert json.loads((tmp_path / 'data' / 'white.json').read_text())['manifest']['configuration'] == {'policy': {}}
    report = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert '[white-1-in-2.pgn](trimmed/white-1-in-2.pgn)' in report and '**1 in 2 games** (your goal)' in report
    assert '[Trimmed PGN](trimmed/white-1-in-2.pgn)' in (tmp_path / 'summary.md').read_text(encoding='utf-8')
    assert position_of(position('e4')) == position('e4')
