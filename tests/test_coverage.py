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


def _eval_entry(depth, pvs):
    from repertoire.evals import Evaluation

    return Evaluation(depth=depth, knodes=500, pvs=pvs, source='export', retrieved_at='2026-11-27')


def test_critical_protection_rescues_sharp_lines_skips_trivial_recaptures_and_pins(
    tmp_path, monkeypatch, evaluation_store
):
    from repertoire.evals import Store

    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('Must stay offline'))
    # Main line: 1.e4 e5 -> 2.Nf3 (inaccurate: drops from +80cp to 0cp; pv0=Nf6) -> 2...Nf6 3.Nxe5 (rescued refutation).
    # Rare branch 1...c5 -> 2.Nc3 (cut ancestor) Nc6 -> 3.Bc4 (only-move cliff +15pp vs d4; rescued with 2.Nc3),
    #   whereas 2...e6 3.d4 cxd4 4.Qxd4 (obvious single recapture -> must NOT rescue).
    # Rare branch 1...d5 2.Nc3 e6 3.exd5 (engine >> DB on 1...d5 -> rescues 2.Nc3).
    # Pinned branch 1...g6 2.d4.
    pgn_text = (
        '[ChapterURL "https://lichess.org/study/test/main"]\n[ChapterName "Main"]\n\n'
        '1. e4 e5 2. Nf3 Nf6 3. Nxe5 *\n\n'
        '[ChapterURL "https://lichess.org/study/test/sic"]\n[ChapterName "Sicilian"]\n\n'
        '1. e4 c5 2. Nc3 Nc6 (2... e6 3. d4 cxd4 4. Qxd4) 3. Bc4 *\n\n'
        '[ChapterURL "https://lichess.org/study/test/scand"]\n[ChapterName "Scandi"]\n\n'
        '1. e4 d5 2. Nc3 e6 3. exd5 *\n\n'
        '[ChapterURL "https://lichess.org/study/test/pin"]\n[ChapterName "Pin"]\n\n'
        '1. e4 g6 2. d4 *'
    )
    g = graph(tmp_path, pgn_text)
    cache = tmp_path / 'cache'
    cache.mkdir()
    tables = {
        position('e4'): [('e7e5', 900, 0, 100), ('c7c5', 20, 0, 20), ('d7d5', 15, 0, 25), ('g7g6', 10, 0, 10)],
        position('e4 e5 Nf3'): [('b8c6', 445, 0, 445), ('g8f6', 5, 0, 5)],
        position('e4 c5 Nc3'): [('b8c6', 10, 0, 10), ('e7e6', 10, 0, 10)],
        position('e4 c5 Nc3 e6 d4'): [('c5d4', 50, 0, 50)],
        position('e4 d5 Nc3'): [('e7e6', 20, 0, 20)],
    }
    for k in g.nodes:
        if k in tables:
            rows = tables[k]
            totals = [sum(r[i] for r in rows) for i in (1, 2, 3)]
            cache_row(cache, k, data(*totals, rows))
        else:
            cache_row(cache, k, data(50, 0, 50))
    # Ensure d5 Nc3 and d5 itself have good continuation score so V_full(d5) >> DB(37.5%)
    cache_row(cache, position('e4 d5 Nc3 e6 exd5'), data(70, 0, 30))

    with Store(evaluation_store) as store:
        # 1. Only-move cliff at 1.e4 c5 2.Nc3 Nc6 (gap > 8pp) -> rescues 3.Bc4 + ancestor 2.Nc3
        store.put(
            position('e4 c5 Nc3 Nc6'),
            _eval_entry(30, [{'cp': 120, 'line': 'f1c4'}, {'cp': -50, 'line': 'd2d4'}]),
        )
        # 2. Obvious single recapture at 1.e4 c5 2.Nc3 e6 3.d4 cxd4 (huge gap for Qxd4 vs a3, must NOT rescue!)
        store.put(
            position('e4 c5 Nc3 e6 d4 cxd4'),
            _eval_entry(30, [{'cp': 50, 'line': 'd1d4'}, {'cp': -300, 'line': 'a2a3'}]),
        )
        # 3. Engine >> DB at 1.e4 d5 (DB=37.5%, engine=60% -> +22.5pp) -> rescues 2.Nc3
        store.put(position('e4 d5'), _eval_entry(30, [{'cp': 110, 'line': 'b1c3'}]))
        store.put(position('e4 d5 Nc3'), _eval_entry(30, [{'cp': 110, 'line': 'e7e6'}]))
        # 4. Inaccurate kept move 1.e4 e5 2.Nf3 (drops cp 80 -> 0) refuted by pv0=Nf6 -> rescues 3.Nxe5
        store.put(position('e4 e5'), _eval_entry(30, [{'cp': 80, 'line': 'f1c4'}]))
        store.put(position('e4 e5 Nf3'), _eval_entry(30, [{'cp': 0, 'line': 'g8f6'}]))

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
    path = tmp_path / 'white.json'
    # Goal = 12 (threshold 1/12 = 0.0833): base keeps only {start, e4 e5}.
    # Protected keeps: 1...c5 2.Nc3 (ancestor), 2...Nc6 3.Bc4 (only-move), 1...d5 2.Nc3 (engine-vs-db),
    # 2...Nf6 3.Nxe5 (refutation). Does NOT keep 3.d4 or 4.Qxd4 (trivial recapture) or 1...g6 2.d4!
    value, text = coverage.analyze(path, cache, goal=12, keep=[{'path': ['e4', 'g6']}])
    prot = value['at_goal']['protected']
    by_pos = {r['position']: r for r in prot['positions']}
    assert position('e4 c5 Nc3 Nc6') in by_pos
    assert any(r['kind'] == 'only_move' for r in by_pos[position('e4 c5 Nc3 Nc6')]['reasons'])
    assert position('e4 d5') in by_pos
    assert any(r['kind'] == 'engine_vs_db' for r in by_pos[position('e4 d5')]['reasons'])
    assert position('e4 e5 Nf3 Nf6') in by_pos
    assert any(r['kind'] == 'refutation' for r in by_pos[position('e4 e5 Nf3 Nf6')]['reasons'])
    assert position('e4 g6') in by_pos
    assert any(r['kind'] == 'pinned' for r in by_pos[position('e4 g6')]['reasons'])
    assert position('e4 c5 Nc3 e6 d4 cxd4') not in by_pos
    assert 'Qxd4' not in text and 'Bc4' in text and 'Nxe5' in text and 'd4' in text
    assert prot['ancestors_added'] >= 1  # 2.Nc3 on 1...c5 route
    assert value['validation']['trimmed_pgn_reproduces_repertoire']
    assert value['validation']['costs_reproduce_change']
    # Opting out via critical_coverage=False keeps only pinned + reach-goal positions.
    disabled, _ = coverage.analyze(path, cache, goal=12, critical=False, keep=[{'path': ['e4', 'g6']}])
    disabled_pos = {r['position'] for r in disabled['at_goal']['protected']['positions']}
    assert disabled_pos == {position('e4 g6')}


def test_build_reruns_coverage_only_when_critical_coverage_or_keep_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('Must stay offline'))
    text = '1. e4 e5 2. Nf3 Nc6 *\n\n1. d4 d5 2. c4 e6 *'
    white, black = tmp_path / 'white.pgn', tmp_path / 'black.pgn'
    white.write_text(text)
    black.write_text(text)
    cache = tmp_path / 'cache'
    cache.mkdir()
    for k, n in graph(tmp_path, text).nodes.items():
        size = 100 // max(1, len(n.edges))
        cache_row(cache, k, data(50, 0, 50, [(m, size // 2, 0, size // 2) for m in n.edges]))
    wcfg, bcfg = tmp_path / 'white.json', tmp_path / 'black.json'
    wcfg.write_text('{"policy": {}}')
    bcfg.write_text('{"policy": {}}')

    def run():
        return build.build(
            white,
            black,
            white_config=wcfg,
            black_config=bcfg,
            directory=tmp_path / 'data',
            cache=cache,
            offline=True,
        )

    run()
    score_bytes = (tmp_path / 'data' / 'white.json').read_bytes()
    wcfg.write_text(json.dumps({'policy': {}, 'critical_coverage': False, 'keep': [{'path': ['d4', 'd5']}]}))
    changed = run()
    assert {s['step'] for s in changed['steps'] if s['status'] == 'built'} == {'white.coverage', 'render'}
    assert (tmp_path / 'data' / 'white.json').read_bytes() == score_bytes
