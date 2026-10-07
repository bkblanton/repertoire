import json
from pathlib import Path

import chess
import httpx
import pytest
from helpers import cache_row, data, position

from repertoire import build, render
from repertoire.board_cache import geometry, owner_outcome
from repertoire.character import board_fingerprint
from repertoire.explorer import Explorer
from repertoire.graph import Graph, Node, key, parse, resolve
from repertoire.preparation import chess_facts


def test_shared_geometry_preserves_legal_moves_terminals_and_graph_membership():
    boards = [
        chess.Board(),
        chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'),
        chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),
        chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1'),
        chess.Board('7k/5K2/6Q1/8/8/8/8/8 b - - 0 1'),
    ]
    for board in boards:
        k = key(board)
        expected = {}
        for move in board.legal_moves:
            child = board.copy()
            child.push(move)
            expected[move.uci()] = key(child)
        assert dict(geometry(k).moves) == expected
        terminal = (
            float(board.turn != chess.WHITE)
            if board.is_checkmate()
            else 0.5
            if board.is_stalemate() or board.is_insufficient_material()
            else None
        )
        assert owner_outcome(k, chess.WHITE) == terminal
        assert owner_outcome(k, chess.BLACK) == (None if terminal is None else 1 - terminal)
    root, after = position('e4'), position('e4 e5')
    lone = Graph({root: Node(root + ' 0 1', [])}, [], [root])
    joined = Graph(dict(lone.nodes, **{after: Node(after + ' 0 1', [])}), [], [root])
    assert resolve(lone, chess.WHITE, {})[root] == {}
    assert resolve(joined, chess.WHITE, {})[root]['e7e5'][0] == after
    evidence = {root: data(50, 0, 50, [('e7e5', 50, 0, 50)])}
    assert chess_facts(lone, chess.WHITE, evidence)[root]['possible_targets'] == set()
    assert after in chess_facts(joined, chess.WHITE, evidence)[root]['possible_targets']
    first = board_fingerprint(root, chess.WHITE)
    first[0]['queens'] = 'mutated'
    first[1]['own_isolated'] = 'mutated'
    second = board_fingerprint(root, chess.WHITE)
    assert second[0]['queens'] == 'both queens'
    assert second[1].get('own_isolated') != 'mutated'


def test_deferred_render_is_scoped_and_standalone_remains_eager(monkeypatch):
    calls = []
    monkeypatch.setattr(render, 'load_report', lambda p: calls.append(p) or {'color': 'white'})
    with render.defer_report_outputs():
        render.update_report_outputs('not-created.json')
    assert calls == []
    # Outside the context, it must attempt to load the result as before.
    monkeypatch.setattr(render, 'load_report', lambda p: (_ for _ in ()).throw(ValueError('eager')))
    with pytest.raises(ValueError, match='eager'):
        render.update_report_outputs('not-created.json')


def test_checkpoints_validate_bytes_and_recover_from_failure(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.write_text('one')
    runner = build.Builder(tmp_path / 'state.json')
    count = []

    def action():
        count.append(1)
        output.write_text(source.read_text())

    def step():
        runner.step('work', lambda: build.file_inputs([source]), [output], action)

    step()
    step()
    assert len(count) == 1
    output.write_text('corrupt')
    step()
    source.write_text('two')
    step()
    assert len(count) == 3

    def fail():
        output.write_text('partial')
        raise RuntimeError('interrupted')

    runner.force = True
    with pytest.raises(RuntimeError, match='interrupted'):
        runner.step('work', lambda: build.file_inputs([source]), [output], fail)
    assert 'work' not in json.loads(runner.path.read_text())['steps']
    resumed = build.Builder(runner.path)
    resumed.step('work', lambda: build.file_inputs([source]), [output], action)
    assert len(count) == 4 and output.read_text() == 'two'


def test_cache_dependencies_include_misses_but_ignore_unrelated_tables(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    parent, missing = position('e4'), position('e4 e5')
    cache_row(cache, parent, data(50, 0, 50))
    runner = build.Builder(tmp_path / 'state.json')
    calls, output = [], tmp_path / 'out'

    def action():
        calls.append(1)
        explorer = Explorer(cache, offline=True)
        try:
            explorer.get(parent)
            try:
                explorer.get(missing)
            except ValueError as exc:
                assert str(exc).startswith('Offline cache miss:')
        finally:
            explorer.close()
        output.write_text('done')

    def step():
        runner.step('work', lambda: {}, [output], action)

    step()
    step()
    cache_row(cache, position('d4'), data(50, 0, 50))
    step()
    assert len(calls) == 1
    cache_row(cache, missing, data(50, 0, 50))
    step()
    cache_row(cache, parent, data(60, 0, 40))
    step()
    assert len(calls) == 3


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('Batch must stay offline'))
    text = (
        '[ChapterURL "https://lichess.org/study/test/a"]\n[ChapterName "King pawn"]\n\n'
        '1. e4 e5 2. Nf3 Nc6 *\n\n'
        '[ChapterURL "https://lichess.org/study/test/b"]\n[ChapterName "Queen pawn"]\n\n'
        '1. d4 d5 2. Nf3 Nf6 *'
    )
    white, black = tmp_path / 'white.pgn', tmp_path / 'black.pgn'
    white.write_text(text)
    black.write_text(text)
    cache = tmp_path / 'cache'
    cache.mkdir()
    for k, n in parse(white).nodes.items():
        size = 100 // max(1, len(n.edges))
        row = data(50, 0, 50, [(m, size // 2, 0, size // 2) for m in n.edges])
        cache_row(cache, k, row)
    renders = []
    original = build.generate

    def counted(*args, **kwargs):
        renders.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(build, 'generate', counted)

    def run(**options):
        return build.build(
            white,
            black,
            white_config=None,
            black_config=None,
            directory=tmp_path / 'data',
            cache=cache,
            offline=True,
            **options,
        )

    return run, white, black, renders


def test_batch_renders_once_and_reuses_unchanged_color(batch):
    run, white, black, renders = batch
    first = run()
    assert first['built'] == 17 and renders == [1]
    data_dir = white.parent / 'data'
    assert (data_dir / 'opponent-rating-score-correlation.json').is_file()
    assert (data_dir / 'prepared-depth-gain-correlation.json').is_file()
    assert '\n## Correlations\n' in (white.parent / 'report.md').read_text(encoding='utf-8')
    assert '\n### Future preparation gain\n' in (white.parent / 'report.md').read_text(encoding='utf-8')
    assert '\n### Opponent rating and score improvement\n' in (white.parent / 'report.md').read_text(encoding='utf-8')
    old = build.file_inputs(data_dir.glob('white*.json'))
    second = run()
    assert second['built'] == 0 and second['reused'] == 17 and renders == [1]
    black.write_text(black.read_text().replace('e4 e5', 'e4 {new comment} e5'))
    changed = run()
    assert changed['reused'] == 7 and changed['built'] == 10 and renders == [1, 1]
    assert build.file_inputs(data_dir.glob('white*.json')) == old
    forced = run(force=True)
    assert forced['built'] == 17 and renders == [1, 1, 1]


def test_failed_batch_preserves_readable_reports_and_resumes(batch, monkeypatch):
    run, white, black, renders = batch
    full = white.parent / 'report.md'
    full.write_text('previous successful report')
    original = build.preparation.analyze
    monkeypatch.setattr(
        build.preparation, 'analyze', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('failed prep'))
    )
    with pytest.raises(RuntimeError, match='failed prep'):
        run()
    assert renders == [] and full.read_text() == 'previous successful report'
    monkeypatch.setattr(build.preparation, 'analyze', original)
    resumed = run()
    assert resumed['reused'] == 4 and resumed['built'] == 13 and renders == [1]


def test_presentation_change_only_renders_and_missing_analysis_rebuilds_dependants(batch, monkeypatch):
    run, white, black, renders = batch
    run()
    original = build.code_inputs
    monkeypatch.setattr(
        build,
        'code_inputs',
        lambda presentation=False: (
            dict(original(presentation), presentation_revision='new') if presentation else original(False)
        ),
    )
    changed = run()
    assert changed['built'] == 1 and changed['reused'] == 16
    assert changed['steps'][-1]['step'] == 'render' and renders == [1, 1]
    (black.parent / 'data' / 'black.character.json').unlink()
    recovered = run()
    assert {s['step'] for s in recovered['steps'] if s['status'] == 'built'} == {
        'black.character',
        'black.ratings',
        'black.insights',
        'rating-correlations',
        'render',
    }


def test_rendering_code_invalidates_only_the_render_step():
    analysis, presentation = build.code_inputs(), build.code_inputs(presentation=True)
    report_files = {p for p in presentation if Path(p).parent.name == 'report'}
    assert any(p.endswith('pages.py') for p in report_files)
    assert not report_files & set(analysis)
    assert any(p.endswith('render.py') for p in presentation) and not any(p.endswith('render.py') for p in analysis)
    assert any(p.endswith('uncertainty.py') for p in analysis)
