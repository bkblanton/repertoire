import json
import re
from pathlib import Path

import httpx
import pytest
from helpers import cache_row, data, graph, position

from repertoire import build, changes, cli, evals
from repertoire.report.changes import render


def check_links(directory):
    """Every relative link on every generated Markdown page reaches an existing file and anchor."""
    pages = list(Path(directory).rglob('*.md'))
    anchors = {p.resolve(): set(re.findall(r'<a id="([^"]+)"></a>', p.read_text(encoding='utf-8'))) for p in pages}
    checked = 0
    for page in pages:
        for target in re.findall(r'\]\(([^)\s]+)\)', page.read_text(encoding='utf-8')):
            if target.startswith('https://'):
                continue
            name, _, anchor = target.partition('#')
            destination = (page.parent / name).resolve() if name else page.resolve()
            assert destination.exists(), (page, target)
            assert not anchor or anchor in anchors[destination], (page, target)
            checked += 1
    assert checked
    return pages


def _cache_all(cache, *pgn_texts, tables=None):
    cache.mkdir(exist_ok=True)
    tables = tables or {}
    nodes = set(tables)
    for text in pgn_texts:
        g = graph(cache.parent, text)
        nodes.update(g.nodes)
    for k in nodes:
        if k in tables:
            rows = tables[k]
            totals = [sum(r[i] for r in rows) for i in (1, 2, 3)] if rows else [50, 0, 50]
            cache_row(cache, k, data(*totals, rows))
        else:
            cache_row(cache, k, data(50, 0, 50))


def _make_workspace(tmp_path, monkeypatch, white_pgn, black_pgn, *, tables=None, white_config=None, black_config=None):
    monkeypatch.setattr(httpx.Client, 'request', lambda *a, **k: pytest.fail('Must stay offline'))
    white = tmp_path / 'white.pgn'
    black = tmp_path / 'black.pgn'
    white.write_text(white_pgn, encoding='utf-8')
    black.write_text(black_pgn, encoding='utf-8')
    wcfg = tmp_path / 'white.config.json'
    bcfg = tmp_path / 'black.config.json'
    wcfg.write_text(json.dumps(white_config or {'policy': {}}), encoding='utf-8')
    bcfg.write_text(json.dumps(black_config or {'policy': {}}), encoding='utf-8')
    cache = tmp_path / 'cache'
    _cache_all(cache, white_pgn, black_pgn, tables=tables)

    def run(**kwargs):
        return build.build(
            white,
            black,
            white_config=wcfg,
            black_config=bcfg,
            directory=tmp_path / 'data',
            cache=cache,
            offline=True,
            **kwargs,
        )

    return run, white, black, wcfg, bcfg, cache


CH_OPEN = (
    '[ChapterURL "https://lichess.org/study/w/open"]\n[ChapterName "Open Games"]\n\n'
    '1. e4 e5 2. Nf3 Nc6 (2... Nf6 3. Nxe5) 3. Bb5 *'
)
CH_FRENCH = (
    '[ChapterURL "https://lichess.org/study/w/french"]\n[ChapterName "French Defense"]\n\n1. e4 e6 2. d4 d5 3. Nd2 *'
)
CH_CARO = '[ChapterURL "https://lichess.org/study/w/caro"]\n[ChapterName "Caro-Kann"]\n\n1. e4 c6 2. d4 d5 3. exd5 *'
BASE_WHITE = '\n\n'.join([CH_OPEN, CH_FRENCH, CH_CARO])

BASE_BLACK = '[ChapterURL "https://lichess.org/study/b/main"]\n[ChapterName "Black Main"]\n\n1. e4 e5 2. Nf3 Nc6 *'

BASE_TABLES = {
    position(''): [('e2e4', 500, 0, 500)],
    position('e4'): [('e7e5', 250, 0, 250), ('e7e6', 150, 0, 150), ('c7c6', 100, 0, 100), ('c7c5', 100, 0, 100)],
    position('e4 e5 Nf3'): [('b8c6', 150, 0, 150), ('g8f6', 100, 0, 100)],
    position('e4 e5 Nf3 Nc6 Bb5'): [('a7a6', 90, 0, 60), ('g8f6', 60, 0, 40)],
    position('e4 e5 Nf3 Nc6 Bb5 a6 Ba4'): [('g8f6', 70, 0, 30)],
    position('e4 e5 Nf3 Nc6 Bc4'): [('f8c5', 85, 0, 15)],
    position('e4 e6 d4'): [('d7d5', 150, 0, 150)],
    position('e4 e6 d4 d5 Nd2'): [('g8f6', 90, 0, 60), ('c7c5', 60, 0, 40)],
    position('e4 e6 d4 d5 e5'): [('c7c5', 55, 0, 95)],
    position('e4 c6 d4'): [('d7d5', 100, 0, 100)],
    position('e4 c6 d4 d5 exd5'): [('c6d5', 60, 0, 40)],
    position('e4 c5 Nc3'): [('b8c6', 65, 0, 35)],
}


def test_first_build_saves_baseline_and_unchanged_rebuilds_preserve_previous_report(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    first = run()
    assert first['changes']['status'] == 'baseline'
    assert (tmp_path / 'data' / changes.SNAPSHOT_NAME).is_file()
    assert not (tmp_path / 'changes.md').exists()
    assert not (tmp_path / 'data' / 'changes.json').exists()
    assert 'changes.md' not in (tmp_path / 'summary.md').read_text(encoding='utf-8')
    assert 'changes.md' not in (tmp_path / 'report.md').read_text(encoding='utf-8')
    check_links(tmp_path)

    # Unchanged build: still no change report, render reused.
    unchanged = run()
    assert unchanged['built'] == 0
    assert unchanged['changes']['status'] == 'unchanged'
    assert not (tmp_path / 'changes.md').exists()

    # Make a real change: add a continuation line in Open Games.
    updated_white = BASE_WHITE.replace('3. Bb5 *', '3. Bb5 a6 4. Ba4 *')
    _cache_all(cache, updated_white, tables=BASE_TABLES)
    white.write_text(updated_white, encoding='utf-8')
    second = run()
    assert second['changes']['status'] == 'updated'
    assert second['changes']['changed'] is True
    changes_md = tmp_path / 'changes.md'
    changes_json = tmp_path / 'data' / 'changes.json'
    assert changes_md.is_file() and changes_json.is_file()
    first_report_bytes = changes_md.read_bytes()
    first_json_bytes = changes_json.read_bytes()
    first_md_text = first_report_bytes.decode('utf-8')
    assert '\u2014' not in first_md_text
    assert '[Changes](changes.md)' in (tmp_path / 'summary.md').read_text(encoding='utf-8')
    assert '- [Changes](changes.md)' in (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert '3... a6 4. Ba4' in first_md_text
    assert 'Black had no changes.' in first_md_text
    check_links(tmp_path)

    # 1. Reused unchanged build takes the fast path (no snapshot re-parse) and preserves previous report byte-for-byte.
    original_capture = changes.capture
    monkeypatch.setattr(changes, 'capture', lambda *a, **k: pytest.fail('Unchanged build must skip snapshot capture'))
    noop = run()
    monkeypatch.setattr(changes, 'capture', original_capture)
    assert noop['built'] == 0
    assert noop['changes']['status'] == 'unchanged'
    assert noop['changes']['preserved'] is True
    assert changes_md.read_bytes() == first_report_bytes
    assert changes_json.read_bytes() == first_json_bytes

    # 2. Forced rebuild with identical inputs must ALSO preserve previous change report.
    forced = run(force=True)
    assert forced['changes']['status'] == 'unchanged'
    assert forced['changes']['preserved'] is True
    assert changes_md.read_bytes() == first_report_bytes
    assert changes_json.read_bytes() == first_json_bytes

    # 3. Comment-only PGN edit must ALSO preserve previous change report.
    white.write_text(updated_white.replace('2. Nf3', '2. Nf3 { solid }'), encoding='utf-8')
    comment_only = run()
    assert comment_only['changes']['status'] == 'unchanged'
    assert comment_only['changes']['preserved'] is True
    assert changes_md.read_bytes() == first_report_bytes
    assert changes_json.read_bytes() == first_json_bytes

    # 4. Presentation-only code change must ALSO preserve previous change report.
    original_code = build.code_inputs
    monkeypatch.setattr(
        build,
        'code_inputs',
        lambda presentation=False: (
            dict(original_code(presentation), presentation_revision='v2') if presentation else original_code(False)
        ),
    )
    pres = run()
    assert pres['built'] == 1
    assert pres['changes']['status'] == 'unchanged'
    assert pres['changes']['preserved'] is True
    assert changes_md.read_bytes() == first_report_bytes
    assert changes_json.read_bytes() == first_json_bytes
    assert '[Changes](changes.md)' in (tmp_path / 'summary.md').read_text(encoding='utf-8')
    check_links(tmp_path)

    # 5. Another real change replaces the previous change report (only keeping the most recent one).
    monkeypatch.setattr(build, 'code_inputs', original_code)
    next_white = updated_white.replace('[ChapterName "Caro-Kann"]', '[ChapterName "Caro-Kann Exchange"]')
    white.write_text(next_white, encoding='utf-8')
    third = run()
    assert third['changes']['status'] == 'updated'
    second_md_text = changes_md.read_text(encoding='utf-8')
    assert 'Caro-Kann Exchange' in second_md_text
    assert '3... a6 4. Ba4' not in second_md_text
    # Only one change report file exists (no archives or history files).
    assert [p.name for p in tmp_path.glob('*change*')] == ['changes.md']
    assert [p.name for p in (tmp_path / 'data').glob('*change*')] == ['changes.json']
    check_links(tmp_path)


def test_chapter_additions_deletions_renames_and_reordering(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    run()

    # Delete Caro-Kann, rename French Defense -> French Tarrasch, reorder French ahead of Open Games,
    # and add a new Sicilian chapter.
    new_white = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/w/french"]\n[ChapterName "French Tarrasch"]\n\n'
            '1. e4 e6 2. d4 d5 3. Nd2 *',
            '[ChapterURL "https://lichess.org/study/w/open"]\n[ChapterName "Open Games"]\n\n'
            '1. e4 e5 2. Nf3 Nc6 (2... Nf6 3. Nxe5) 3. Bb5 *',
            '[ChapterURL "https://lichess.org/study/w/sicilian"]\n[ChapterName "Closed Sicilian"]\n\n'
            '1. e4 c5 2. Nc3 Nc6 3. g3 *',
        ]
    )
    _cache_all(cache, new_white, tables=BASE_TABLES)
    white.write_text(new_white, encoding='utf-8')
    run()

    diff = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    w = diff['colors']['white']
    assert [c['id'] for c in w['chapters']['added']] == ['sicilian']
    assert [c['id'] for c in w['chapters']['deleted']] == ['caro']
    assert [(c['id'], c['old_name'], c['new_name']) for c in w['chapters']['renamed']] == [
        ('french', 'French Defense', 'French Tarrasch')
    ]
    assert w['chapters']['report_order']['reordered'] is True
    assert len(w['chapters']['report_order']['moved']) == 1
    assert w['lines'] == []  # Added/deleted chapters are reported under Chapters, not spammed in Line changes.

    md = (tmp_path / 'changes.md').read_text(encoding='utf-8')
    assert '**Added chapters (1)**' in md and 'Closed Sicilian' in md
    assert '**Deleted chapters (1)**' in md and 'Caro-Kann' in md
    assert '**Renamed chapters (1)**' in md and 'French Defense' in md and 'French Tarrasch' in md
    assert '**Reordered chapters (1 moved)**' in md
    assert '\u2014' not in md
    check_links(tmp_path)


def test_inserting_chapter_without_reordering_surviving_chapters_is_not_flagged_as_reorder(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    run()

    # Insert Sicilian at the top without changing the relative order of Open, French, Caro.
    new_white = (
        '[ChapterURL "https://lichess.org/study/w/sicilian"]\n[ChapterName "Closed Sicilian"]\n\n'
        '1. e4 c5 2. Nc3 Nc6 3. g3 *\n\n' + BASE_WHITE
    )
    _cache_all(cache, new_white, tables=BASE_TABLES)
    white.write_text(new_white, encoding='utf-8')
    run()

    diff = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    order = diff['colors']['white']['chapters']['report_order']
    assert order['reordered'] is False
    assert order['moved'] == []
    assert len(order['shifted']) == 3
    md = (tmp_path / 'changes.md').read_text(encoding='utf-8')
    assert 'Reordered chapters' not in md
    assert 'kept their relative order and shifted numbering' in md
    check_links(tmp_path)


def test_study_order_reorder_detected_even_under_opening_chapter_order(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path,
        monkeypatch,
        BASE_WHITE,
        BASE_BLACK,
        tables=BASE_TABLES,
        white_config={'policy': {}, 'chapter_order': 'opening'},
    )
    run()

    # Reorder chapters in the study PGN; opening order stays [open, french, caro] because reaches didn't change.
    swapped = '\n\n'.join([CH_CARO, CH_FRENCH, CH_OPEN])
    white.write_text(swapped, encoding='utf-8')
    res = run()
    assert res['changes']['status'] == 'updated'

    diff = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    w = diff['colors']['white']
    assert w['chapters']['report_order']['reordered'] is False
    assert w['chapters']['study_order']['reordered'] is True
    # Minimum-displacement LCS keeps French (#2 -> #2) fixed and marks Caro (#3 -> #1) and Open (#1 -> #3) moved.
    assert [m['id'] for m in w['chapters']['study_order']['moved']] == ['caro', 'open']
    md = (tmp_path / 'changes.md').read_text(encoding='utf-8')
    assert 'Study chapter order changed' in md
    assert 'report chapter numbering (`opening` order) was unaffected' in md
    check_links(tmp_path)


def test_line_additions_deletions_variation_reordering_and_moved_lines(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    run()

    # In Open Games:
    # - Reorder opponent variations at 2.Nf3: promote 2...Nf6 ahead of 2...Nc6 (mainline changed!)
    # - Add a new continuation 3.Bb5 a6 4.Ba4 (new to repertoire)
    # - Delete 3.Nxe5 after 2...Nf6 (removed from repertoire)
    # Also move a shared line: add 3.Nd2 c5 in Caro (no, add 2...Nf6 3.Nxe5 into French? Better: add 3.Nd2 Nf6 in French
    # and also copy 3.exd5 cxd5 into Open? Let's test a genuine move across chapters on a shared branch!)
    new_white = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/w/open"]\n[ChapterName "Open Games"]\n\n'
            '1. e4 e5 2. Nf3 Nf6 (2... Nc6 3. Bb5 a6 4. Ba4) *',
            '[ChapterURL "https://lichess.org/study/w/french"]\n[ChapterName "French Defense"]\n\n'
            '1. e4 e6 2. d4 d5 3. Nd2 Nf6 (3... c5 4. exd5) *',
            '[ChapterURL "https://lichess.org/study/w/caro"]\n[ChapterName "Caro-Kann"]\n\n1. e4 c6 2. d4 d5 3. exd5 *',
        ]
    )
    _cache_all(cache, new_white, tables=BASE_TABLES)
    white.write_text(new_white, encoding='utf-8')
    run()

    diff = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    w = diff['colors']['white']
    by_id = {ch['id']: ch for ch in w['lines']}
    assert set(by_id) == {'open', 'french'}

    open_diff = by_id['open']
    assert len(open_diff['added']) == 1
    assert open_diff['added'][0]['continuation'] == '3...a6 4.Ba4'
    assert open_diff['added'][0]['scope'] == 'new'
    assert len(open_diff['deleted']) == 1
    assert open_diff['deleted'][0]['continuation'] == '3.Nxe5'
    assert open_diff['deleted'][0]['scope'] == 'removed'
    assert len(open_diff['reordered']) == 1
    reorder = open_diff['reordered'][0]
    assert reorder['mainline_changed'] is True
    assert reorder['old_order'] == ['2...Nc6', '2...Nf6']
    assert reorder['new_order'] == ['2...Nf6', '2...Nc6']

    french_diff = by_id['french']
    assert len(french_diff['added']) == 2
    assert french_diff['added_moves'] == 3  # Nf6 + c5 + exd5

    md = (tmp_path / 'changes.md').read_text(encoding='utf-8')
    assert '### Line changes' in md
    assert 'Reordered variations (1)' in md
    assert 'Main line changed (2... Nc6 → 2... Nf6)' in md
    assert '3... a6 4. Ba4' in md
    assert '3. Nxe5' in md
    check_links(tmp_path)

    # Move a line between chapters: remove 3...c5 4.exd5 from French and add it to Caro.
    moved_white = '\n\n'.join(
        [
            '[ChapterURL "https://lichess.org/study/w/open"]\n[ChapterName "Open Games"]\n\n'
            '1. e4 e5 2. Nf3 Nf6 (2... Nc6 3. Bb5 a6 4. Ba4) *',
            '[ChapterURL "https://lichess.org/study/w/french"]\n[ChapterName "French Defense"]\n\n'
            '1. e4 e6 2. d4 d5 3. Nd2 Nf6 *',
            '[ChapterURL "https://lichess.org/study/w/caro"]\n[ChapterName "Caro-Kann"]\n\n'
            '1. e4 c6 (1... e6 2. d4 d5 3. Nd2 c5 4. exd5) 2. d4 d5 3. exd5 *',
        ]
    )
    white.write_text(moved_white, encoding='utf-8')
    run()
    diff2 = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    by_id2 = {ch['id']: ch for ch in diff2['colors']['white']['lines']}
    assert by_id2['french']['deleted'][0]['scope'] == 'moved'
    assert by_id2['french']['deleted'][0]['related_chapters'] == ['caro']
    assert by_id2['caro']['added'][0]['scope'] == 'moved'
    assert by_id2['caro']['added'][0]['related_chapters'] == ['french']
    md2 = (tmp_path / 'changes.md').read_text(encoding='utf-8')
    assert 'Moved to [W3](chapters/W3.md)' in md2
    assert 'Moved from [W2](chapters/W2.md)' in md2
    check_links(tmp_path)


def test_move_selection_policy_and_config_changes(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    run()

    # Add a higher-scoring side variation 3.Bc4 in Open Games and change coverage_goal + chapter_order.
    competing_white = BASE_WHITE.replace('3. Bb5 *', '3. Bb5 (3. Bc4) *')
    _cache_all(cache, competing_white, tables=BASE_TABLES)
    white.write_text(competing_white, encoding='utf-8')
    wcfg.write_text(json.dumps({'policy': {}, 'coverage_goal': 100, 'chapter_order': 'opening'}), encoding='utf-8')
    run()

    diff = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    w = diff['colors']['white']
    assert any(c['key'] == 'coverage_goal' and c['old'] == 300 and c['new'] == 100 for c in w['config'])
    assert any(c['key'] == 'chapter_order' and c['old'] == 'study' and c['new'] == 'opening' for c in w['config'])
    assert len(w['played_moves']) == 1
    pm = w['played_moves'][0]
    assert pm['old_label'] == 'Bb5' and pm['new_label'] == 'Bc4'
    assert pm['reason'] == 'Competing alternative won'
    assert len(w['alternatives']['added']) == 1

    md = (tmp_path / 'changes.md').read_text(encoding='utf-8')
    assert '### Move selection and alternatives' in md
    assert 'Competing alternative won' in md
    assert '1 in 300 games → 1 in 100 games' in md
    check_links(tmp_path)

    # Override the competing position explicitly back to 3.Bb5 via config policy.
    wcfg.write_text(
        json.dumps(
            {
                'policy': {position('e4 e5 Nf3 Nc6'): 'f1b5'},
                'coverage_goal': 100,
                'chapter_order': 'opening',
            }
        ),
        encoding='utf-8',
    )
    run()
    diff2 = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    w2 = diff2['colors']['white']
    assert any(c['key'] == 'policy' for c in w2['config'])
    assert w2['played_moves'][0]['old_label'] == 'Bc4'
    assert w2['played_moves'][0]['new_label'] == 'Bb5'
    assert w2['played_moves'][0]['reason'] == 'Policy override added'
    check_links(tmp_path)


def test_failed_build_preserves_snapshot_and_previous_change_report_until_resume(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    run()

    # Build 2 creates a change report.
    white_v2 = BASE_WHITE.replace('3. Bb5 *', '3. Bb5 a6 *')
    _cache_all(cache, white_v2, tables=BASE_TABLES)
    white.write_text(white_v2, encoding='utf-8')
    run()
    prev_md = (tmp_path / 'changes.md').read_bytes()
    prev_snapshot = (tmp_path / 'data' / changes.SNAPSHOT_NAME).read_bytes()

    # Build 3 edits French, then fails during preparation after scoring already rewrote white.json.
    white_v3 = white_v2.replace('3. Nd2 *', '3. Nd2 Nf6 *')
    _cache_all(cache, white_v3, tables=BASE_TABLES)
    white.write_text(white_v3, encoding='utf-8')
    original_prep = build.preparation.analyze
    monkeypatch.setattr(
        build.preparation, 'analyze', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('mid-build failure'))
    )
    with pytest.raises(RuntimeError, match='mid-build failure'):
        run()
    # Both the previous change report and the previous baseline snapshot must still be intact.
    assert (tmp_path / 'changes.md').read_bytes() == prev_md
    assert (tmp_path / 'data' / changes.SNAPSHOT_NAME).read_bytes() == prev_snapshot

    # Resume Build 3 -> now completes and diffs v3 against v2!
    monkeypatch.setattr(build.preparation, 'analyze', original_prep)
    resumed = run()
    assert resumed['changes']['status'] == 'updated'
    diff = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    by_id = {ch['id']: ch for ch in diff['colors']['white']['lines']}
    assert list(by_id) == ['french']
    assert by_id['french']['added'][0]['continuation'] == '3...Nf6'
    check_links(tmp_path)


def test_engine_evaluation_import_triggers_change_report(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    run()

    # Populate the isolated test evaluation store without changing any PGN or config.
    store = evals.Store(evals.STORE)
    try:
        store.put(
            position('e4 e5 Nf3 Nc6 Bb5'),
            evals.Evaluation(
                depth=30,
                knodes=500,
                pvs=[{'cp': 35, 'line': 'a7a6'}],
                source='export',
                retrieved_at='2026-11-26',
            ),
        )
        store.commit()
    finally:
        store.close()

    res = run()
    assert res['changes']['status'] == 'updated'
    diff = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    assert diff['colors']['white']['engine']['changed'] is True
    assert diff['colors']['white']['engine']['after']['resolved'] is True
    md = (tmp_path / 'changes.md').read_text(encoding='utf-8')
    assert '**Engine evaluations:** imported' in md
    check_links(tmp_path)


def test_explorer_evidence_refresh_triggers_change_report(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    run()

    # Refresh one cached Explorer table with a newer timestamp and updated counts.
    info = cache_row(cache, position('e4 c6 d4 d5 exd5'), data(70, 0, 30, [('c6d5', 70, 0, 30)]))
    cached_file = cache / f"{info['cache_key']}.json"
    payload = json.loads(cached_file.read_text(encoding='utf-8'))
    payload['retrieved_at'] = '2026-11-26'
    cached_file.write_text(json.dumps(payload), encoding='utf-8')

    res = run()
    assert res['changes']['status'] == 'updated'
    diff = json.loads((tmp_path / 'data' / 'changes.json').read_text(encoding='utf-8'))
    assert diff['colors']['white']['evidence']['changed'] is True
    assert diff['colors']['white']['evidence']['refreshed_positions'] == 1
    md = (tmp_path / 'changes.md').read_text(encoding='utf-8')
    assert '**Explorer evidence:** refreshed for 1 position' in md
    check_links(tmp_path)


def test_bootstrap_snapshot_captures_existing_build_before_overwrite(tmp_path, monkeypatch):
    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    run()
    # Simulate an upgraded workspace that already has a completed build but no .build-snapshot.json yet.
    snap_file = tmp_path / 'data' / changes.SNAPSHOT_NAME
    snap_file.unlink()
    assert not snap_file.exists()

    bootstrapped = changes.bootstrap_snapshot(
        tmp_path / 'data', dict(white=white, black=black), dict(white=wcfg, black=bcfg)
    )
    assert bootstrapped is True
    assert snap_file.is_file()
    # Second bootstrap call is a no-op since snapshot now exists.
    assert (
        changes.bootstrap_snapshot(tmp_path / 'data', dict(white=white, black=black), dict(white=wcfg, black=bcfg))
        is False
    )

    # Now edit white.pgn and build -> immediately detects the change against the bootstrapped snapshot!
    white.write_text(BASE_WHITE.replace('3. exd5 *', '3. exd5 cxd5 *'), encoding='utf-8')
    _cache_all(cache, white.read_text(encoding='utf-8'), tables=BASE_TABLES)
    res = run()
    assert res['changes']['status'] == 'updated'
    assert '3... cxd5' in (tmp_path / 'changes.md').read_text(encoding='utf-8')
    check_links(tmp_path)


def test_changes_cli_shows_summary_and_rerenders(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    cli.main(['changes', '--directory', str(data_dir)])
    assert 'No change report yet' in capsys.readouterr().out

    run, white, black, wcfg, bcfg, cache = _make_workspace(
        tmp_path, monkeypatch, BASE_WHITE, BASE_BLACK, tables=BASE_TABLES
    )
    run()
    white.write_text(BASE_WHITE.replace('3. exd5 *', '3. exd5 cxd5 *'), encoding='utf-8')
    _cache_all(cache, white.read_text(encoding='utf-8'), tables=BASE_TABLES)
    run()

    md_path = tmp_path / 'changes.md'
    expected = md_path.read_text(encoding='utf-8')
    md_path.unlink()
    cli.main(['changes', '--directory', str(data_dir), '--render'])
    out = capsys.readouterr().out
    assert 'White:' in out and '1 line added' in out
    assert md_path.read_text(encoding='utf-8') == expected
    assert render(json.loads((data_dir / 'changes.json').read_text(encoding='utf-8'))) == expected
