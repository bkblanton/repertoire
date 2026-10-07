import json
import re
from pathlib import Path

import chess
import pytest
from helpers import check_score_tables, run_fixture

from repertoire_score.character import analyze as character
from repertoire_score.report.bundle import load
from repertoire_score.report.derive import combined_overall, exit_points, non_sparse_rows, position_contributions
from repertoire_score.report.format import (
    centipawn_delta,
    centipawn_equivalent,
    cp,
    cp_change,
    display,
    elo_equivalent,
    headline_delta,
    line,
    per_thousand,
    spread_display,
)
from repertoire_score.report.generate import generate, page_names
from repertoire_score.report.links import Chapters
from repertoire_score.report.markdown import compressed_columns, report_navigation, table
from repertoire_score.report.pages import summary_report
from repertoire_score.report.sections import (
    common_positions_for_scope,
    common_positions_section,
    exits_section,
    position_tree,
    strengths_section,
    vulnerabilities_section,
)


def chapter_pages(directory, report):
    """Concatenated chapter pages, in chapter order."""
    prefix = report['color'][0].upper()
    return [
        (Path(directory) / 'chapters' / f'{prefix}{i}.md').read_text(encoding='utf-8')
        for i in range(1, len(report['chapters']) + 1)
    ]


def check_links(directory):
    """Every relative link on every generated page reaches an existing file and anchor."""
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


def test_full_and_summary_preserve_metrics_sources_and_separate_reply_tables(complete):
    path, report = complete
    originals = {p: p.read_bytes() for p in path.parent.glob('*.json')}
    bundles = generate([path])
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    chapters = chapter_pages(path.parent, report)
    prefix = f'| White | 50.00% | 40.00% | {headline_delta(0.4, 0.5)} |'
    assert prefix in full and headline_delta(0.4, 0.5) == f'-10.00% ({centipawn_delta(0.4, 0.5):+.0f} cp)'
    assert '| White | 50.0% | 40.0% | -10.0% (-110 cp) | -70 |' in summary
    assert not re.search(r'^\| Black \|', full, flags=re.M) and '[Black repertoire](#black)' not in full
    assert prefix + ' -70.4 |' in full
    assert '[Combined repertoire](#combined)' not in full
    assert 'Chapter reach (incl. transpositions)' in summary and 'Overall-policy reach' in summary
    assert 'Tarrasch Nf6' in summary and '**(alternative)**' in summary
    assert 'Prepared opponent replies' in '\n'.join([full, *chapters])
    assert '**Unprepared opponent replies**' not in '\n'.join([full, *chapters])  # no positive unprepared drag here
    assert 'No resolved, reachable results in this category.' not in full
    assert '### Strengths' in full
    assert 'Strongest and weakest stopping outcomes' not in full
    assert 'Preparation, replies, and resulting positions' in full
    assert 'Most frequently used own decisions' not in full
    assert all('## Most common positions' in page and '## Where preparation ends' in page for page in chapters)
    assert 'Chapter-by-chapter analysis' not in full and '#### W1.' not in full
    assert 'Optional memorization and trimming analysis' not in full
    assert 'Trim gain (pp)' not in full
    assert 'optional trimming' not in summary.lower()
    assert all('Exact first-entry positions' in page for page in chapters)
    assert 'Posterior mean' not in full and 'Empirical score' not in full
    everything = '\n'.join([full, summary, *chapters])
    assert 'Study description' not in everything and '\u2014' not in everything
    for rendered in (full, summary, *chapters):
        assert 'Most common opening source' in rendered and 'Unclassified' in rendered
        assert 'Unclassified (100.00%)' not in rendered and 'Unclassified (100.0%)' not in rendered
        for header in rendered.splitlines():
            if header.startswith('| ') and any(
                label in header for label in ('Chapter source', 'Chapter context', 'Entry-position sources')
            ):
                assert 'Most common opening source' in header
    assert not bundles[0]['unavailable'] and bundles[0]['current_source']
    for p, content in originals.items():
        assert p.read_bytes() == content
    anchors = set(re.findall(r'<a id="([^"]+)"', full))
    assert all(target in anchors for target in re.findall(r'\]\(#([^)]+)\)', full))
    assert all(target in anchors for target in re.findall(r'\]\(report.md#([^)]+)\)', summary))
    assert '](chapters/W1.md)' in full + summary and '](https://lichess.org/analysis/standard/' in full + summary
    check_links(path.parent)
    assert full.count('<details>') == full.count('</details>')
    assert not list(path.parent.glob('*.study-description.md'))


def test_equivalent_gap_reach_in_overall_and_chapter_reports(complete):
    from repertoire_score.report.format import gap_percentage

    path, report = complete
    bundle = generate([path])[0]
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    overall = bundle['report']['gap_coverage']
    assert '### Equivalent gap reach' in full
    with display(digits=1):
        assert f'Equivalent gap reach **{gap_percentage(overall)}**' in summary
    for text in (full, summary):
        assert 'Weighted gap reach contribution' in text
        assert 'Equivalent gap reach after entry' in text
    for chapter, page in zip(bundle['report']['chapters'], chapter_pages(path.parent, report)):
        metrics = chapter['gap_coverage']
        assert metrics
        if metrics['entry_probability'] is not None:
            assert metrics['weighted_equivalent_gap_reach_bounds'] == pytest.approx(
                [metrics['entry_probability'] * p for p in metrics['equivalent_gap_reach_bounds']]
            )
        assert f"weighted gap reach contribution **{gap_percentage(metrics, weighted=True)}**" in page
        assert '## Equivalent gap reach' in page
    assert 'Chapters overlap, so their values are not additive' in full


def test_gap_display_keeps_unknown_reach_bounded():
    from repertoire_score.report.format import gap_percentage

    assert gap_percentage(None) == 'unavailable'
    assert gap_percentage(dict(equivalent_gap_reach=0.0)) == '0.00%'
    assert gap_percentage(dict(equivalent_gap_reach_bounds=[0.1, 0.2])) == '10.00% to 20.00% (bounds)'
    assert (
        gap_percentage(dict(weighted_equivalent_gap_reach_bounds=[0.02, 0.04]), weighted=True)
        == '2.00% to 4.00% (bounds)'
    )


def test_recursive_sharpness_in_overall_and_every_chapter(complete):
    path, report = complete
    bundle = generate([path])[0]
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    assert '| Score spread |' in full and '| Score spread |' in summary
    assert '| Sharpness |' not in full + summary
    assert '**Outcome volatility:**' in full
    # Volatility barely varies between scopes, so it stays in spread breakdowns rather than headlines.
    assert 'Outcome volatility' not in summary and 'outcome volatility **' not in full
    pages = chapter_pages(path.parent, report)
    assert all('outcome volatility **' not in page and 'Outcome volatility (0%-100%)' in page for page in pages)
    scopes = {s['id']: s for s in bundle['character']['scopes']}
    for sid, score in [
        ('overall', bundle['report']['overall']),
        *[(c['id'], c['score']) for c in bundle['report']['chapters']],
    ]:
        outcomes = score['outcomes']
        assert outcomes == scopes[sid]['outcomes']
        assert outcomes['win_probability'] + outcomes['draw_probability'] / 2 == pytest.approx(
            score['raw_empirical_score']
        )


@pytest.mark.parametrize('field', ['report_sha256', 'input_sha256', 'filters', 'color'])
def test_mismatched_companion_rejected_before_any_output_changes(complete, field):
    path, report = complete
    generate([path])
    target = path.parent / 'report.md'
    before = target.read_bytes()
    companion = path.with_suffix('.character.json')
    data = json.loads(companion.read_text())
    if field == 'color':
        data[field] = 'black'
    else:
        data['manifest'][field] = 'wrong'
    companion.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='different score snapshot'):
        generate([path])
    assert target.read_bytes() == before
    generate([path], strict=False)
    after = target.read_text(encoding='utf-8')
    assert 'character: belongs to a different score snapshot' in after
    assert 'effective opponent replies' not in after


def test_files_from_another_schema_version_are_rejected(complete):
    path, _ = complete
    companion = path.with_suffix('.openings.json')
    data = json.loads(companion.read_text())
    data['manifest']['schema_version'] = 0
    companion.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='different program version'):
        generate([path])
    generate([path], strict=False)
    assert 'openings: written by a different program version' in (path.parent / 'report.md').read_text(encoding='utf-8')
    report = json.loads(path.read_text())
    report['manifest']['schema_version'] = 0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='different program version'):
        generate([path], strict=False)


def test_saved_snapshot_and_missing_analyses_are_explicit(tmp_path, monkeypatch):
    report, _ = run_fixture(tmp_path, monkeypatch, common_entry=True)
    path = tmp_path / 'white.json'
    Path(report['manifest']['input_path']).write_text('changed PGN')
    generate([path])
    full = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert 'snapshot notice' in full and 'not the current PGN' in full
    assert 'vulnerabilities: not generated' in full
    with pytest.raises(ValueError, match='incomplete'):
        generate([path], require_complete=True)


def test_stale_correlations_rejected_and_matching_estimates_shown(complete):
    from repertoire_score.position_correlations import analyze

    path, _ = complete
    correlations = path.parent / 'prepared-depth-gain-correlation.json'
    analyze([path], correlations, path.parent / 'cache')
    generate([path], require_complete=True)
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    assert 'point estimates from the observed database counts' in full and '| Rank correlation |' in full
    assert '95% interval' not in full.split('### Future preparation gain', 1)[1].split('## Definitions', 1)[0]
    assert '[Correlations](#correlations)' in full
    assert '[Future preparation gain](#preparation-correlation)' in full
    assert 'Both main analyses are reach-weighted' in full
    assert 'Prepared depth and score improvement' not in full
    data = json.loads(correlations.read_text())
    original = dict(data['provenance']['white'])
    for key in ('report_sha256', 'vulnerabilities_sha256', 'prior', 'sparse_threshold'):
        data['provenance']['white'][key] = 'wrong'
        correlations.write_text(json.dumps(data))
        with pytest.raises(ValueError, match='different score snapshots'):
            generate([path])
        data['provenance']['white'][key] = original[key]
    generate([path], strict=False)
    assert (
        'Correlation analysis unavailable: belongs to different score snapshots'
        in (path.parent / 'report.md').read_text()
    )


def test_rating_correlations_render_beside_depth_and_reject_stale_inputs(complete):
    from repertoire_score import rating_correlations

    path, _ = complete
    result = rating_correlations.analyze([path])
    rows = [
        dict(position=str(i), rating=x + 400 * i, score=y + 0.1 * i, weight=w)
        for i in range(4)
        for x, y, w in ((1500, 0.6, 1), (1600, 0.55, 2), (1700, 0.5, 1))
    ]
    result['results']['white']['reply_associations']['all'] = rating_correlations.within_parent(rows)
    target = path.parent / 'opponent-rating-score-correlation.json'
    target.write_text(json.dumps(result), encoding='utf-8')
    generate([path])
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    assert (
        full.index('\n## Correlations\n')
        < full.index('\n### Future preparation gain\n')
        < full.index('\n### Opponent rating and score improvement\n')
        < full.index('\n## Definitions and evidence\n')
    )
    assert '[Opponent rating and score improvement](#rating-correlations)' in full
    assert '| -1.000 | -5.000% |' in full
    assert 'reach-weighted mean continuation score' in full and 'These are point estimates.' in full
    assert '<summary>Prepared replies, unprepared replies, and larger samples</summary>' in full
    originals = {name: (path.parent / name).read_bytes() for name in ('report.md', 'summary.md')}
    hashes = result['results']['white']['provenance']['hashes']
    for family in ('score', 'ratings', 'vulnerabilities'):
        original = hashes[family]
        hashes[family] = 'wrong'
        target.write_text(json.dumps(result), encoding='utf-8')
        with pytest.raises(ValueError, match='rating correlations belong to different analysis snapshots'):
            generate([path])
        assert all((path.parent / name).read_bytes() == value for name, value in originals.items())
        hashes[family] = original
    generate([path], strict=False)
    assert (
        'Rating correlation analysis unavailable: belongs to different analysis snapshots'
        in (path.parent / 'report.md').read_text()
    )


def test_compact_chapter_sources_and_readable_notation(complete):
    _, report = complete
    refs = Chapters(report)
    row = {'chapter_attribution': {'source_ids': [c['id'] for c in report['chapters']]}}
    assert refs.sources(row) == '[All 3 White chapters](#white-chapters)'
    row = {'chapter_attribution': {'context_ids': ['advance']}}
    assert refs.sources(row) == 'Unprepared; context: [W1](#white-chapter-1)'
    row = {'chapter_attribution': {'transposition_ids': ['split']}}
    assert refs.sources(row) == 'Transposition: [W3](#white-chapter-3)'
    assert line('1.e4 1...e6 2.d4 2...d5 3.Nd2') == '1. e4 e6 2. d4 d5 3. Nd2'
    assert line('1...c5') == '1... c5'
    assert line('A|B <tag>') == 'A&#124;B &lt;tag&gt;'


def test_destinations_and_relative_links(complete):
    path, _ = complete
    full = path.parent / 'full' / 'all.md'
    summary = path.parent / 'short' / 'short.md'
    generate([path], full, summary)
    assert '[Summary](../short/short.md)' in full.read_text()
    assert '[Scores](../white.json)' in full.read_text()
    assert '[Complete report](../full/all.md)' in summary.read_text()
    assert '../full/chapters/W1.md' in summary.read_text() and '../full/all.md#def-' in summary.read_text()
    page = (full.parent / 'chapters' / 'W1.md').read_text(encoding='utf-8')
    assert '[Summary](../../short/short.md)' in page and '[Full report](../all.md)' in page
    check_links(path.parent)
    with pytest.raises(ValueError, match='different files'):
        generate([path], full, full)
    with pytest.raises(ValueError, match='Markdown'):
        generate([path], path)
    with pytest.raises(ValueError, match='one score report per color'):
        load([path, path])


def test_entire_cached_pipeline_keeps_two_readable_reports(tmp_path, monkeypatch):
    import sys

    from repertoire_score import (
        character,
        openings,
        position_correlations,
        preparation,
        rating_correlations,
        ratings,
        report_insights,
        vulnerabilities,
    )

    directory = tmp_path / 'reports'
    data = directory / 'data'
    data.mkdir(parents=True)
    report, cache = run_fixture(data, monkeypatch)
    path = data / 'white.json'
    original = path.read_bytes()
    for module in [vulnerabilities, preparation, character, ratings, openings, report_insights]:
        monkeypatch.setattr(sys, 'argv', ['analysis', str(path), '--cache', str(cache)])
        module.main()
    monkeypatch.setattr(sys, 'argv', ['correlations', str(path), '--cache', str(cache)])
    position_correlations.main()
    generate([path], require_complete=True)
    assert path.read_bytes() == original
    position_correlations.analyze([path], data / 'prepared-depth-gain-correlation.json', cache)
    rating_result = rating_correlations.analyze([path])
    (data / 'opponent-rating-score-correlation.json').write_text(json.dumps(rating_result), encoding='utf-8')
    generate([path], require_complete=True)
    assert set(p.relative_to(directory).as_posix() for p in directory.rglob('*.md')) == {
        'report.md',
        'summary.md',
        *page_names(report),
    }
    check_links(directory)
    # A chapter that disappears from the study also loses its stale page.
    stale = directory / 'chapters' / 'W99.md'
    stale.write_text('old chapter', encoding='utf-8')
    (directory / 'chapters' / 'notes.md').write_text('kept', encoding='utf-8')
    generate([path], require_complete=True)
    assert not stale.exists() and (directory / 'chapters' / 'notes.md').exists()
    full = (directory / 'report.md').read_text()
    assert '[Scores](data/white.json)' in full
    # Published reports show the PGN relative to the report, never a local absolute path.
    source = Path(report['manifest']['input_path'])
    assert f"PGN: `{source.relative_to(directory).as_posix()}`" in full
    assert str(tmp_path) not in full and tmp_path.as_posix() not in full
    assert '(data/white.vulnerabilities.json)' in full
    assert 'point estimates from the observed database counts' in full
    assert '### Strengths' in full
    assert 'Strongest and weakest stopping outcomes' not in full
    assert 'not generated' not in full
    assert (data / '.report-index.json').exists()
    assert (data / 'prepared-depth-gain-correlation.json').exists()
    assert (data / 'opponent-rating-score-correlation.json').exists()
    assert load([path], require_complete=True)[0]['current_source']


def test_elo_equivalent_inverts_expected_score_and_handles_boundaries():
    assert elo_equivalent(0.5, 0.5) == pytest.approx(0)
    assert elo_equivalent(0.55, 0.5) == pytest.approx(34.86007028756008)
    assert elo_equivalent(0.4, 0.5) == pytest.approx(-elo_equivalent(0.6, 0.5))
    assert elo_equivalent(0.55, 0.55) == pytest.approx(0)
    shift = elo_equivalent(0.57, 0.52)
    initial_elo = elo_equivalent(0.52, 0.5)
    assert 1 / (1 + 10 ** (-(initial_elo + shift) / 400)) == pytest.approx(0.57)
    for score, baseline in [(None, 0.5), (0.5, None), (0, 0.5), (1, 0.5), (0.5, 0), (0.5, 1)]:
        assert elo_equivalent(score, baseline) is None


def test_centipawn_equivalent_is_direct_and_delta_subtracts_converted_scores():
    import math

    assert centipawn_equivalent(0.5) == pytest.approx(0)
    assert centipawn_equivalent(0.524967433244753) == pytest.approx(27.145761493072506)
    assert centipawn_delta(0.55, 0.55) == pytest.approx(0)
    assert centipawn_equivalent(0.4) == pytest.approx(-centipawn_equivalent(0.6))
    value = centipawn_delta(0.57, 0.52) + centipawn_equivalent(0.52)
    assert 1 / (1 + math.exp(-0.00368208 * value)) == pytest.approx(0.57)
    assert centipawn_equivalent(0.5472049870420534) == pytest.approx(51.43396313235625)
    assert centipawn_delta(0.8, 0.6) != pytest.approx(centipawn_delta(0.6, 0.4))
    assert centipawn_delta(0.524967433244753, 0.4808664795231225) == pytest.approx(47.94147281528)
    for score in [None, 0, 1, -0.1, 1.1]:
        assert centipawn_equivalent(score) is None
        assert cp(score) == 'unavailable'
    for score, baseline in [(None, 0.5), (0.5, None), (0, 0.5), (1, 0.5), (0.5, 0), (0.5, 1)]:
        assert centipawn_delta(score, baseline) is None
        assert cp_change(score, baseline) == 'unavailable'


def test_compressed_columns_preserve_values_and_inputs():
    headers = [
        'Line',
        'Position reach',
        'Avg games per encounter',
        'Opening',
        'ECO',
        'Avg opponent rating',
        'Rating Δ vs parent',
    ]
    rows = [['1. e4', '20.00%', '5', 'Kings Pawn', 'B00', '1,800 (90.0% rated)', '+30']]
    original = [r[:] for r in rows]
    names, values = compressed_columns(headers, rows)
    assert names == [
        'Line',
        'Position reach<br>Avg games per encounter',
        'Opening / ECO',
        'Avg opponent rating<br>Rating Δ vs parent',
    ]
    assert values == [['1. e4', '20.00%<br>1 per 5 games', 'Kings Pawn<br>B00', '1,800 (90.0% rated)<br>Δ +30']]
    assert rows == original and len(headers) == 7
    assert compressed_columns(names, values) == (names, values)
    with pytest.raises(ValueError, match='lengths differ'):
        table(['One'], [['one', 'extra']])


def test_top_navigation_covers_all_major_sections_and_preserves_existing_anchors():
    document = [
        '# Report',
        '',
        '<!-- report-navigation -->',
        '',
        '<a id="overview"></a>',
        '',
        '## Score overview',
        '',
        '### Score uncertainty',
        '',
        '<a id="white"></a>',
        '',
        '## White repertoire',
        '',
        '### Equivalent gap reach',
        '',
        '<a id="white-chapters"></a>',
        '',
        '### White chapters (2)',
        '',
        '#### W1. Test',
        '',
        '<a id="black"></a>',
        '',
        '## Black repertoire',
        '',
        '### Equivalent gap reach',
        '',
        '<a id="methods"></a>',
        '',
        '## Definitions and evidence',
        '',
        '### White evidence',
        '',
    ]
    rendered = '\n'.join(report_navigation(document))
    navigation = rendered.split('<a id="overview">', 1)[0]
    anchors = re.findall(r'<a id="([^"]+)"></a>\n\n#{2,3} ', rendered)
    assert set(anchors) <= set(re.findall(r'\]\(#([^)]+)\)', navigation))
    assert len(anchors) == len(set(anchors))
    assert '**Table of contents**\n\n- [Summary](summary.md)' in navigation
    assert '\n- [White repertoire](#white)\n    - [Equivalent gap reach](#white-equivalent-gap-reach)' in navigation
    assert '\n    - [Chapter-by-chapter table](#white-chapters)' in navigation
    assert 'white-equivalent-gap-reach' in anchors and 'black-equivalent-gap-reach' in anchors
    assert rendered.count('<a id="white-chapters">') == 1
    assert '#### W1. Test' in rendered and 'W1. Test' not in navigation
    assert '<!-- report-navigation -->' not in rendered


def test_score_table_checker_reproduces_headline_cp_and_rejects_cp_elsewhere():
    header = '| Repertoire | Starting baseline | Repertoire score | Delta |\n| --- | --- | --- | --- |\n'
    assert check_score_tables(header + f'| White | 50.00% | 60.00% | {headline_delta(0.6, 0.5)} |\n') == 1
    with pytest.raises(AssertionError):
        check_score_tables(header + '| White | 50.00% | 60.00% | +10.00% (+40 cp) |\n')
    with pytest.raises(AssertionError):
        check_score_tables('| Line | Repertoire score |\n| --- | --- |\n| e4 | 60.00% (+110 cp) |\n')


def test_cp_appears_only_beside_headline_deltas(complete):
    path, report = complete
    generate([path])
    for name in ('report.md', 'summary.md'):
        assert check_score_tables((path.parent / name).read_text(encoding='utf-8')) == 1
    for page in chapter_pages(path.parent, report):
        assert check_score_tables(page) == 0
        assert [row for row in page.splitlines() if ' cp)' in row] == [
            row for row in page.splitlines() if row.startswith('Chapter reach **')
        ]
        headline = next(row for row in page.splitlines() if row.startswith('Chapter reach **'))
        assert re.search(r'delta \*\*[+-][\d.]+% \([+-]\d+ cp\)\*\*', headline)


def test_all_repertoire_score_tables_show_spread_without_extra_columns(complete):
    path, report = complete
    bundles = generate([path])
    documents = [
        ('report.md', (path.parent / 'report.md').read_text(encoding='utf-8')),
        ('summary.md', (path.parent / 'summary.md').read_text(encoding='utf-8')),
        *[('chapter', page) for page in chapter_pages(path.parent, report)],
    ]
    for filename, text in documents:
        for block in re.findall(r'(?m)(?:^\|.*\|\n)+', text):
            rows = [[c.strip() for c in row.strip().strip('|').split('|')] for row in block.splitlines()]
            headers = rows[0]
            if headers[0] == 'Repertoire':
                continue  # the headline table keeps only the summary's leading measures
            for score_header, spread_headers in [
                ('Repertoire score', ('Score spread',)),
                ('Before reply (repertoire)', ('Score spread<br>Before / after', 'Score spread before reply')),
                ('After reply (repertoire)', ('Score spread<br>Before / after',)),
            ]:
                if score_header not in headers:
                    continue
                spread_header = next(h for h in spread_headers if h in headers)
                for row in rows[2:]:
                    if re.search(r'\d+\.\d+%', row[headers.index(score_header)]):
                        assert re.match(r'\d+\.\d+%', row[headers.index(spread_header)])
            # Unprepared replies end preparation, so their spread is zero by definition and is not a column.
            if 'After reply (database)' in headers or 'Database score' in headers:
                assert 'Score spread' not in headers and 'Score spread<br>Before / after' not in headers
            if 'Position type' in headers and 'Score' in headers:
                assert 'Score spread' in headers
            if filename == 'summary.md' and headers[0] == 'Chapter':
                assert len(headers) == 9
    for scope in [bundles[0]['vulnerabilities']['overall'], *bundles[0]['vulnerabilities']['chapters']]:
        for row in scope['all_signed_rows']:
            if row['move_outcomes']:
                assert row['move_outcomes']['resolved_score'] == pytest.approx(row['move_score'])
            if row['kind'] == 'opponent' and row['reference_outcomes']:
                assert row['reference_outcomes']['resolved_score'] == pytest.approx(row['reference_score'])


def test_reply_vulnerabilities_show_local_drag_weighted_drag_and_signed_cp_delta(complete):
    path, _ = complete
    bundle = load([path])[0]
    refs = Chapters(bundle['report'])
    original = next(r for r in bundle['vulnerabilities']['overall']['all_signed_rows'] if r['kind'] == 'opponent')
    common = dict(
        original,
        reference_score=0.575,
        move_score=0.525,
        branch_reach=0.02,
        branch_probability=0.1,
        local_drop_pp=5.0,
        weighted_drag_pp=0.1,
        sample_count=100,
        sparse=False,
    )
    scope = dict(
        id='overall',
        rankings={
            'own': [],
            'opponent': [
                dict(common, line='Prepared reply', prepared=True),
                dict(common, line='Unprepared reply', prepared=False),
            ],
        },
    )
    full = '\n'.join(vulnerabilities_section(scope, refs, 10))
    before, after = (spread_display(common.get(k), False) for k in ('reference_spread', 'move_spread'))
    prepared, unprepared = full.split('**Prepared opponent replies**')[::-1]
    assert f'| 57.50% | 52.50% | {before}<br>{after} | 5.00%<br>95%:' in prepared
    assert f'| 57.50% | 52.50% | {before} | 5.00%<br>95%:' in unprepared  # nothing after an unprepared reply
    assert '| Score spread before reply |' in unprepared and '| Score spread<br>Before / after |' in prepared
    # 0.1 percentage points per game is one point per 1,000 games.
    assert per_thousand(0.1) == '1.0' and full.count('| 1.0 |') == 2
    assert full.count('| Drag | Drag per 1,000 games |') == 2
    assert ' cp' not in full and centipawn_delta(0.525, 0.575) < 0
    assert check_score_tables(full) == 0
    bundle['vulnerabilities']['overall'] = scope
    summary = summary_report([bundle])
    # Summary rows label the parent board with its shared route, so identify the unprepared reply by its scores.
    assert summary.count('57.5%<br>52.5%') == 1 and '<summary>Costly unprepared replies</summary>' in summary
    assert check_score_tables(summary) == 1  # only the headline delta


def test_sparse_rankings_filter_before_limits_and_keep_summary_consistent(complete):
    from helpers import position

    path, report = complete
    bundle = load([path])[0]
    refs = Chapters(bundle['report'])
    old_scope = bundle['vulnerabilities']['overall']
    own = next(r for r in old_scope['all_signed_rows'] if r['kind'] == 'own')
    reply = next(r for r in old_scope['all_signed_rows'] if r['kind'] == 'opponent')
    # Without a move SAN the row keeps its own label, which lets the assertions identify each row.
    own = dict(
        own,
        sparse=False,
        parent_sparse=False,
        continuation_endpoint_sparse=False,
        move_san=None,
        local_gain_pp=10.0,
        local_drop_pp=10.0,
        sample_count=100,
        parent_sample_count=100,
    )
    own_rows = [
        dict(own, line='EXCLUDED_LOCAL', sparse=True),
        dict(own, line='EXCLUDED_PARENT', parent_sparse=True),
        dict(own, line='EXCLUDED_ENDPOINT', continuation_endpoint_sparse=True),
        dict(own, line='Supported own move'),
    ]
    replies = []
    for prepared in (False, True):
        label = 'prepared' if prepared else 'unprepared'
        replies.extend(
            [
                dict(reply, line=f'EXCLUDED_REPLY_{label}', prepared=prepared, sparse=True, move_san=None),
                dict(reply, line=f'Supported {label} reply', prepared=prepared, sparse=False, move_san=None),
            ]
        )
    scope = dict(id='overall', rankings={'own': own_rows, 'opponent': replies}, strengths=own_rows)

    def reached(path, route, reach, games, kind='opponent_reply', **extra):
        return dict(
            position=position(path),
            line=route,
            reach=reach,
            games=games,
            kind=kind,
            is_starting_position=False,
            to_move='black',
            repertoire_score=0.6,
            database_score=0.6,
            **extra,
        )

    positions = dict(
        bundle['character']['scopes'][0],
        positions=[
            reached('e4 c5', 'EXCLUDED_POSITION', 0.5, 10),
            reached(
                'e4 d5',
                'EXCLUDED_POOLED_POSITION',
                0.7,
                60,
                kind='unprepared_reply',
                unprepared_origins=[dict(games=10), dict(games=50)],
            ),
            reached('e4 e6', 'Supported reached position', 0.3, 100),
        ],
    )
    full = '\n'.join(vulnerabilities_section(scope, refs, 1) + strengths_section(scope, positions, refs, 1))
    assert 'EXCLUDED_' not in full
    assert 'Supported own move' in full
    assert 'Supported prepared reply' in full and 'Supported unprepared reply' in full
    assert 'Supported reached position' in full
    bundle['vulnerabilities']['overall'] = scope
    bundle['character']['scopes'][0] = positions
    summary = summary_report([bundle])
    review_blocks = dict(re.findall(r'<details>\n<summary>([^<]+)</summary>\n(.*?)</details>', summary, re.S))
    review_blocks['Own moves to review'] = summary.split('### Own moves to review', 1)[1].split('<details>', 1)[0]
    for title in ('Own moves to review', 'Costly unprepared replies', 'Strongest moves'):
        assert 'EXCLUDED_' not in review_blocks[title]
    assert '### Largest position contributions' not in summary
    assert 'Supported own move' in review_blocks['Own moves to review']
    assert 'Supported unprepared reply' in review_blocks['Costly unprepared replies']
    assert 'Supported reached position' in review_blocks['Most common positions']


@pytest.mark.parametrize('color', ['white', 'black'])
def test_position_contributions_include_intermediate_boards_and_cached_unprepared_replies(color):
    def row(label, reach, score, kind='opponent_reply', games=100, **extra):
        return dict(
            position=label,
            line=label,
            reach=reach,
            kind=kind,
            games=games,
            to_move='black' if color == 'white' else 'white',
            is_starting_position=False,
            repertoire_score=score,
            database_score=0.1,
            **extra,
        )

    root = row('starting board', 1.0, 0.99)
    root['is_starting_position'] = True
    scope = dict(
        id='chapter',
        positions=[
            root,
            row('1.e4', 1.0, 0.57),
            row('1.e4 e5', 0.5, 0.62, kind='own_move'),
            row('1.e4 e5 2.Nc3', 0.5, 0.62),
            dict(
                row('unprepared transposed reply', 0.3, None, kind='unprepared_reply'),
                database_score=0.8,
                unprepared_origins=[dict(reach=0.1, games=100), dict(reach=0.2, games=100)],
            ),
            row('prepared endpoint', 0.2, 0.85, kind='theory_leaf'),
            row('unresolved', 0.9, None, kind='unresolved_distribution'),
            row('sparse position', 0.9, 0.99, games=10),
            row('unknown local games', 0.9, 0.99, games=None),
        ],
    )
    ranked = non_sparse_rows(position_contributions(scope, color))
    assert [r['line'] for r in ranked] == ['1.e4', '1.e4 e5 2.Nc3', 'unprepared transposed reply', 'prepared endpoint']
    assert [r['contribution_pp'] for r in ranked] == pytest.approx([57.0, 31.0, 24.0, 17.0])
    assert ranked[2]['score'] == 0.8 and ranked[2]['score_basis'] == 'database'
    assert sum(r['contribution_pp'] for r in ranked) > 100  # overlapping continuation values are not additive
    refs = Chapters(dict(color=color, chapters=[]))
    displayed = '\n'.join(strengths_section(None, scope, refs, 2))
    assert '| [1. e4](' in displayed and '| [1. e4 e5 2. Nc3](' in displayed
    assert 'prepared endpoint' not in displayed
    assert 'Position reach after chapter entry' in displayed
    assert 'Rows overlap and must not be added' in displayed
    # Both displayed rows are prepared positions, so the uniform type column is dropped.
    assert '| Position type |' not in displayed and per_thousand(57.0) == '570' and '| 570 |' in displayed
    assert check_score_tables(displayed) == 0


def score_bundle(color, score, baseline, depth, chapters, games):
    return {
        'report': {
            'color': color,
            'overall': {'raw_empirical_score': score, 'prepared_depth': {'expected_moves': depth}},
            'starting_position_reference': {'owner_score': baseline, 'sample_count': games},
            'chapters': [{}] * chapters,
            'manifest': {'filters': {'speeds': 'blitz,rapid,classical'}, 'overall_basis': 'standard starting position'},
        }
    }


def test_combined_score_uses_equal_color_weights_before_elo_conversion():
    white = score_bundle('white', 0.7, 0.55, 8, 20, 1000)
    black = score_bundle('black', 0.5, 0.45, 4, 3, 1)
    result = combined_overall([black, white])
    assert result['weights'] == {'white': 0.5, 'black': 0.5}
    assert result['repertoire_score'] == pytest.approx(0.6)
    assert result['starting_baseline'] == pytest.approx(0.5)
    assert result['difference_pp'] == pytest.approx(10)
    assert result['elo_equivalent'] == pytest.approx(70.43650362227247)
    assert result['centipawn_equivalent'] == pytest.approx(centipawn_equivalent(0.6))
    assert result['baseline_centipawn_equivalent'] == pytest.approx(0)
    assert result['centipawn_delta'] == pytest.approx(centipawn_delta(0.6, 0.5))
    assert result['expected_prepared_depth'] == pytest.approx(6)
    assert result['chapters'] == 23
    averaged_elo = (elo_equivalent(0.7, 0.55) + elo_equivalent(0.5, 0.45)) / 2
    assert result['elo_equivalent'] != pytest.approx(averaged_elo)
    averaged_cp = (centipawn_equivalent(0.7) + centipawn_equivalent(0.5)) / 2
    assert result['centipawn_equivalent'] != pytest.approx(averaged_cp)
    assert combined_overall([white]) is None
    black['report']['overall']['raw_empirical_score'] = None
    result = combined_overall([white, black])
    assert result['repertoire_score'] is None and result['elo_equivalent'] is None
    assert result['centipawn_equivalent'] is None and result['centipawn_delta'] is None


def test_incompatible_color_scopes_are_not_combined():
    white = score_bundle('white', 0.56, 0.52, 4, 1, 10)
    black = score_bundle('black', 0.52, 0.48, 4, 1, 10)
    black['report']['manifest']['filters']['speeds'] = 'blitz'
    assert 'different Explorer filters' in combined_overall([white, black])['unavailable']
    black['report']['manifest']['filters'] = dict(white['report']['manifest']['filters'])
    black['report']['manifest']['overall_basis'] = 'conditional on custom PGN root'
    assert 'standard starting position' in combined_overall([white, black])['unavailable']


def test_combined_sharpness_uses_owner_relative_wdl_mixture():
    from repertoire_score.sharpness import summarize
    from repertoire_score.spread import stopping_counts

    white = score_bundle('white', 1.0, 0.5, 1, 1, 1000)
    black = score_bundle('black', 0.0, 0.5, 1, 1, 1)
    white['report']['overall']['outcomes'] = summarize([1.0, 0.0, 0.0, 0.0])
    black['report']['overall']['outcomes'] = summarize([0.0, 0.0, 1.0, 0.0])
    white['report']['overall']['branch_score_spread'] = stopping_counts([0, 0, 0], True, 1.0)
    black['report']['overall']['branch_score_spread'] = stopping_counts([0, 0, 0], False, 0.0)
    combined = combined_overall([white, black])
    outcomes = combined['outcomes']
    assert outcomes['win_probability'] == outcomes['loss_probability'] == 0.5
    assert outcomes['sharpness'] == 100.0  # both separate sharpness values are zero
    assert outcomes['resolved_score'] == 0.5
    assert combined['branch_score_spread']['variance'] == 0.25
    assert combined['branch_score_spread']['standard_deviation'] == 0.5  # separate spreads are zero


def test_combined_row_and_elo_are_rendered_in_both_documents(complete, monkeypatch):
    from types import SimpleNamespace

    from repertoire_score import score as cli

    path, white = complete
    args = SimpleNamespace(
        config=str(path.parent / 'config.json'),
        pgn=white['manifest']['input_path'],
        color='black',
        output=str(path.parent / 'black'),
        command='run',
        cache=str(path.parent / 'cache'),
        offline=True,
        refresh=False,
        prior=[0.5] * 3,
        sparse_threshold=30,
        tolerance=1,
    )
    cli.analyze(args)
    black_path = path.parent / 'black.json'
    bundles = generate([path, black_path])
    result = combined_overall(bundles)
    delta = headline_delta(result['repertoire_score'], result['starting_baseline'])
    assert delta == f"{result['difference_pp']:+.2f}% ({result['centipawn_delta']:+.0f} cp)"
    prefix = (
        f"| **Combined** | {100 * result['starting_baseline']:.2f}% | {100 * result['repertoire_score']:.2f}% | "
        f"{delta} | {result['elo_equivalent']:+.1f} |"
    )
    for filename in ['report.md', 'summary.md']:
        text = (path.parent / filename).read_text(encoding='utf-8')
        if filename == 'report.md':
            assert prefix in text
        else:
            with display(digits=1):
                short = headline_delta(result['repertoire_score'], result['starting_baseline'])
            assert (
                f"| **Combined** | {100 * result['starting_baseline']:.1f}% | "
                f"{100 * result['repertoire_score']:.1f}% | "
                f"{short} | {result['elo_equivalent']:+.0f} |"
            ) in text
        assert 'Elo equivalent' in text and '50% each' in text
        assert '| Score CP |' not in text and '| Baseline CP |' not in text and '| CP delta |' not in text
        assert check_score_tables(text) == 3  # White, Black and Combined headline deltas
        assert 'rating forecasts' in text
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    assert '[Combined repertoire](#combined)' in full and '<a id="combined"></a>' in full
    assert '## Combined repertoire' in full and '50% each' in full
    assert full.count('| **Combined** |') == 1
    for bundle in bundles:
        report = bundle['report']
        color = report['color']
        subsection = re.split(r'^## ', full.split(f'<a id="{color}"></a>', 1)[1], flags=re.M)[1]
        headings = re.findall(r'^### (.+)$', subsection, flags=re.M)
        expected = [
            f'{color.title()} chapters ({len(report["chapters"])})',
            'Where preparation ends',
            'Most common positions',
            'Openings reached',
            'Vulnerabilities',
            'Strengths',
            'Equivalent gap reach',
            'Prepared-depth distribution',
            'Branch score spread',
            'Preparation, replies, and resulting positions',
            'Score and evidence limits',
            'Uncertainty priorities and prior sensitivity',
        ]
        assert headings == [heading for heading in expected if heading in headings]
        if not bundle['unavailable']:
            assert headings == expected
        elo = elo_equivalent(
            report['overall']['raw_empirical_score'], report['starting_position_reference']['owner_score']
        )
        assert f'| {color.title()} |' in full and f'| {elo:+.1f} |' in full
        assert subsection.index(f'{color.title()} repertoire') < subsection.index(f'<a id="{color}-exits"')
        assert subsection.index(f'<a id="{color}-exits"') < subsection.index(f'<a id="{color}-common-positions"')
        if '### Vulnerabilities' in subsection:
            assert subsection.index('### Most common positions') < subsection.index('### Vulnerabilities')
        written = [p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*.md')]
        assert all(
            page in written for page in page_names(report) if 'openings' in bundle or not page.startswith('openings/')
        )
        assert (f'openings/{color}.md' in written) == ('openings' in bundle)
    check_links(path.parent)
    generate([path])
    assert '| **Combined** |' not in (path.parent / 'report.md').read_text(encoding='utf-8')
    # The one-color render removes the other color's chapter and opening pages.
    assert not (path.parent / 'chapters' / 'B1.md').exists() and not (path.parent / 'openings' / 'black.md').exists()


def test_common_positions_are_prominent_and_link_to_chapters(complete):
    path, _ = complete
    bundles = generate([path], position_top=2)
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    assert '[Most common positions](#white-common-positions)' in full
    assert (
        full.index('## White repertoire') < full.index('### Most common positions') < full.index('### Vulnerabilities')
    )
    assert not re.search(r'^## Most common positions$', full, flags=re.M)
    block = full.split('### Most common positions', 1)[1].split('### Openings reached', 1)[0]
    assert (
        '| Position (representative line) | Chapter source | Most common '
        'opening source | Position reach<br>Avg games per encounter | '
        'Repertoire score | Score spread | Games at position / reply |' in block
    )
    scope = bundles[0]['character']['scopes'][0]
    e4 = next(r for r in scope['positions'] if r['line'] == '1.e4')
    url = 'https://lichess.org/analysis/standard/' + '_'.join(e4['position'].split() + ['0', '1'])
    assert f'| [1. e4]({url}) |' in block
    assert f"| 100.00%<br>1 per 1.0 games | 40.00% | {spread_display(e4['branch_score_spread'])} | 100 |" in block
    assert '(PGN root)' not in block
    assert '[1. e4 e6](' not in block
    assert '| [1. e4 e6 2. d4](' in block
    assert 'Showing 2 of' in block
    assert '(chapters/W' in block or '(#white-chapters)' in block
    assert sum(r.startswith('| [1.') for r in block.splitlines()) == 2
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    tree = summary.split('<summary>Most common positions</summary>', 1)[1].split('</details>', 1)[0]
    assert f'- **[1. e4]({url})**: 100.0% of games, score 40.0%' in tree
    with pytest.raises(ValueError, match='positive table lengths'):
        generate([path], position_top=0)
    scope = bundles[0]['character']['scopes'][0]
    scope['unresolved_opponent_distribution_mass'] = 0.1
    assert 'Only known reach is ranked' in '\n'.join(common_positions_section(bundles, 2))
    scope.pop('positions')
    assert 'Position reach is not available' in '\n'.join(common_positions_section(bundles, 2))


def test_common_positions_collapse_guaranteed_replies_and_flag_unanswered_endpoints():
    from helpers import position

    def row(path, kind, turn, reach):
        return dict(
            position=position(path),
            line=path,
            kind=kind,
            to_move=turn,
            reach=reach,
            is_starting_position=False,
            repertoire_score=0.6,
            database_score=0.45,
            games=1234,
        )

    scope = {
        'id': 'overall',
        'positions': [
            row('e4 e5', 'own_move', 'white', 0.5),
            row('e4 e5 Nf3', 'opponent_reply', 'black', 0.5),
            row('e4 c5', 'unprepared_reply', 'white', 0.3),
            row('e4 e6 d4', 'theory_leaf', 'black', 0.2),
            row('e4 d5', 'theory_leaf', 'white', 0.1),
        ],
    }
    scope['positions'][2]['unprepared_origins'] = [dict(reach=0.1), dict(reach=0.2)]
    bundle = {'report': {'color': 'white', 'chapters': []}, 'character': {'scopes': [scope]}}
    text = '\n'.join(common_positions_section([bundle]))
    assert '[e4 e5](' not in text and '| [e4 e5 Nf3](' in text
    assert re.search(r'\[e4 c5\]\([^)]+\)<br>White to move; no prepared reply', text)
    assert 'e4 e6 d4<br>' not in text and 'e4 e6 d4)<br>' not in text
    assert 'Black to move' not in text and '| To move |' not in text
    assert '50.00%' in text and '30.00%' in text
    prepared, unprepared = text.split('#### Unprepared opponent replies', 1)
    assert 'e4 e5 Nf3' in prepared and 'e4 c5' not in prepared
    assert 'e4 c5' in unprepared and 'e4 e5 Nf3' not in unprepared
    assert re.search(r'\[e4 d5\]\([^)]+\)<br>White to move; no prepared reply', unprepared)
    assert 'e4 d5' not in prepared
    assert (
        '| Repertoire score | Score spread | Games at position / reply |' in prepared
        and '| 60.00% | unavailable | 1,234 |' in prepared
    )
    # Unprepared replies end preparation, so the always-zero spread column is omitted.
    assert '| Database score | Games at position / reply |' in unprepared and '| 45.00% | 1,234† |' in unprepared
    # Filter before the limit: the lower-frequency unprepared reply still appears.
    limited = '\n'.join(common_positions_section([bundle], 1))
    assert 'e4 e5 Nf3' in limited and 'e4 c5' in limited and 'e4 e6 d4' not in limited
    tree = '\n'.join(position_tree(scope, Chapters(bundle['report']), 2))
    assert '- **[e4 e5 Nf3](' in tree and '  - **[e4 e6 d4](' not in tree and 'e4 c5' not in tree
    assert '- **[e4 e6 d4](' in tree  # not a prefix extension of e4 e5 Nf3, so a separate root
    scope['id'] = 'chapter'
    chapter = '\n'.join(common_positions_for_scope(scope, Chapters(bundle['report']), 1, level='#####'))
    assert '##### Most common positions' in chapter
    assert '###### Unprepared opponent replies' in chapter
    assert 'Position reach after chapter entry' in chapter and 'Position reach<br>' not in chapter
    assert 'e4 e5 Nf3' in chapter and 'e4 c5' in chapter and 'e4 e6 d4' not in chapter


def test_position_tree_nests_by_route_and_shows_only_new_moves():
    from helpers import position

    def row(route, reach):
        moves = ' '.join(token.split('.')[-1] for token in route.split())
        return dict(
            position=position(moves),
            line=route,
            kind='opponent_reply',
            to_move='black',
            reach=reach,
            is_starting_position=False,
            repertoire_score=0.55,
            games=10,
        )

    scope = {
        'id': 'overall',
        'positions': [
            row('1.e4', 1.0),
            row('1.e4 1...e5 2.Nc3', 0.5),
            row('1.e4 1...e5 2.Nc3 2...Nc6 3.g3', 0.2),
            row('1.e4 1...c5 2.Nc3', 0.3),
        ],
    }
    tree = position_tree(scope, Chapters({'color': 'white', 'chapters': []}))
    labels = [re.match(r'( *)- \*\*\[([^\]]+)\]', item).groups() for item in tree if item]
    assert labels == [('', '1. e4'), ('  ', '1... e5 2. Nc3'), ('    ', '2... Nc6 3. g3'), ('  ', '1... c5 2. Nc3')]
    assert '50.00% of games, score 55.00%' in '\n'.join(tree)


def test_exit_points_group_unprepared_replies_by_last_prepared_board():
    from helpers import position

    parent, other = position('e4 c5 Nf3 Nc6 Bc4 e6'), position('e4 c5')

    def stop(board, move, reach, score):
        child = chess.Board(board + ' 0 1')
        child.push_uci(move)
        return dict(
            type='deviation',
            parent_position=board,
            position=' '.join(child.fen().split()[:4]),
            move=move,
            line=f'route {move}',
            reach=reach,
            score=score,
        )

    scope = {
        'id': 'overall',
        'positions': [
            dict(position=parent, line='1.e4 1...c5 2.Nf3 2...Nc6 3.Bc4 3...e6', reach=0.06, kind='opponent_reply'),
            dict(position=other, line='1.e4 1...c5', reach=0.6, kind='opponent_reply'),
            dict(position='leaf', line='1.e4 1...d5', reach=0.01, kind='theory_leaf'),
        ],
        'stopping_outcomes': [
            stop(parent, 'e1g1', 0.03, 0.5),
            stop(parent, 'd2d4', 0.02, 0.6),
            stop(parent, 'b1c3', 0.01, None),
            stop(other, 'd2d3', 0.04, 0.52),
            dict(
                type='theory_leaf',
                parent_position='leaf',
                position='leaf',
                move=None,
                line='leaf',
                reach=0.01,
                score=0.4,
            ),
            dict(type='terminal', parent_position=other, position='mate', move=None, line='mate', reach=0.5, score=1.0),
        ],
    }
    rows = exit_points(scope)
    assert [r['position'] for r in rows] == [parent, other, 'leaf']
    assert rows[0]['reach'] == pytest.approx(0.06) and rows[0]['share'] == pytest.approx(1.0)
    assert rows[0]['score'] == pytest.approx(
        (0.03 * 0.5 + 0.02 * 0.6) / 0.05
    )  # unresolved replies stay out of the score
    assert rows[1]['share'] == pytest.approx(0.04 / 0.6) and rows[2]['leaf'] and not rows[2]['replies']
    text = '\n'.join(exits_section(scope, Chapters({'color': 'black', 'chapters': []}), 2))
    assert '| [1. e4 c5 2. Nf3 Nc6 3. Bc4 e6](' in text and '| 6.00%<br>1 per 17 games | 100.00% | 3 |' in text
    assert re.search(r'\[4\. O-O\]\([^)]+\) 3\.00%, \[4\. d4\]\([^)]+\) 2\.00%, \[4\. Nc3\]\([^)]+\) 1\.00%', text)
    assert '| 4.00%<br>1 per 25 games | 6.67% | 1 |' in text and 'leaf' not in text
    assert (
        'These 2 positions are where **10.00%** of Black games leave '
        'preparation. Preparation ends at 3 positions in all.' in text
    )


def test_chapter_common_positions_use_conditional_reach_and_alternative_policy(tmp_path, monkeypatch):
    report, cache = run_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'white.json'
    path.with_suffix('.character.json').write_text(json.dumps(character(path, cache)), encoding='utf-8')
    before = {p: p.read_bytes() for p in tmp_path.glob('*.json')}
    bundles = generate([path], position_top=1)
    text = (tmp_path / 'report.md').read_text(encoding='utf-8')
    refs = Chapters(bundles[0]['report'])
    assert 'Most frequently used own decisions' not in text
    for chapter in report['chapters']:
        anchor = refs.anchor(chapter['id'])
        block = (tmp_path / 'chapters' / refs.page(chapter['id'])).read_text(encoding='utf-8')
        assert block.startswith(f'<a id="{anchor}"></a>')
        assert f'<a id="{anchor}-common-positions"></a>' in block
        assert '\n## Most common positions\n' in block
        assert 'Position reach after chapter entry' in block
        assert ' cp)' in block and 'Avg opponent rating' in block
        assert block.index('\n## Most common positions\n') < block.index('\n## Strengths\n')
        assert chapter['url'] in block
    alternative = next(c for c in report['chapters'] if c['id'] == 'tarrasch')
    assert alternative['score']['entry_probability'] == pytest.approx(0.6)
    block = (tmp_path / 'chapters' / 'W2.md').read_text(encoding='utf-8')
    block = block.split('<a id="white-chapter-2-common-positions"></a>', 1)[1].split('## Strengths', 1)[0]
    assert '| [1. e4 e6 2. d4 d5 3. Nd2 Nf6 4. e5](' in block
    assert (
        '| 100.00%<br>1 per 1.0 games | 90.00% |' in block
    )  # conditional on entry, not multiplied by the 60% chapter reach
    overall = text.split('### Most common positions', 1)[1].split('### Openings reached', 1)[0]
    assert '3. Nd2 Nf6' not in overall  # overall still selects the Advance
    assert all(p.read_bytes() == data for p, data in before.items())


def test_summary_leads_with_exits_and_own_moves_and_keeps_snapshot_notices_visible(complete):
    path, report = complete
    Path(report['manifest']['input_path']).write_text('newer study')
    generate([path])
    for name in ('report.md', 'summary.md', 'chapters/W1.md'):
        content = (path.parent / name).read_text(encoding='utf-8')
        assert 'source PGN has changed' in content
        assert content.count('<details>') == content.count('</details>')
    for name, row in (('report.md', '| White | 50.00% |'), ('summary.md', '| White | 50.0% |')):
        content = (path.parent / name).read_text(encoding='utf-8')
        assert content.index('snapshot notice') < content.index('| Repertoire |')
        assert 'scored ' in content and 'database evidence retrieved' in content
        assert content.count(row) == 1
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    assert (
        full.index('### White chapters') < full.index('### Where preparation ends') < full.index('### Vulnerabilities')
    )
    assert '<summary>Chapter comparisons (3 chapters)</summary>' in summary
    assert '### Openings reached' not in summary
    assert '| Opening family / variation |' not in summary
    assert 'Most common opening source' in summary
    summary_headings = re.findall(r'^### (.+)$', summary.split('## White repertoire', 1)[1], flags=re.M)
    assert summary_headings == [h for h in ('Where preparation ends', 'Own moves to review') if h in summary_headings]
    assert 'Where preparation ends' in summary_headings
    visible, depth = [], 0
    for row in summary.splitlines():
        if row == '<details>':
            depth += 1
            assert depth == 1
        elif row == '</details>':
            depth -= 1
            assert depth >= 0
        elif depth == 0:
            visible.append(row)
    assert depth == 0
    visible = '\n'.join(visible)
    assert 'snapshot notice' in visible
    assert (
        '| Preparation ends after |' in visible and '| Games leaving prep here<br>Avg games per encounter |' in visible
    )
    assert 'Costly unprepared replies' not in visible and '| Position (representative line) |' not in visible
    assert re.findall(r'^### (.+)$', visible, flags=re.M) == summary_headings
    assert 'Chapters |' not in visible and 'Outcome volatility' not in visible
    assert 'database evidence retrieved' not in visible
    assert '<summary>Most common positions</summary>' in summary
    assert '<summary>Preparation and variability</summary>' in summary
    assert (
        summary.index('### Where preparation ends')
        < summary.index('<summary>Most common positions')
        < summary.index('<summary>Chapter comparisons')
        < summary.index('<summary>Preparation and variability')
    )
    assert '### Largest position contributions' not in summary
    assert '### Equivalent gap reach' not in summary
    assert '**Gaps driving repeat encounters**' not in summary
    assert '| Repeat-gap share |' not in summary
    assert '<summary>Evidence and definitions</summary>' in summary
    assert 'Depth and improvement' not in summary and 'cluster-bootstrap' not in summary
    assert 'Strongest and weakest' not in full
    # One decimal place and compact counts in the summary; two decimals in the full report.
    assert not re.search(r'\d+\.\d{2}%', summary.split('<summary>Evidence and definitions</summary>')[0])
    assert re.search(r'\d+\.\d{2}%', full)
