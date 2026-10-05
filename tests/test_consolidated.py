import hashlib
import json
import re
from pathlib import Path

import pytest

from repertoire_score.ratings import analyze as ratings
from repertoire_score.openings import analyze as openings
from repertoire_score.character import analyze as character
from repertoire_score.consolidated import (Chapters, centipawn_delta, centipawn_equivalent, common_positions_for_scope, common_positions_section,
    combined_overall, cp, cp_change, elo_equivalent, generate, line, load, non_sparse_rows,
    position_contributions, strengths_section, summary_report, vulnerabilities_section)
from repertoire_score.consolidated import sharpness_display, spread_display
from repertoire_score.consolidated import compressed_columns, report_navigation, table
from repertoire_score.preparation import analyze as preparation
from repertoire_score.vulnerabilities import analyze as vulnerabilities
from repertoire_score.report_insights import analyze as report_insights
from test_chapter_policies import run_fixture


@pytest.fixture
def complete(tmp_path, monkeypatch):
    report, cache = run_fixture(tmp_path, monkeypatch, common_entry=True)
    path = tmp_path / 'white.json'
    for name, analyze in [('vulnerabilities', vulnerabilities), ('preparation', preparation), ('character', character), ('ratings', ratings), ('openings', openings), ('insights', report_insights)]:
        data = analyze(path, cache)
        path.with_suffix(f'.{name}.json').write_text(json.dumps(data))
    return path, report


def test_full_and_summary_preserve_metrics_sources_and_separate_reply_tables(complete):
    path, report = complete
    originals = {p: p.read_bytes() for p in path.parent.glob('*.json')}
    bundles = generate([path])
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    prefix = f'| White | 50.00% (+0.00 cp) | 40.00% ({cp(.4)} cp) | -10.00% ({cp_change(.4, .5)} cp) |'
    assert prefix in full
    assert f'| White | 50.00% (+0.00 cp) | 40.00% ({cp(.4)} cp) | -10.00% ({cp_change(.4, .5)} cp) |' in summary
    assert not re.search(r'^\| Black \|', full, flags=re.M) and '[Black](#black)' not in full
    assert prefix + ' -70.44 |' in full
    assert '[Combined](#combined)' not in full
    assert 'Chapter reach (incl. transpositions)' in summary and 'Overall-policy reach' in summary
    assert 'Tarrasch Nf6' in summary and '**(alternative)**' in summary
    assert 'Prepared opponent replies' in full
    assert '**Unprepared opponent replies**' not in full  # no positive unprepared reply drag in this fixture
    assert 'No resolved, reachable results in this category.' not in full
    assert '### Strengths' in full
    assert 'Strongest and weakest stopping outcomes' not in full
    assert 'Preparation, replies, and resulting positions' in full
    assert 'Most frequently used own decisions' not in full
    assert full.count('##### Most common positions') == len(report['chapters'])
    assert 'Optional memorization and trimming analysis' not in full
    assert 'Trim gain (pp)' not in full
    assert 'optional trimming' not in summary.lower()
    assert 'Exact first-entry positions' in full
    assert 'Posterior mean' not in full and 'Empirical score' not in full
    assert 'Study description' not in full + summary and '\u2014' not in full + summary
    for rendered in (full, summary):
        assert 'Most common opening source' in rendered and 'Unclassified (100.00%)' in rendered
        for header in rendered.splitlines():
            if header.startswith('| ') and any(label in header for label in
                    ('Chapter source', 'Chapter context', 'Entry-position sources')):
                assert 'Most common opening source' in header
    assert not bundles[0]['unavailable'] and bundles[0]['current_source']
    for p, content in originals.items():
        assert p.read_bytes() == content
    anchors = set(re.findall(r'<a id="([^"]+)"', full))
    assert all(target in anchors for target in re.findall(r'\]\(#([^)]+)\)', full))
    assert all(target in anchors for target in re.findall(r'\]\(report.md#([^)]+)\)', summary))
    assert full.count('<details>') == full.count('</details>')
    assert not list(path.parent.glob('*.study-description.md'))


def test_equivalent_gap_reach_in_overall_and_chapter_reports(complete):
    from repertoire_score.consolidated import gap_percentage
    path, report = complete
    bundle = generate([path])[0]
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    overall = bundle['report']['gap_coverage']
    assert '### Equivalent gap reach' in full
    assert f'**Equivalent gap reach: {gap_percentage(overall)}.**' in summary
    for text in (full, summary):
        assert 'Weighted gap reach contribution' in text
        assert 'Equivalent gap reach after entry' in text
    for chapter in bundle['report']['chapters']:
        metrics = chapter['gap_coverage']
        assert metrics
        if metrics['entry_probability'] is not None:
            assert metrics['weighted_equivalent_gap_reach_bounds'] == pytest.approx([
                metrics['entry_probability'] * p for p in metrics['equivalent_gap_reach_bounds']])
        assert f"weighted gap reach contribution **{gap_percentage(metrics, weighted=True)}**" in full
    assert 'weighted contributions are not additive' in full


def test_gap_display_keeps_unknown_reach_bounded():
    from repertoire_score.consolidated import gap_percentage
    assert gap_percentage(None) == 'unavailable'
    assert gap_percentage(dict(equivalent_gap_reach=0.)) == '0.00%'
    assert gap_percentage(dict(equivalent_gap_reach_bounds=[.1, .2])) == '10.00% to 20.00% (bounds)'
    assert gap_percentage(dict(weighted_equivalent_gap_reach_bounds=[.02, .04]), weighted=True) == '2.00% to 4.00% (bounds)'


def test_recursive_sharpness_in_overall_and_every_chapter(complete):
    from repertoire_score.consolidated import number
    path, report = complete
    bundle = generate([path])[0]
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    assert '| Score spread |' in full and '| Score spread |' in summary
    assert '| Sharpness |' not in full + summary
    assert '**Outcome volatility (formerly sharpness):**' in full
    scopes = {s['id']: s for s in bundle['character']['scopes']}
    for sid, score in [('overall', bundle['report']['overall']),
                       *[(c['id'], c['score']) for c in bundle['report']['chapters']]]:
        outcomes = score['outcomes']
        assert outcomes == scopes[sid]['outcomes']
        assert outcomes['win_probability'] + outcomes['draw_probability'] / 2 == pytest.approx(score['raw_empirical_score'])
        if sid != 'overall':
            assert f"outcome volatility **{number(outcomes['sharpness'])}%**" in full


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


def test_stale_correlations_rejected_and_matching_intervals_shown(complete):
    from repertoire_score.correlations import METRICS
    path, _ = complete
    correlations = path.parent / 'depth-delta-correlation.json'
    bundle = load([path])[0]
    result = dict(n=3, cluster_count=2, **{k: .3 for k in METRICS},
                  confidence_intervals_95={k: {'bounds': [.1, .5]} for k in METRICS})
    correlations.write_text(json.dumps(dict(results={'white': result}, bootstrap={'repetitions': 1000},
        provenance={'white': dict(report_sha256=bundle['digest'],
            input_sha256=bundle['report']['manifest']['input_sha256'], filters=bundle['report']['manifest']['filters'])})))
    generate([path], require_complete=True)
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    assert '95% cluster-bootstrap confidence intervals' in full and 'Rank correlation' in full
    data = json.loads(correlations.read_text())
    data['provenance']['white']['report_sha256'] = 'wrong'
    correlations.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='different score snapshots'):
        generate([path])
    generate([path], strict=False)
    assert 'Correlation analysis unavailable: belongs to different score snapshots' in (path.parent / 'report.md').read_text()


def test_rating_correlation_intervals_render_beside_depth_and_reject_stale_inputs(complete):
    from repertoire_score import rating_correlations
    path, _ = complete
    result = rating_correlations.analyze([path], repetitions=1000, seed=7)
    rows = [dict(position=str(i), rating=x + 400*i, score=y + .1*i, weight=w)
            for i in range(4) for x, y, w in ((1500, .6, 1), (1600, .55, 2), (1700, .5, 1))]
    result['results']['white']['reply_associations']['all'] = rating_correlations.within_parent(rows, 1000, 7)
    target = path.parent / 'opponent-rating-score-correlation.json'
    target.write_text(json.dumps(result), encoding='utf-8')
    generate([path])
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    assert full.index('## Prepared depth and score improvement') < full.index('## Opponent rating and score improvement') < full.index('## Definitions and evidence')
    assert '[Rating correlations](#rating-correlations)' in full
    assert '| -1.000 [-1.000, -1.000] | -5.000% [-5.000%, -5.000%] |' in full
    assert 'reach-weighted mean continuation score' in full and '1,000 replicates' in full
    assert 'parent-bootstrap confidence intervals' in full
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
    assert 'Rating correlation analysis unavailable: belongs to different analysis snapshots' in (path.parent / 'report.md').read_text()


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
    assert '../full/all.md#white-chapter-' in summary.read_text()
    with pytest.raises(ValueError, match='different files'):
        generate([path], full, full)
    with pytest.raises(ValueError, match='Markdown'):
        generate([path], path)
    with pytest.raises(ValueError, match='one score report per color'):
        load([path, path])


def test_entire_cached_pipeline_keeps_two_readable_reports(tmp_path, monkeypatch):
    import sys
    from repertoire_score import vulnerabilities, preparation, character, ratings, correlations, rating_correlations, attribution, openings, report_insights
    directory = tmp_path / 'reports'
    data = directory / 'data'
    data.mkdir(parents=True)
    report, cache = run_fixture(data, monkeypatch)
    path = data / 'white.json'
    original = path.read_bytes()
    for module in [vulnerabilities, preparation, character, ratings, openings, report_insights]:
        monkeypatch.setattr(sys, 'argv', ['analysis', str(path), '--cache', str(cache)])
        module.main()
    monkeypatch.setattr(sys, 'argv', ['correlations', str(path), '--bootstrap-samples', '1000'])
    correlations.main()
    generate([path], require_complete=True)
    assert path.read_bytes() == original
    attribution.regenerate([path])
    rating_result = rating_correlations.analyze([path], repetitions=1000)
    (data / 'opponent-rating-score-correlation.json').write_text(json.dumps(rating_result), encoding='utf-8')
    generate([path], require_complete=True)
    assert set(p.relative_to(directory).as_posix() for p in directory.rglob('*.md')) == {'report.md', 'summary.md'}
    full = (directory / 'report.md').read_text()
    assert '[Scores](data/white.json)' in full
    assert '(data/white.vulnerabilities.json)' in full
    assert '95% cluster-bootstrap confidence intervals' in full
    assert '### Strengths' in full
    assert 'Strongest and weakest stopping outcomes' not in full
    assert 'not generated' not in full
    assert (data / '.report-index.json').exists()
    assert (data / 'depth-delta-correlation.json').exists()
    assert (data / 'opponent-rating-score-correlation.json').exists()
    assert load([path], require_complete=True)[0]['current_source']


def test_elo_equivalent_inverts_expected_score_and_handles_boundaries():
    assert elo_equivalent(.5, .5) == pytest.approx(0)
    assert elo_equivalent(.55, .5) == pytest.approx(34.86007028756008)
    assert elo_equivalent(.4, .5) == pytest.approx(-elo_equivalent(.6, .5))
    assert elo_equivalent(.55, .55) == pytest.approx(0)
    shift = elo_equivalent(.57, .52)
    initial_elo = elo_equivalent(.52, .5)
    assert 1 / (1 + 10 ** (-(initial_elo + shift) / 400)) == pytest.approx(.57)
    for score, baseline in [(None, .5), (.5, None), (0, .5), (1, .5), (.5, 0), (.5, 1)]:
        assert elo_equivalent(score, baseline) is None


def test_centipawn_equivalent_is_direct_and_delta_subtracts_converted_scores():
    import math
    assert centipawn_equivalent(.5) == pytest.approx(0)
    assert centipawn_equivalent(.524967433244753) == pytest.approx(27.145761493072506)
    assert centipawn_delta(.55, .55) == pytest.approx(0)
    assert centipawn_equivalent(.4) == pytest.approx(-centipawn_equivalent(.6))
    value = centipawn_delta(.57, .52) + centipawn_equivalent(.52)
    assert 1 / (1 + math.exp(-.00368208 * value)) == pytest.approx(.57)
    assert centipawn_equivalent(.5472049870420534) == pytest.approx(51.43396313235625)
    assert centipawn_delta(.8, .6) != pytest.approx(centipawn_delta(.6, .4))
    assert centipawn_delta(.524967433244753, .4808664795231225) == pytest.approx(47.94147281528)
    for score in [None, 0, 1, -.1, 1.1]:
        assert centipawn_equivalent(score) is None
        assert cp(score) == 'unavailable'
    for score, baseline in [(None, .5), (.5, None), (0, .5), (1, .5), (.5, 0), (.5, 1)]:
        assert centipawn_delta(score, baseline) is None
        assert cp_change(score, baseline) == 'unavailable'


def check_score_tables(text):
    """Round-trip displayed CP back to its adjacent score and verify delta signs."""
    import math
    def numeric(value):
        bounded = re.fullmatch(r'([+-]?)<(\d+(?:\.\d+)?)', value)
        if bounded:
            return (-1 if bounded[1] == '-' else 1) * float(bounded[2]) / 2
        return float(value)
    score_headers = {'Starting baseline', 'Entry baseline', 'Parent database score', 'Move database score', 'Repertoire score',
                     'Before reply (repertoire)', 'After reply (repertoire)', 'After reply (database)',
                     'Score at stop', 'Score', 'Database score', 'Approximate 95% score interval',
                     'Sparse-evidence sensitivity', 'Scores before / after (CP)'}
    checked = 0
    for block in re.findall(r'(?m)(?:^\|.*\|\n)+', text):
        rows = [[cell.strip() for cell in row.strip().strip('|').split('|')] for row in block.splitlines()]
        headers = rows[0]
        for row in rows[2:]:
            assert len(row) == len(headers)
            for i, header in enumerate(headers):
                score = header in score_headers or (header == 'Value' and row[0] in (
                    'Approximate model-based 95% score interval', 'Conditional score bounds', 'Sparse-score sensitivity'))
                if not score: continue
                if ' cp)' in row[i]:
                    pairs = re.findall(r'([\d.]+)% \(([+-]?(?:<)?[\d.]+|unavailable) cp\)', row[i])
                    for percent, equivalent in pairs:
                        if equivalent == 'unavailable': continue
                        reproduced = 100 / (1 + math.exp(-.00368208 * numeric(equivalent)))
                        assert reproduced == pytest.approx(float(percent), abs=.0055)
                    assert pairs or 'unresolved' in row[i]
                    checked += 1
                    continue
                assert 'CP' in headers[i + 1].upper() or 'CENTIPAWN' in headers[i + 1].upper()
                percentages = re.findall(r'([\d.]+)%', row[i])
                values = row[i + 1].split(' to ')
                if not percentages:
                    assert row[i + 1] == 'unavailable'
                else:
                    assert len(percentages) == len(values)
                    for percent, equivalent in zip(percentages, values):
                        if equivalent == 'unavailable': continue  # exact boundary or missing evidence
                        reproduced = 100 / (1 + math.exp(-.00368208 * numeric(equivalent)))
                        assert reproduced == pytest.approx(float(percent), abs=.0055)
                checked += 1
            if 'CP delta' in headers and any(h in headers for h in ('Before CP', 'Parent CP', 'Baseline CP')):
                before_header = next(h for h in ('Before CP', 'Parent CP', 'Baseline CP') if h in headers)
                after_header = 'After CP' if 'After CP' in headers else 'Score CP'
                before, after, delta = (row[headers.index(h)] for h in (before_header, after_header, 'CP delta'))
                if 'unavailable' in (before, after):
                    assert delta == 'unavailable'
                else:
                    assert numeric(delta) == pytest.approx(numeric(after) - numeric(before), abs=.0151)
            before_header = next((h for h in ('Starting baseline', 'Entry baseline', 'Parent database score',
                                              'Before reply (repertoire)') if h in headers), None)
            after_header = next((h for h in ('Repertoire score', 'After reply (repertoire)', 'After reply (database)')
                                 if h in headers), None)
            delta_header = next((h for h in ('Delta', 'Gain / CP delta', 'Drag / CP delta') if h in headers), None)
            if before_header and after_header and delta_header:
                converted = [re.findall(r'\(([+-]?(?:<)?[\d.]+|unavailable) cp\)', row[headers.index(h)])
                             for h in (before_header, after_header, delta_header)]
                if all(converted):
                    before, after, delta = (values[0] for values in converted)
                    if before == 'unavailable' or after == 'unavailable':
                        assert delta == 'unavailable'
                    else:
                        assert numeric(delta) == pytest.approx(numeric(after) - numeric(before), abs=.0151)
    assert checked > 0


def test_compressed_columns_preserve_values_notes_and_inputs():
    headers = ['Chapter', 'Chapter reach (incl. transpositions)', 'Overall-policy reach',
               'Entry baseline', 'Baseline CP', 'Repertoire score', 'Score CP', 'Delta', 'CP delta',
               'Weighted gap reach contribution', 'Equivalent gap reach after entry', 'Avg opponent rating', 'Rating Δ vs parent']
    rows = [['Test', '20.00%', '15.00%', '50.00%', '+0.00', '60.00%', '+110.11', '+10.00%', '+110.11',
             '1.00%', '5.00%', '1,800 (90.0% rated)', '+30']]
    original = [r[:] for r in rows]
    names, values = compressed_columns(headers, rows)
    assert len(names) == 7
    assert values == [['Test', '20.00%<br>Overall 15.00%', '50.00% (+0.00 cp)', '60.00% (+110.11 cp)',
                       '+10.00% (+110.11 cp)', '5.00%<br>Weighted 1.00%', '1,800 (90.0% rated)<br>Δ +30']]
    assert rows == original and len(headers) == 13
    names, values = compressed_columns(['CP delta', 'Drag', 'Weighted drag'], [['-10.00', '1.00%<br>95%: 0.50% to 1.50%', '.10%']])
    assert names == ['Drag / CP delta', 'Weighted drag']
    assert values == [['1.00% (-10.00 cp)<br>95%: 0.50% to 1.50%', '.10%']]
    assert compressed_columns(names, values) == (names, values)


def test_compressed_intervals_and_probability_rows_keep_cp_semantics():
    rendered = '\n'.join(table(['Measure', 'Value', 'Centipawn equivalent (cp)'], [
        ['Conditional score bounds', '50.00% to 60.00%', '+0.00 to ' + cp(.6)],
        ['Unresolved probability', '0.00%', 'n/a']]))
    assert '| Measure | Value |' in rendered
    assert f'50.00% (+0.00 cp) to 60.00% ({cp(.6)} cp)' in rendered
    assert '| Unresolved probability | 0.00% |' in rendered
    check_score_tables(rendered)
    with pytest.raises(ValueError, match='lengths differ'):
        table(['One'], [['one', 'extra']])


def test_top_navigation_covers_all_major_sections_and_preserves_existing_anchors():
    document = ['# Report', '', '<!-- report-navigation -->', '', '<a id="overview"></a>', '',
                '## Score overview', '', '### Score uncertainty', '', '<a id="white"></a>', '',
                '## White repertoire', '', '### Equivalent gap reach', '', '<a id="white-chapters"></a>', '',
                '### White chapters (2)', '', '#### W1. Test', '', '<a id="black"></a>', '',
                '## Black repertoire', '', '### Equivalent gap reach', '', '<a id="methods"></a>', '',
                '## Definitions and evidence', '', '### White evidence', '']
    rendered = '\n'.join(report_navigation(document))
    navigation = rendered.split('<a id="overview">', 1)[0]
    anchors = re.findall(r'<a id="([^"]+)"></a>\n\n#{2,3} ', rendered)
    assert set(anchors) <= set(re.findall(r'\]\(#([^)]+)\)', navigation))
    assert len(anchors) == len(set(anchors))
    assert '[Chapter-by-chapter table](#white-chapters)' in navigation
    assert 'white-equivalent-gap-reach' in anchors and 'black-equivalent-gap-reach' in anchors
    assert rendered.count('<a id="white-chapters">') == 1
    assert '#### W1. Test' in rendered and 'W1. Test' not in navigation
    assert '<!-- report-navigation -->' not in rendered


@pytest.mark.parametrize('sign', ['+', '-'])
def test_score_table_checker_accepts_bounded_small_equivalents(sign):
    check_score_tables('| Entry baseline | Baseline CP | Repertoire score | Score CP | CP delta |\n'
                       '| --- | --- | --- | --- | --- |\n'
                       f'| 50.00% | +0.00 | 50.00% | {sign}<0.01 | {sign}<0.01 |\n')


def test_every_score_table_has_direct_cp_and_comparisons_have_cp_delta(complete):
    path, _ = complete
    generate([path])
    for name in ('report.md', 'summary.md'):
        check_score_tables((path.parent/name).read_text(encoding='utf-8'))


def test_all_repertoire_score_tables_show_spread_without_extra_columns(complete):
    path, _ = complete
    bundles = generate([path])
    for filename in ('report.md', 'summary.md'):
        text = (path.parent / filename).read_text(encoding='utf-8')
        for block in re.findall(r'(?m)(?:^\|.*\|\n)+', text):
            rows = [[c.strip() for c in row.strip().strip('|').split('|')] for row in block.splitlines()]
            headers = rows[0]
            for score_header, spread_header in [('Repertoire score', 'Score spread'),
                    ('Before reply (repertoire)', 'Score spread<br>Before / after'),
                    ('After reply (repertoire)', 'Score spread<br>Before / after')]:
                if score_header not in headers:
                    continue
                assert spread_header in headers
                for row in rows[2:]:
                    if re.search(r'\d+\.\d+%', row[headers.index(score_header)]):
                        assert re.match(r'\d+\.\d+%', row[headers.index(spread_header)])
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
    common = dict(original, reference_score=.575, move_score=.525, branch_reach=.02,
                  branch_probability=.1, local_drop_pp=5., weighted_drag_pp=.1, sample_count=100,
                  sparse=False)
    scope = dict(id='overall', rankings={'own': [], 'opponent': [
        dict(common, line='Prepared reply', prepared=True), dict(common, line='Unprepared reply', prepared=False)]})
    full = '\n'.join(vulnerabilities_section(scope, refs, 10))
    expected = (f"| 57.50% ({cp(.575)} cp) | 52.50% ({cp(.525)} cp) | "
                f"{spread_display(common.get('reference_spread'), False)}<br>{spread_display(common.get('move_spread'), False)} | "
                f"5.00% ({cp_change(.525, .575)} cp)<br>95%:")
    assert full.count(expected) == 2 and full.count('0.1000%') == 2
    assert full.count('| Drag / CP delta | Weighted drag |') == 2
    assert centipawn_delta(.525, .575) < 0
    check_score_tables(full)
    bundle['vulnerabilities']['overall'] = scope
    summary = summary_report([bundle], path.parent/'report.md', None, None)
    assert f'{cp(.575)} cp)' in summary and f'{cp(.525)} cp)' in summary and 'Prepared reply |' not in summary
    check_score_tables(summary)


def test_sparse_rankings_filter_before_limits_and_keep_summary_consistent(complete):
    from test_model import position
    path, report = complete
    bundle = load([path])[0]
    refs = Chapters(bundle['report'])
    old_scope = bundle['vulnerabilities']['overall']
    own = next(r for r in old_scope['all_signed_rows'] if r['kind'] == 'own')
    reply = next(r for r in old_scope['all_signed_rows'] if r['kind'] == 'opponent')
    own = dict(own, sparse=False, parent_sparse=False, continuation_endpoint_sparse=False,
               local_gain_pp=10., local_drop_pp=10., sample_count=100, parent_sample_count=100)
    own_rows = [dict(own, line='EXCLUDED_LOCAL', sparse=True),
                dict(own, line='EXCLUDED_PARENT', parent_sparse=True),
                dict(own, line='EXCLUDED_ENDPOINT', continuation_endpoint_sparse=True),
                dict(own, line='Supported own move')]
    replies = []
    for prepared in (False, True):
        label = 'prepared' if prepared else 'unprepared'
        replies.extend([dict(reply, line=f'EXCLUDED_REPLY_{label}', prepared=prepared, sparse=True),
                        dict(reply, line=f'Supported {label} reply', prepared=prepared, sparse=False)])
    scope = dict(id='overall', rankings={'own':own_rows, 'opponent':replies}, strengths=own_rows)
    def reached(path, route, reach, games, kind='opponent_reply', **extra):
        return dict(position=position(path), line=route, reach=reach, games=games, kind=kind,
                    is_starting_position=False, to_move='black', repertoire_score=.6, database_score=.6, **extra)
    positions = dict(bundle['character']['scopes'][0], positions=[
        reached('e4 c5', 'EXCLUDED_POSITION', .5, 10),
        reached('e4 d5', 'EXCLUDED_POOLED_POSITION', .7, 60, kind='unprepared_reply',
                unprepared_origins=[dict(games=10), dict(games=50)]),
        reached('e4 e6', 'Supported reached position', .3, 100)])
    full = '\n'.join(vulnerabilities_section(scope, refs, 1) + strengths_section(scope, positions, refs, 1))
    assert 'EXCLUDED_' not in full
    assert 'Supported own move' in full
    assert 'Supported prepared reply' in full and 'Supported unprepared reply' in full
    assert 'Supported reached position' in full
    bundle['vulnerabilities']['overall'] = scope
    bundle['character']['scopes'][0] = positions
    summary = summary_report([bundle], path.parent/'report.md', None, None)
    contribution_block = summary.split('### Largest position contributions', 1)[1].split('Preparation ends', 1)[0]
    assert 'EXCLUDED_' not in contribution_block
    assert 'Supported own move' in summary and 'Supported unprepared reply' in summary
    assert 'Supported reached position' in summary


@pytest.mark.parametrize('color', ['white', 'black'])
def test_position_contributions_include_intermediate_boards_and_cached_unprepared_replies(color):
    def row(label, reach, score, kind='opponent_reply', games=100, **extra):
        return dict(position=label, line=label, reach=reach, kind=kind, games=games,
                    to_move='black' if color == 'white' else 'white', is_starting_position=False,
                    repertoire_score=score, database_score=.1, **extra)
    root = row('starting board', 1., .99)
    root['is_starting_position'] = True
    scope = dict(id='chapter', positions=[root,
        row('1.e4', 1., .57),
        row('1.e4 e5', .5, .62, kind='own_move'),
        row('1.e4 e5 2.Nc3', .5, .62),
        dict(row('unprepared transposed reply', .3, None, kind='unprepared_reply'), database_score=.8,
             unprepared_origins=[dict(reach=.1, games=100), dict(reach=.2, games=100)]),
        row('prepared endpoint', .2, .85, kind='theory_leaf'),
        row('unresolved', .9, None, kind='unresolved_distribution'),
        row('sparse position', .9, .99, games=10),
        row('unknown local games', .9, .99, games=None)])
    ranked = non_sparse_rows(position_contributions(scope, color))
    assert [r['line'] for r in ranked] == ['1.e4', '1.e4 e5 2.Nc3', 'unprepared transposed reply', 'prepared endpoint']
    assert [r['contribution_pp'] for r in ranked] == pytest.approx([57., 31., 24., 17.])
    assert ranked[2]['score'] == .8 and ranked[2]['score_basis'] == 'database'
    assert sum(r['contribution_pp'] for r in ranked) > 100  # overlapping continuation values are not additive
    refs = Chapters(dict(color=color, chapters=[]))
    displayed = '\n'.join(strengths_section(None, scope, refs, 2))
    assert '| 1. e4 |' in displayed and '| 1. e4 e5 2. Nc3 |' in displayed
    assert 'endpoint |' not in displayed
    assert 'Reach after chapter entry' in displayed
    assert 'Rows overlap and must not be added' in displayed
    check_score_tables(displayed)


def score_bundle(color, score, baseline, depth, chapters, games):
    return {'report': {'color': color, 'overall': {'raw_empirical_score': score, 'prepared_depth': {'expected_moves': depth}},
            'starting_position_reference': {'owner_score': baseline, 'sample_count': games}, 'chapters': [{}] * chapters,
            'manifest': {'filters': {'speeds': 'blitz,rapid,classical'}, 'overall_basis': 'standard starting position'}}}


def test_combined_score_uses_equal_color_weights_before_elo_conversion():
    white = score_bundle('white', .7, .55, 8, 20, 1000)
    black = score_bundle('black', .5, .45, 4, 3, 1)
    result = combined_overall([black, white])
    assert result['weights'] == {'white': .5, 'black': .5}
    assert result['repertoire_score'] == pytest.approx(.6)
    assert result['starting_baseline'] == pytest.approx(.5)
    assert result['difference_pp'] == pytest.approx(10)
    assert result['elo_equivalent'] == pytest.approx(70.43650362227247)
    assert result['centipawn_equivalent'] == pytest.approx(centipawn_equivalent(.6))
    assert result['baseline_centipawn_equivalent'] == pytest.approx(0)
    assert result['centipawn_delta'] == pytest.approx(centipawn_delta(.6, .5))
    assert result['expected_prepared_depth'] == pytest.approx(6)
    assert result['chapters'] == 23
    averaged_elo = (elo_equivalent(.7, .55) + elo_equivalent(.5, .45)) / 2
    assert result['elo_equivalent'] != pytest.approx(averaged_elo)
    averaged_cp = (centipawn_equivalent(.7) + centipawn_equivalent(.5)) / 2
    assert result['centipawn_equivalent'] != pytest.approx(averaged_cp)
    assert combined_overall([white]) is None
    black['report']['overall']['raw_empirical_score'] = None
    result = combined_overall([white, black])
    assert result['repertoire_score'] is None and result['elo_equivalent'] is None
    assert result['centipawn_equivalent'] is None and result['centipawn_delta'] is None


def test_incompatible_color_scopes_are_not_combined():
    white = score_bundle('white', .56, .52, 4, 1, 10)
    black = score_bundle('black', .52, .48, 4, 1, 10)
    black['report']['manifest']['filters']['speeds'] = 'blitz'
    assert 'different Explorer filters' in combined_overall([white, black])['unavailable']
    black['report']['manifest']['filters'] = dict(white['report']['manifest']['filters'])
    black['report']['manifest']['overall_basis'] = 'conditional on custom PGN root'
    assert 'standard starting position' in combined_overall([white, black])['unavailable']


def test_combined_sharpness_uses_owner_relative_wdl_mixture():
    from repertoire_score.sharpness import summarize
    from repertoire_score.spread import stopping_counts
    white = score_bundle('white', 1., .5, 1, 1, 1000)
    black = score_bundle('black', 0., .5, 1, 1, 1)
    white['report']['overall']['outcomes'] = summarize([1., 0., 0., 0.])
    black['report']['overall']['outcomes'] = summarize([0., 0., 1., 0.])
    white['report']['overall']['branch_score_spread'] = stopping_counts([0, 0, 0], True, 1.)
    black['report']['overall']['branch_score_spread'] = stopping_counts([0, 0, 0], False, 0.)
    combined = combined_overall([white, black])
    outcomes = combined['outcomes']
    assert outcomes['win_probability'] == outcomes['loss_probability'] == .5
    assert outcomes['sharpness'] == 100.  # both separate sharpness values are zero
    assert outcomes['resolved_score'] == .5
    assert combined['branch_score_spread']['variance'] == .25
    assert combined['branch_score_spread']['standard_deviation'] == .5  # separate spreads are zero


def test_combined_row_and_elo_are_rendered_in_both_documents(complete, monkeypatch):
    from types import SimpleNamespace
    from repertoire_score import __main__ as cli
    path, white = complete
    args = SimpleNamespace(config=str(path.parent/'config.json'), pgn=white['manifest']['input_path'], color='black',
                           output=str(path.parent/'black'), command='run', cache=str(path.parent/'cache'), offline=True,
                           refresh=False, simulations=100, seed=1, prior=[.5]*3, sparse_threshold=30, tolerance=1)
    cli.analyze(args)
    black_path = path.parent/'black.json'
    bundles = generate([path, black_path])
    result = combined_overall(bundles)
    prefix = f"| **Combined** | {100*result['starting_baseline']:.2f}% ({cp(result['starting_baseline'])} cp) | {100*result['repertoire_score']:.2f}% ({cp(result['repertoire_score'])} cp) | {result['difference_pp']:+.2f}% ({cp_change(result['repertoire_score'], result['starting_baseline'])} cp) | {result['elo_equivalent']:+.2f} |"
    for filename in ['report.md', 'summary.md']:
        text = (path.parent/filename).read_text(encoding='utf-8')
        if filename == 'report.md':
            assert prefix in text
        else:
            assert f"| **Combined** | {100*result['starting_baseline']:.2f}% ({cp(result['starting_baseline'])} cp) | {100*result['repertoire_score']:.2f}% ({cp(result['repertoire_score'])} cp) |" in text
            assert f"| {result['elo_equivalent']:+.2f} |" in text
        assert 'Elo equivalent' in text and '50% each' in text
        if filename == 'report.md':
            assert '(+0.00 cp)' in text and 'CP delta' in text
            assert '| Score CP |' not in text and '| Baseline CP |' not in text
        else:
            assert 'Starting baseline' in text and 'Repertoire score' in text and ' cp)' in text
        assert 'not a measured rating gain' in text
    full = (path.parent/'report.md').read_text(encoding='utf-8')
    assert '[Combined](#combined)' in full and '<a id="combined"></a>' in full
    assert '## Combined repertoire' in full and '50% each' in full
    assert full.count('| **Combined** |') == 1
    for bundle in bundles:
        report = bundle['report']; color = report['color']
        subsection = re.split(r'^## ', full.split(f'<a id="{color}"></a>', 1)[1], flags=re.M)[1]
        headings = re.findall(r'^### (.+)$', subsection, flags=re.M)
        expected = [
            f'{color.title()} chapters ({len(report["chapters"])})', 'Most common positions', 'Openings reached',
            'Vulnerabilities', 'Strengths', 'Equivalent gap reach', 'Prepared-depth distribution',
            'Branch score spread', 'Preparation, replies, and resulting positions', 'Chapter-by-chapter analysis',
            'Score and evidence limits', 'Uncertainty priorities and prior sensitivity']
        assert headings == [heading for heading in expected if heading in headings]
        if not bundle['unavailable']:
            assert headings == expected
        elo = elo_equivalent(report['overall']['raw_empirical_score'], report['starting_position_reference']['owner_score'])
        assert f'| {color.title()} |' in full and f'| {elo:+.2f} |' in full
        assert subsection.index(f'{color.title()} repertoire') < subsection.index(f'<a id="{color}-common-positions"')
        if '### Vulnerabilities' in subsection:
            assert subsection.index('### Most common positions') < subsection.index('### Vulnerabilities')
    generate([path])
    assert '| **Combined** |' not in (path.parent/'report.md').read_text(encoding='utf-8')


def test_common_positions_are_prominent_and_link_to_chapters(complete):
    path, _ = complete
    bundles = generate([path], position_top=2)
    full = (path.parent/'report.md').read_text(encoding='utf-8')
    assert '[Most common positions](#white-common-positions)' in full
    assert full.index('## White repertoire') < full.index('### Most common positions') < full.index('### Vulnerabilities')
    assert not re.search(r'^## Most common positions$',full,flags=re.M)
    block = full.split('### Most common positions', 1)[1].split('### Openings reached', 1)[0]
    assert '| Position (representative line) | Chapter source | Most common opening source | Reach in repertoire<br>Avg games per encounter | Repertoire score | Score spread | Games at position / reply |' in block
    scope = bundles[0]['character']['scopes'][0]
    e4 = next(r for r in scope['positions'] if r['line'] == '1.e4')
    assert '| 1. e4 |' in block and f"| 100.00%<br>1 per 1.0 games | 40.00% ({cp(.4)} cp) | {spread_display(e4['branch_score_spread'])} | 100 |" in block
    assert '(PGN root)' not in block
    assert '1. e4' in block and '100.00%' in block
    assert '| 1. e4 e6 |' not in block
    assert '| 1. e4 e6 2. d4 |' in block
    assert 'Showing 2 of' in block
    assert '(#white-chapter-' in block or '(#white-chapters)' in block
    assert sum(r.startswith('| 1.') for r in block.splitlines()) == 2
    assert 'Most common positions' not in (path.parent/'summary.md').read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='positive table lengths'):
        generate([path], position_top=0)
    scope = bundles[0]['character']['scopes'][0]
    scope['unresolved_opponent_distribution_mass'] = .1
    assert 'Only known reach is ranked' in '\n'.join(common_positions_section(bundles,2))
    scope.pop('positions')
    assert 'Position reach is not available' in '\n'.join(common_positions_section(bundles,2))


def test_common_positions_collapse_guaranteed_replies_and_flag_unanswered_endpoints():
    from test_model import position
    def row(path, kind, turn, reach):
        return dict(position=position(path), line=path, kind=kind, to_move=turn, reach=reach, is_starting_position=False,
                    repertoire_score=.6, database_score=.45, games=1234)
    scope = {'id': 'overall', 'positions': [
        row('e4 e5', 'own_move', 'white', .5),
        row('e4 e5 Nf3', 'opponent_reply', 'black', .5),
        row('e4 c5', 'unprepared_reply', 'white', .3),
        row('e4 e6 d4', 'theory_leaf', 'black', .2),
        row('e4 d5', 'theory_leaf', 'white', .1)]}
    scope['positions'][2]['unprepared_origins'] = [dict(reach=.1),dict(reach=.2)]
    bundle = {'report': {'color': 'white', 'chapters': []}, 'character': {'scopes': [scope]}}
    text = '\n'.join(common_positions_section([bundle]))
    assert '| e4 e5 |' not in text and '| e4 e5 Nf3 |' in text
    assert 'e4 c5<br>White to move; no prepared reply' in text
    assert 'e4 e6 d4<br>' not in text
    assert 'Black to move' not in text and '| To move |' not in text
    assert '50.00%' in text and '30.00%' in text
    prepared, unprepared = text.split('#### Unprepared opponent replies',1)
    assert 'e4 e5 Nf3' in prepared and 'e4 c5' not in prepared
    assert 'e4 c5' in unprepared and 'e4 e5 Nf3' not in unprepared
    assert 'e4 d5<br>White to move; no prepared reply' in unprepared
    assert 'e4 d5' not in prepared
    assert '| Repertoire score | Score spread | Games at position / reply |' in prepared and f'| 60.00% ({cp(.6)} cp) | unavailable | 1,234 |' in prepared
    assert '| Database score | Score spread | Games at position / reply |' in unprepared and f'| 45.00% ({cp(.45)} cp) | unavailable | 1,234† |' in unprepared
    # Filter before the limit: the lower-frequency unprepared reply still appears.
    limited = '\n'.join(common_positions_section([bundle],1))
    assert 'e4 e5 Nf3' in limited and 'e4 c5' in limited and 'e4 e6 d4' not in limited
    scope['id'] = 'chapter'
    chapter = '\n'.join(common_positions_for_scope(scope, Chapters(bundle['report']), 1, level='#####'))
    assert '##### Most common positions' in chapter
    assert '###### Unprepared opponent replies' in chapter
    assert 'Reach after chapter entry' in chapter and 'Reach in repertoire' not in chapter
    assert 'e4 e5 Nf3' in chapter and 'e4 c5' in chapter and 'e4 e6 d4' not in chapter


def test_chapter_common_positions_use_conditional_reach_and_alternative_policy(tmp_path, monkeypatch):
    report, cache = run_fixture(tmp_path, monkeypatch)
    path = tmp_path/'white.json'
    path.with_suffix('.character.json').write_text(json.dumps(character(path, cache)), encoding='utf-8')
    before = {p:p.read_bytes() for p in tmp_path.glob('*.json')}
    bundles = generate([path], position_top=1)
    text = (tmp_path/'report.md').read_text(encoding='utf-8')
    refs = Chapters(bundles[0]['report'])
    assert 'Most frequently used own decisions' not in text
    for chapter in report['chapters']:
        anchor = refs.anchor(chapter['id'])
        block = text.split(f'<a id="{anchor}"></a>', 1)[1].split('</details>', 1)[0]
        assert f'<a id="{anchor}-common-positions"></a>' in block
        assert '##### Most common positions' in block
        assert 'Reach after chapter entry' in block
        assert ' cp)' in block and 'Avg opponent rating' in block
        assert block.index('##### Most common positions') < block.index('##### Strengths')
    alternative = next(c for c in report['chapters'] if c['id'] == 'tarrasch')
    assert alternative['score']['entry_probability'] == pytest.approx(.6)
    block = text.split('<a id="white-chapter-2-common-positions"></a>', 1)[1].split('##### Strengths', 1)[0]
    assert '| 1. e4 e6 2. d4 d5 3. Nd2 Nf6 4. e5 |' in block
    assert f'| 100.00%<br>1 per 1.0 games | 90.00% ({cp(.9)} cp) |' in block  # conditional on entry, not multiplied by the 60% chapter reach
    overall = text.split('### Most common positions', 1)[1].split('### Openings reached', 1)[0]
    assert '3. Nd2 Nf6' not in overall  # overall still selects the Advance
    assert all(p.read_bytes() == data for p, data in before.items())


def test_first_pass_places_snapshot_notice_first_and_hides_chapter_tables(complete):
    path, report = complete
    Path(report['manifest']['input_path']).write_text('newer study')
    generate([path])
    for name in ('report.md', 'summary.md'):
        content = (path.parent / name).read_text(encoding='utf-8')
        assert content.index('snapshot notice') < content.index('| Repertoire |')
        assert 'source PGN has changed' in content
        assert 'scored ' in content and 'database evidence retrieved' in content
        assert content.count('| White | 50.00% (+0.00 cp) |') == 1
        assert content.count('<details>') == content.count('</details>')
    full = (path.parent / 'report.md').read_text(encoding='utf-8')
    summary = (path.parent / 'summary.md').read_text(encoding='utf-8')
    assert full.index('### White chapters') < full.index('### Vulnerabilities')
    assert '<summary>All 3 White chapters</summary>' in summary
    assert summary.index('<summary>All 3 White chapters</summary>') < summary.index('### Openings reached')
    summary_headings = re.findall(r'^### (.+)$', summary.split('## White repertoire', 1)[1], flags=re.M)
    expected = ['Openings reached', 'Own moves with the largest weighted drag',
                'Unprepared replies with the largest weighted drag', 'Own moves with the largest weighted gain',
                'Largest position contributions', 'Equivalent gap reach']
    assert summary_headings == [heading for heading in expected if heading in summary_headings]
    assert 'Depth and improvement' not in summary and 'cluster-bootstrap' not in summary
    assert 'Strongest and weakest' not in full


def test_merged_stopping_contributions_preserve_score_mass_and_flow_weighted_rating():
    from repertoire_score.consolidated import stopping_contributions
    from repertoire_score.ratings import rating
    def stop(reach, score, games, mean, source):
        return dict(position='same board', parent_position=source, move='m', type='deviation', line=source,
                    reach=reach, score=score, contribution_pp=None if score is None else 100*reach*score,
                    sample_count=games, sparse=False, chapter_attribution={'context_ids': [source]},
                    opponent_rating=rating(1., mean, basis='reply'))
    rows = [stop(.2, .6, 1000, 2000, 'a'), stop(.3, .4, 2, 1600, 'b')]
    merged = stopping_contributions({'stops': rows})
    assert len(merged) == 1 and merged[0]['pooled']
    assert merged[0]['reach'] == .5 and merged[0]['score'] == pytest.approx(.48)
    assert merged[0]['contribution_pp'] == pytest.approx(24)
    assert merged[0]['opponent_rating']['mean'] == pytest.approx(1760)
    assert merged[0]['chapter_attribution']['context_ids'] == ['a', 'b']
    rows.append(stop(.1, None, 0, 1500, 'c'))
    merged = stopping_contributions({'stops': rows})
    assert merged[0]['score'] is None and merged[0]['contribution_pp'] == pytest.approx(24)
    assert merged[0]['known_score_mass'] == .5


def test_saved_comparison_refresh_uses_exact_continuations_and_preserves_rating_contexts(complete, monkeypatch):
    import httpx
    from repertoire_score.vulnerabilities import refresh_saved, write_outputs
    path, report = complete
    def forbidden(*args, **kwargs): raise AssertionError('network forbidden')
    monkeypatch.setattr(httpx.Client, 'request', forbidden)
    old = json.loads(path.with_suffix('.vulnerabilities.json').read_text())
    rating_before = json.loads(path.with_suffix('.ratings.json').read_text())
    unchanged = {p: p.read_bytes() for p in (path, path.with_suffix('.character.json'), path.with_suffix('.preparation.json'))}
    Path(report['manifest']['input_path']).write_text('newer study')
    refreshed = refresh_saved(path)
    assert refreshed['overall']['rankings']['opponent'] == old['overall']['rankings']['opponent']
    character = json.loads(path.with_suffix('.character.json').read_text())
    scopes = {s['id']: {r['position']: r for r in s['positions']} for s in character['scopes']}
    for sid, scope in [('overall', refreshed['overall']), *[(s['id'], s) for s in refreshed['chapters']]]:
        for row in scope['all_signed_rows']:
            if row['kind'] != 'own': continue
            after = scopes[sid][row['target']]['repertoire_score']
            assert row['move_score'] == after
            if after is not None:
                assert row['local_drop_pp'] == pytest.approx(100*(row['reference_score'] - after))
    write_outputs(refreshed, path, refresh_rating_provenance=True)
    rating_after = json.loads(path.with_suffix('.ratings.json').read_text())
    assert rating_after['scopes'] == rating_before['scopes']
    assert rating_after['manifest']['supporting_sha256']['vulnerabilities'] == hashlib.sha256(path.with_suffix('.vulnerabilities.json').read_bytes()).hexdigest()
    generate([path])
    for p, before in unchanged.items(): assert p.read_bytes() == before
    # Changed continuations cannot be reused even when the original score hash matches.
    companion = path.with_suffix('.character.json')
    companion.write_bytes(companion.read_bytes() + b' ')
    with pytest.raises(ValueError, match='continuation analysis changed'):
        generate([path])


def test_saved_comparison_refresh_rejects_stale_rating_provenance_before_writing(complete):
    from repertoire_score.vulnerabilities import refresh_saved, write_outputs
    path, _ = complete
    refreshed = refresh_saved(path)
    output = path.with_suffix('.vulnerabilities.json')
    before = output.read_bytes()
    rating_path = path.with_suffix('.ratings.json')
    ledger = json.loads(rating_path.read_text())
    ledger['manifest']['supporting_sha256']['preparation'] = 'wrong'
    rating_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='supporting analysis changed'):
        write_outputs(refreshed, path, refresh_rating_provenance=True)
    assert output.read_bytes() == before
