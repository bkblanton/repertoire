import hashlib
import json

import pytest

from repertoire_score.attribution import Attribution, enrich, chapter_text, regenerate, write_text
from repertoire_score.graph import parse
from test_model import graph, position
from test_chapter_policies import run_fixture


def test_recorded_move_sources_do_not_inherit_all_parent_chapters(tmp_path):
    g=graph(tmp_path,'[ChapterName "E-pawn"]\n\n1. e4 e5 *\n\n[ChapterName "D-pawn"]\n\n1. d4 d5 *')
    attribution=Attribution(g)
    row=attribution.position_or_move(position(''),'e2e4')
    assert row['source_ids']==['1']
    assert row['context_ids']==['1','2']
    label=chapter_text({'chapter_attribution':row},attribution.catalog)
    assert label=='E-pawn' and 'D-pawn' not in label
    unprepared=attribution.position_or_move(position(''),'a2a3')
    assert unprepared['source_ids']==[]
    assert chapter_text({'chapter_attribution':unprepared},attribution.catalog)=='Unprepared; parent: E-pawn; D-pawn'


def test_shared_moves_and_unrecorded_transposition(tmp_path):
    g=graph(tmp_path,'[ChapterName "First"]\n\n1. Nf3 d5 2. g3 Nf6 3. Bg2 *\n\n[ChapterName "Second"]\n\n1. g3 Nf6 2. Nf3 e6 *\n\n[ChapterName "Copy"]\n\n1. Nf3 d5 *')
    attribution=Attribution(g)
    shared=attribution.position_or_move(position(''),'g1f3')
    assert shared['source_ids']==['1','3']
    transposition=attribution.position_or_move(position('g3 Nf6 Nf3'),'d7d5')
    assert transposition['source_ids']==[]
    assert transposition['context_ids']==['2']
    assert transposition['transposition_ids']==['1']
    assert chapter_text({'chapter_attribution':transposition},attribution.catalog)=='Transposition into: First'
    excluded=parse(tmp_path/'fixture.pgn',['3'])
    assert Attribution(excluded).position_or_move(position(''),'g1f3')['source_ids']==['1']


def test_links_and_table_escaping():
    catalog=[dict(id='a',name='A | [test]',url='https://lichess.org/study/example/a')]
    row={'chapter_attribution':dict(source_ids=['a'],context_ids=[],transposition_ids=[],basis='position')}
    text=chapter_text(row,catalog)
    assert text=='[A &#124; &#91;test&#93;](https://lichess.org/study/example/a)'
    assert chapter_text({'chapter_attribution':dict(source_ids=[],context_ids=[],transposition_ids=[],basis='position')},catalog)=='None'


def test_atomic_refresh_retries_sync_lock_without_truncating_previous_file(tmp_path,monkeypatch):
    import os
    target=tmp_path/'report.json'
    target.write_text('previous')
    replace=os.replace
    attempts=[]
    def locked_once(source,destination):
        attempts.append(source)
        if len(attempts)==1:
            assert target.read_text()=='previous'
            raise OSError(22,'Temporary synchronization lock')
        replace(source,destination)
    monkeypatch.setattr(os,'replace',locked_once)
    write_text(target,'complete replacement')
    assert target.read_text()=='complete replacement'
    assert len(attempts)==2
    assert not list(tmp_path.glob('.chapter-attribution-*.tmp'))


def test_stopping_sources_include_shared_moves_and_specific_providers(tmp_path):
    g=graph(tmp_path,'1. e4 e5 *\n\n1. e4 c5 *')
    shared=dict(parent_position=position(''),move='e2e4')
    specific=dict(parent_position=position('e4'),move='e7e5')
    report=dict(scopes=[dict(stops=[shared,specific])])
    enrich(report,g)
    assert shared['chapter_attribution']['source_ids']==['1','2']
    assert specific['chapter_attribution']['source_ids']==['1']


def without_attribution(item):
    if isinstance(item,list): return [without_attribution(x) for x in item]
    if isinstance(item,dict):
        return {k:without_attribution(v) for k,v in item.items()
                if k not in ('chapter_catalog','chapter_attribution','example_chapter_attribution','report_sha256')}
    return item


def test_refresh_all_saved_reports_preserves_metrics_and_rejects_stale_source(tmp_path,monkeypatch):
    from repertoire_score.vulnerabilities import analyze as vulnerabilities
    from repertoire_score.preparation import analyze as preparation
    from repertoire_score.character import analyze as character
    score,cache=run_fixture(tmp_path,monkeypatch,common_entry=True)
    score_path=tmp_path/'white.json'
    companions={kind:fn(score_path,cache) for kind,fn in
                [('vulnerabilities',vulnerabilities),('preparation',preparation),('character',character)]}
    companions['character']['report_filename']='white.character.md'
    # Exercise compatibility with saved reports from before provenance was added.
    original_score=without_attribution(score)
    score_path.write_text(json.dumps(original_score,indent=2))
    digest=hashlib.sha256(score_path.read_bytes()).hexdigest()
    originals={}
    for kind,report in companions.items():
        report=without_attribution(report)
        report['manifest']['report_sha256']=digest
        originals[kind]=without_attribution(report)
        score_path.with_suffix(f'.{kind}.json').write_text(json.dumps(report,indent=2))
    correlation_path=tmp_path/'depth-delta-correlation.json'
    correlation=dict(provenance={'white':dict(report_path=str(score_path.resolve()),report_sha256=digest)},results={'r':.25})
    correlation_path.write_text(json.dumps(correlation))
    improvement_path=tmp_path/'white-improvements.json'
    improvement=dict(input_sha256=score['manifest']['input_sha256'],nonoverlapping_change_contributions=[
        dict(position=position('e4 e6 d4 d5 e5'),representative_line='1.e4 e6 2.d4 d5 3.e5',contribution_pp=.25)])
    improvement_path.write_text(json.dumps(improvement))
    improvement_path.with_suffix('.md').write_text('| First changed continuation | Change |\n|---|---|\n| 1.e4 e6 2.d4 d5 3.e5 | Added |\n')
    source_before=(tmp_path/'fixture.pgn').read_bytes()
    regenerate([score_path])
    refreshed=json.loads(score_path.read_text())
    assert without_attribution(refreshed)==original_score
    for kind in companions:
        report=json.loads(score_path.with_suffix(f'.{kind}.json').read_text())
        assert without_attribution(report)==originals[kind]
        assert report['chapter_catalog']
        assert report['manifest']['report_sha256']==hashlib.sha256(score_path.read_bytes()).hexdigest()
        assert not score_path.with_suffix(f'.{kind}.md').exists()
    text=(tmp_path/'report.md').read_text(encoding='utf-8')
    assert 'Chapter source' in text
    assert '\u2014' not in text
    assert not score_path.with_suffix('.md').exists()
    assert not (tmp_path/'vulnerabilities.md').exists()
    refreshed_correlation=json.loads(correlation_path.read_text())
    assert refreshed_correlation['results']==correlation['results']
    assert refreshed_correlation['provenance']['white']['report_sha256']==hashlib.sha256(score_path.read_bytes()).hexdigest()
    refreshed_improvement=json.loads(improvement_path.read_text())
    assert without_attribution(refreshed_improvement)==improvement
    assert 'Source chapters' in improvement_path.with_suffix('.md').read_text()
    assert '[Advance](https://lichess.org/study/test/advance)' in improvement_path.with_suffix('.md').read_text()
    assert (tmp_path/'fixture.pgn').read_bytes()==source_before
    saved=score_path.read_bytes()
    saved_improvement=improvement_path.with_suffix('.md').read_bytes()
    regenerate([score_path])
    assert score_path.read_bytes()==saved
    assert improvement_path.with_suffix('.md').read_bytes()==saved_improvement
    (tmp_path/'fixture.pgn').write_bytes(source_before+b'\n')
    with pytest.raises(ValueError,match='PGN changed'):
        regenerate([score_path])
    assert score_path.read_bytes()==saved
