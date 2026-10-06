

from repertoire_score.attribution import Attribution, enrich, chapter_text, write_text
from repertoire_score.graph import parse
from test_model import graph, position


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
