from collections import Counter
import hashlib
import json

import numpy as np
import pytest

from repertoire_score.correlations import connected_groups, cluster_sample, correlation, rank, metrics, bootstrap, analyze, report_rows, markdown


def row(i, cluster, color='white', depth=None, delta=None, baseline=.5):
    depth = i+1 if depth is None else depth
    delta = 2*depth if delta is None else delta
    return dict(id=str(i),color=color,cluster=cluster,family='test',name=str(i),
                depth=depth,delta=delta,baseline=baseline,score=baseline+delta/100,reach=.1)


def test_overlap_groups_include_indirect_links():
    assert connected_groups([{'a','b'},{'b','c'},{'c','d'},{'e'}]) == [[0,1,2],[3]]


def test_cluster_resampling_preserves_pairs_and_color_strata():
    rows = [row(0,'white-1'),row(1,'white-1'),row(2,'white-2'),
            row(3,'black-1','black'),row(4,'black-2','black'),row(5,'black-2','black')]
    sample = cluster_sample(rows,np.random.default_rng(7))
    counts = Counter(r['id'] for r in sample)
    assert counts['0'] == counts['1']
    assert counts['4'] == counts['5']
    assert counts['0'] + counts['2'] == 2
    assert counts['3'] + counts['4'] == 2
    assert all(any(r is original for original in rows) for r in sample)


def test_rank_weighting_and_partial_adjustment():
    assert rank(np.array([3,1,1,4])) == pytest.approx([3,1.5,1.5,4])
    x,y,w = np.array([1,2,5]),np.array([4,3,8]),np.array([1,2,4])
    assert correlation(x,y,w) == pytest.approx(np.corrcoef(np.repeat(x,w),np.repeat(y,w))[0,1])
    baseline = np.array([.3,.4,.5,.6,.7])
    u,v = np.array([1,-2,2,-2,1]),np.array([-1,2,0,-2,1])
    rows = [row(i,str(i),depth=3+baseline[i]+.1*u[i],delta=3+2*baseline[i]+.1*v[i],baseline=baseline[i]) for i in range(5)]
    result = metrics(rows)
    assert result['partial_correlation_controlling_entry_baseline'] == pytest.approx(0,abs=1e-12)
    assert result['pearson'] > .3


def test_bootstrap_reproducibility_perfect_association_and_degenerate_case():
    rows = [row(i,str(i)) for i in range(8)]
    a = bootstrap(rows,200,11)
    b = bootstrap(rows,200,11)
    assert a == b
    assert a['pearson'] == pytest.approx(1)
    assert a['confidence_intervals_95']['pearson']['bounds'] == pytest.approx([1,1])
    assert a['confidence_intervals_95']['depth_vs_entry_baseline']['bounds'] is None
    assert a['confidence_intervals_95']['depth_vs_entry_baseline']['invalid_replicates'] == 200
    with pytest.raises(ValueError,match='two chapter clusters'):
        bootstrap([row(0,'one'),row(1,'one')],200,11)


def test_report_generation_and_source_hash_guard(tmp_path):
    from test_model import position
    source = tmp_path/'study.pgn'
    replies = ['e5','c5','e6','d5','c6']
    source.write_text('\n\n'.join(f'[ChapterURL "https://lichess.org/study/test/{i}"]\n\n1. e4 {reply} 2. Nf3 *'
                                 for i,reply in enumerate(replies)),encoding='utf-8')
    chapters = []
    for i,reply in enumerate(replies):
        item = row(i,str(i),delta=[2,3,1,5,4][i])
        chapters.append({'id':str(i),'name':f'Test {i}', 'entries':[{'position':position(f'e4 {reply}')}],
                         'score':{'prepared_depth':{'expected_moves':item['depth']},'raw_empirical_score':item['score'],'entry_probability':.1},
                         'entry_baseline':{'raw_score':item['baseline'],'difference_pp':item['delta']}})
    report = {'color':'white','chapters':chapters,'manifest':{'input_path':str(source),
               'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'configuration':{},'filters':{}}}
    path = tmp_path/'white.json'
    path.write_text(json.dumps(report),encoding='utf-8')
    before = path.read_bytes()
    result = analyze([path],tmp_path/'correlations',200,7)
    assert result['results']['white']['cluster_count'] == 5
    assert result['results']['white']['n'] == 5
    assert 'pooled' not in result['results']
    assert '95% cluster-bootstrap confidence interval' in markdown(result)
    assert not (tmp_path/'correlations.md').exists()
    assert (tmp_path/'correlations.json').exists()
    assert path.read_bytes() == before
    source.write_text(source.read_text()+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='PGN differs'):
        report_rows(path)
