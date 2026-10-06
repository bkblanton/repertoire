import pytest

from repertoire_score.depth import prepared_depth_values, summarize_depth, chapter_prepared_depth
from test_model import graph, position, data, setup


def test_depth_counts_own_moves_and_rewards_breadth_and_depth(tmp_path):
    evidence = {
        position('e4'):data(50,0,50,[('e7e5',40,0,40),('c7c5',10,0,10)]),
        position('e4 e5 Nf3'):data(50,0,50,[('b8c6',25,0,25),('a7a6',15,0,15),('d7d6',10,0,10)]),
        # Missing outcome data at a leaf does not make prepared depth unknown.
        position('e4 e5 Nf3 Nc6 Bc4'):data(0,0,0),
        position('e4 e5 Nf3 a6 Bc4'):data(0,0,0),
    }
    results = []
    for text in ('1. e4 e5 2. Nf3 *',
                 '1. e4 e5 2. Nf3 Nc6 3. Bc4 *',
                 '1. e4 e5 2. Nf3 Nc6 (2... a6 3. Bc4) 3. Bc4 *'):
        g = graph(tmp_path,text)
        model,order,raw,*_ = setup(g,True,evidence)
        values = prepared_depth_values(model,order,raw)
        results.append(summarize_depth(values,{g.roots[0]:1})['expected_moves'])
    assert results == pytest.approx([1.8,2.2,2.44])
    assert values[position('e4 e5 Nf3 Nc6 Bc4')] == (0,0)
    # Entry with our move available counts it; after playing it, it is excluded.
    assert values[position('e4 e5')][0] == pytest.approx(1.8)
    assert values[position('e4 e5 Nf3')][0] == pytest.approx(.8)


def test_missing_distribution_retains_depth_bounds(tmp_path):
    g = graph(tmp_path,'1. e4 e5 2. Nf3 *')
    model,order,raw,*_ = setup(g,True,{position('e4'):data(0,0,0),position('e4 e5 Nf3'):data(0,0,0)})
    values = prepared_depth_values(model,order,raw)
    summary = summarize_depth(values,{g.roots[0]:1})
    assert summary['expected_moves'] is None
    assert summary['conditional_bounds'] == [1,2]
    # With no remaining own moves, missing opponent frequencies cannot affect depth.
    g = graph(tmp_path,'1. e4 e5 *')
    model,order,raw,*_ = setup(g,True,{position('e4'):data(0,0,0),position('e4 e5'):data(0,0,0)})
    assert summarize_depth(prepared_depth_values(model,order,raw),{g.roots[0]:1})['expected_moves'] == 1


def test_chapter_depth_uses_first_entry_weights_and_preserves_missing_weights():
    values = {'a':(2,2),'b':(5,5)}
    summary = chapter_prepared_depth(values,['a','b'],{'first_entry_weights':{'a':.8,'b':.2}})
    assert summary['expected_moves'] == pytest.approx(2.6)
    missing = chapter_prepared_depth(values,['a','b'],{})
    assert missing['expected_moves'] is None
    assert missing['conditional_bounds'] == [2,5]
    assert chapter_prepared_depth(values,['b'],{})['expected_moves'] == 5
    assert chapter_prepared_depth(values,[],{})['expected_moves'] is None
