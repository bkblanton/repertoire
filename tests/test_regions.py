import chess
import pytest

from repertoire_score.baseline import chapter_entry_baseline
from repertoire_score.graph import chapter_region, region_entries, resolve
from repertoire_score.evaluate import chapter_score, forward
from repertoire_score.model import Branch, ModelNode
from repertoire_score.transitions import chapter_transitions, hitting_bounds
from test_model import graph, position, data, setup


def test_late_transposition_counts_once_and_reweights_score_and_baseline(tmp_path):
    g = graph(tmp_path, '''[ChapterURL "https://lichess.org/study/test/quiet"]

1. e4 e5 2. Nc3 Nf6 3. g3 Nc6 4. Bg2 Bc5 5. Nge2 *

[ChapterURL "https://lichess.org/study/test/late"]

1. e4 e5 2. Nc3 Nf6 3. g3 Bc5 4. Bg2 Nc6 5. Nge2 d6 6. h3 *''')
    t = resolve(g, True, {})
    early = position('e4 e5 Nc3 Nf6 g3 Nc6')
    late = position('e4 e5 Nc3 Nf6 g3 Bc5 Bg2 Nc6')
    region = chapter_region(g, 'quiet', [early])
    assert late in region
    assert position('e4 e5 Nc3 Nf6 g3') not in region
    assert position('e4 e5 Nc3 Nf6 g3 Bc5 Bg2 Nc6 Nge2 d6') not in region
    entries = region_entries(t, g.roots, region)
    assert set(entries) == {early, late}
    evidence = {}
    for k, node in g.nodes.items():
        if not t[k]:
            evidence[k] = data(80, 0, 20)
        elif not chess.Board(node.fen).turn:
            moves = [(move, 50, 0, 50) for move in t[k]]
            evidence[k] = data(50*len(moves), 0, 50*len(moves), moves)
    evidence[position('e4')] = data(50, 0, 50, [('e7e5',40,0,40),('c7c5',10,0,10)])
    evidence[position('e4 e5 Nc3 Nf6 g3')] = data(50,0,50,[('b8c6',10,0,15),('f8c5',25,0,25),('d7d5',15,0,10)])
    evidence[position('e4 e5 Nc3 Nf6 g3 Bc5 Bg2')] = data(50,0,50,[('b8c6',20,0,20),('d7d6',30,0,30)])
    evidence[early] = data(60,0,40)
    evidence[late] = data(20,0,80)
    m,o,r,s,v,p = setup(g,True,evidence)
    result = chapter_score(m,o,r,v,{g.roots[0]:1},entries,s)
    assert result['entry_probability'] == pytest.approx(.36)
    assert result['first_entry_weights'][early] == pytest.approx(5/9)
    assert result['first_entry_weights'][late] == pytest.approx(4/9)
    assert result['raw_empirical_score'] == pytest.approx(.8)
    baseline = chapter_entry_baseline(entries,result,evidence,True,{})
    assert baseline['raw_score'] == pytest.approx((5*.6+4*.2)/9)
    # Passing every region member gives the same reach and score, with no
    # double counting at the shared junction or the following own move.
    expanded = chapter_score(m,o,r,v,{g.roots[0]:1},region,s)
    assert expanded['entry_probability'] == pytest.approx(result['entry_probability'])
    assert expanded['raw_empirical_score'] == pytest.approx(result['raw_empirical_score'])
    assert sum(x[0] for x in forward(m,o,r,{g.roots[0]:1})[0].values()) == pytest.approx(1)
    with pytest.raises(ValueError, match='does not belong'):
        chapter_region(g,'quiet',[position('e4 e5 Nc3 Nf6 g3 Bc5')])
    with pytest.raises(ValueError, match='requires anchors'):
        chapter_region(g,'quiet',[])


def test_transition_direction_and_simultaneous_entry():
    # Both histories visit A and B, but their order differs. An intersection
    # calculation would incorrectly report 100% in both directions.
    m = {'root':ModelNode('own',[Branch(target='a1'),Branch(target='b1')]),
         'a1':ModelNode('own',[Branch(target='b2')]),
         'b1':ModelNode('own',[Branch(target='a2')]),
         'a2':ModelNode('stop',[Branch(kind='theory_leaf')]),
         'b2':ModelNode('stop',[Branch(kind='theory_leaf')])}
    order = ['b2','a1','a2','b1','root']
    sampled = {k:[(1.,None)] for k in m}
    sampled['root'] = [(.5,None),(.5,None)]
    chapters = [{'id':cid,'name':cid,'score':{'entry_probability':1.,'first_entry_weights':weights}}
                for cid,weights in [('A',{'a1':.5,'a2':.5}),('B',{'b1':.5,'b2':.5})]]
    destinations = {'A':{'a1','a2'},'B':{'b1','b2'}}
    rows = chapter_transitions(m,order,sampled,chapters,destinations)
    assert all(row['conditional_probability'] == .5 for row in rows)
    destinations['B'].add('a2')
    row = next(row for row in chapter_transitions(m,order,sampled,chapters,destinations) if row['source_id']=='A')
    assert row['conditional_probability'] == 1.


def test_transition_missing_distribution_remains_unresolved():
    m = {'a':ModelNode('stop',[Branch(kind='unresolved_distribution')],potential_targets=['b']),
         'b':ModelNode('stop',[Branch(kind='theory_leaf')])}
    sampled = {'a':[(1.,None)],'b':[(1.,.5)]}
    assert hitting_bounds(m,['b','a'],sampled,{'b'})['a'] == (0.,1.)
    assert hitting_bounds(m,['b','a'],sampled,{'a'})['a'] == (1.,1.)
