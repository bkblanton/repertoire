import pytest

from repertoire_score.graph import parse, resolve, topology
from repertoire_score.model import prepare, empirical
from repertoire_score.evaluate import backward, KNOWN, UNKNOWN
from repertoire_score.depth import prepared_depth_values, summarize_depth
from repertoire_score.preparation import Study, Evaluator, MissingEvidence, chess_facts, own_pairs, analyze, render, number
from test_model import graph, data, position
from test_chapter_policies import run_fixture


def sample(tmp_path):
    g=graph(tmp_path,'1. e4 e5 2. Nf3 Nc6 3. Bb5 *')
    evidence={position(''):data(50,0,50),
        position('e4'):data(50,0,50,[('e7e5',50,0,50)]),
        position('e4 e5'):data(60,0,40),
        position('e4 e5 Nf3'):data(38,0,62,[('b8c6',32,0,48),('d7d6',6,0,14)]),
        position('e4 e5 Nf3 Nc6'):data(80,0,20),
        position('e4 e5 Nf3 Nc6 Bb5'):data(20,0,80)}
    return Study(tmp_path/'fixture.pgn'), evidence


def test_all_suffix_cuts_match_full_original_scorer_and_depth(tmp_path):
    study,evidence=sample(tmp_path)
    facts=chess_facts(study.original,True,evidence)
    for cuts in [(), *study.candidates()]:
        g=study.graph(cuts)
        t=resolve(g,True,{})
        order=topology(t,g.roots)
        model=prepare(g,t,order,True,evidence)
        raw=empirical(model,True); values=backward(model,order,raw,30)
        depth=summarize_depth(prepared_depth_values(model,order,raw),{g.roots[0]:1})
        evaluator=Evaluator(g,True,evidence,facts)
        actual=evaluator.evaluate({g.roots[0]:1})
        assert actual[0] == pytest.approx(values[g.roots[0]][KNOWN,0])
        assert actual[1] == values[g.roots[0]][UNKNOWN,0]
        assert actual[2] == pytest.approx(depth['expected_moves'])
        evaluator.reaches({g.roots[0]:1})
    base=Evaluator(study.graph(),True,evidence,facts).evaluate({study.original.roots[0]:1})
    last=len(study.records)-1
    cut=Evaluator(study.graph((last,)),True,evidence,facts).evaluate({study.original.roots[0]:1})
    assert base[0] == pytest.approx(.22)
    assert cut[0] == pytest.approx(.70)
    assert 100*(base[0]-cut[0]) == pytest.approx(-48)


def test_duplicate_providers_require_coordinated_trim(tmp_path):
    graph(tmp_path,'1. e4 e5 2. Nf3 *\n\n1. e4 e5 2. Nf3 *')
    study=Study(tmp_path/'fixture.pgn')
    facts=chess_facts(study.original,True,{})
    before=own_pairs(study.original,True,facts)
    leaves=[i for i,r in enumerate(study.records) if not r.children]
    assert before == own_pairs(study.graph((leaves[0],)),True,facts)
    assert len(before-own_pairs(study.graph(tuple(leaves)),True,facts)) == 1
    assert tuple(leaves) in list(study.candidates())


def test_graph_cuts_match_actual_pgn_deletion_and_first_choice_switch(tmp_path):
    graph(tmp_path,'1. e4 (1. d4 d5) e5 2. Nf3 *\n\n1. d4 Nf6 *')
    study=Study(tmp_path/'fixture.pgn')
    cut=next(i for i,r in enumerate(study.records) if r.path==['e4'])
    g=study.graph((cut,))
    actual=tmp_path/'actual.pgn'
    actual.write_text('1. d4 d5 *\n\n1. d4 Nf6 *')
    expected=parse(actual)
    assert {k:n.edges for k,n in g.nodes.items()} == {k:n.edges for k,n in expected.nodes.items()}
    assert list(resolve(g,True,{})[g.roots[0]]) == ['d2d4']


def test_missing_cache_and_zero_observations_are_not_zero_score(tmp_path):
    study,evidence=sample(tmp_path)
    facts=chess_facts(study.original,True,evidence)
    leaf=position('e4 e5 Nf3 Nc6 Bb5')
    without=dict(evidence); del without[leaf]
    with pytest.raises(MissingEvidence):
        Evaluator(study.graph(),True,without,facts).evaluate({study.original.roots[0]:1})
    without[leaf]=data(0,0,0)
    result=Evaluator(study.graph(),True,without,facts).evaluate({study.original.roots[0]:1})
    assert result[UNKNOWN] == pytest.approx(.8)
    assert result[KNOWN] == pytest.approx(.06)


def test_tiny_positive_cost_is_not_displayed_as_zero():
    assert float(number(.000001)) == pytest.approx(.000001)
    assert number(.000001) != number(0.)


def test_cycle_is_rejected_even_when_observed_probability_is_zero(tmp_path):
    g=graph(tmp_path,'1. Nf3 Nf6 2. Ng1 Ng8 *')
    evidence={k:data(10,0,0,[]) for k in g.nodes}
    facts=chess_facts(g,True,evidence)
    with pytest.raises(ValueError,match='cycle'):
        Evaluator(g,True,evidence,facts).evaluate({g.roots[0]:1})


def test_report_uses_cache_keeps_alternative_chapter_and_source(tmp_path,monkeypatch):
    score_report,cache=run_fixture(tmp_path,monkeypatch,common_entry=True)
    before=(tmp_path/'fixture.pgn').read_bytes()
    result=analyze(tmp_path/'white.json',cache)
    assert result['manifest']['network_requests'] == 0
    assert result['validation']['original_scores_reproduced']
    assert (tmp_path/'fixture.pgn').read_bytes() == before
    scopes={s['id']:s for s in result['scopes']}
    assert scopes['overall']['score'] == pytest.approx(.4)
    assert scopes['tarrasch']['score'] == pytest.approx(.86)
    for scope in scopes.values():
        assert sum(r['contribution_pp'] for r in scope['stops']) == pytest.approx(100*scope['score'])
        assert sum(r['baseline_contribution_pp'] for r in scope['stops']) == pytest.approx(100*(scope['score']-scope['baseline']))
        for row in scope['trims']:
            if row['memorization_value_pp'] is not None:
                assert row['trimming_gain_pp'] == -row['memorization_value_pp']
                assert row['memorization_value_pp'] == pytest.approx(100*(scope['score']-row['score_after']))
                assert row['value_per_move_pp'] == pytest.approx(row['memorization_value_pp']/row['distinct_own_moves_saved'])
    text=render(result,3,1)
    assert 'Tarrasch Nf6' in text and 'Trims that improve' in text and '\u2014' not in text
    assert result['proposals']


@pytest.mark.parametrize('color',[True,False])
def test_transposition_and_opponent_branch_deletion_match_core(tmp_path,color):
    g=graph(tmp_path,'1. Nf3 d5 2. g3 Nf6 3. Bg2 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    study=Study(tmp_path/'fixture.pgn')
    evidence={}
    for k,n in g.nodes.items():
        moves=list(n.edges)
        evidence[k]=data(60*max(1,len(moves)),0,40*max(1,len(moves)),[(m,60,0,40) for m in moves])
    facts=chess_facts(g,color,evidence)
    for cuts in [(), *study.candidates()]:
        trial=study.graph(cuts); transitions=resolve(trial,color,{})
        order=topology(transitions,trial.roots)
        model=prepare(trial,transitions,order,color,evidence)
        raw=empirical(model,color); v=backward(model,order,raw,30)
        fast=Evaluator(trial,color,evidence,facts).evaluate({trial.roots[0]:1})
        assert fast[0] == pytest.approx(v[trial.roots[0]][KNOWN,0])
