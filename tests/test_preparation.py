import chess
import pytest

from repertoire_score.graph import resolve, topology
from repertoire_score.model import prepare, empirical
from repertoire_score.evaluate import backward, KNOWN, UNKNOWN
from repertoire_score.depth import prepared_depth_values, summarize_depth
from repertoire_score.preparation import Evaluator, MissingEvidence, chess_facts, analyze, position_lines
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
    return g, evidence


def test_continuation_matches_core_scorer_and_depth(tmp_path):
    g,evidence=sample(tmp_path)
    t=resolve(g,True,{})
    order=topology(t,g.roots)
    model=prepare(g,t,order,True,evidence)
    raw=empirical(model,True); values=backward(model,order,raw,30)
    depth=summarize_depth(prepared_depth_values(model,order,raw),{g.roots[0]:1})
    evaluator=Evaluator(g,True,evidence,chess_facts(g,True,evidence))
    actual=evaluator.evaluate({g.roots[0]:1})
    assert actual[0] == pytest.approx(values[g.roots[0]][KNOWN,0]) == pytest.approx(.22)
    assert actual[1] == values[g.roots[0]][UNKNOWN,0]
    assert actual[2] == pytest.approx(depth['expected_moves'])
    evaluator.reaches({g.roots[0]:1})


def test_missing_cache_and_zero_observations_are_not_zero_score(tmp_path):
    g,evidence=sample(tmp_path)
    facts=chess_facts(g,True,evidence)
    leaf=position('e4 e5 Nf3 Nc6 Bb5')
    without=dict(evidence); del without[leaf]
    with pytest.raises(MissingEvidence):
        Evaluator(g,True,without,facts).evaluate({g.roots[0]:1})
    without[leaf]=data(0,0,0)
    result=Evaluator(g,True,without,facts).evaluate({g.roots[0]:1})
    assert result[UNKNOWN] == pytest.approx(.8)
    assert result[KNOWN] == pytest.approx(.06)


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
    assert set(result) == {'color', 'scopes', 'manifest', 'validation', 'chapter_catalog'}
    scopes={s['id']:s for s in result['scopes']}
    assert scopes['overall']['score'] == pytest.approx(.4)
    assert scopes['tarrasch']['score'] == pytest.approx(.86)
    assert scopes['tarrasch']['name'] == 'Tarrasch Nf6'
    for scope in scopes.values():
        assert set(scope) <= {'id', 'name', 'starts', 'baseline', 'chapter', 'policy_basis',
                             'stops', 'score', 'prepared_depth', 'status'}
        assert sum(r['contribution_pp'] for r in scope['stops']) == pytest.approx(100*scope['score'])
        assert sum(r['baseline_contribution_pp'] for r in scope['stops']) == pytest.approx(100*(scope['score']-scope['baseline']))
        assert all('chapter_attribution' in row for row in scope['stops'])


@pytest.mark.parametrize('color',[True,False])
def test_transposition_scores_match_core(tmp_path,color):
    g=graph(tmp_path,'1. Nf3 d5 2. g3 Nf6 3. Bg2 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    evidence={}
    for k,n in g.nodes.items():
        moves=list(n.edges)
        evidence[k]=data(60*max(1,len(moves)),0,40*max(1,len(moves)),[(m,60,0,40) for m in moves])
    transitions=resolve(g,color,{})
    order=topology(transitions,g.roots)
    model=prepare(g,transitions,order,color,evidence)
    raw=empirical(model,color); v=backward(model,order,raw,30)
    actual=Evaluator(g,color,evidence,chess_facts(g,color,evidence)).evaluate({g.roots[0]:1})
    assert actual[0] == pytest.approx(v[g.roots[0]][KNOWN,0])


def test_representative_routes_keep_first_transposition_and_custom_root(tmp_path):
    board=chess.Board(); board.push_san('e4'); board.fullmove_number=17
    text='1. Nf3 d5 2. g3 *\n\n1. g3 d5 2. Nf3 *\n\n'
    text+=f'[SetUp "1"]\n[FEN "{board.fen()}"]\n\n17... c5 18. Nc3 *'
    g=graph(tmp_path,text)
    lines=position_lines(g)
    assert lines[position('Nf3 d5 g3')] == '1.Nf3 1...d5 2.g3'
    assert lines[position('e4 c5 Nc3')] == '17...c5 18.Nc3'
    assert set(lines) == set(g.nodes)
