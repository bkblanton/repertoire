import chess
import pytest

from repertoire_score.character import (analyze, board_fingerprint, entropy, exposure,
                                       position_profile, render, reuse_metrics, scope_metrics, summary)
from repertoire_score.preparation import Evaluator, chess_facts
from test_model import graph, data, position
from test_preparation import sample
from test_chapter_policies import run_fixture


def test_reuse_exact_independent_games_and_numerical_stability():
    decisions=[dict(position='a',move='x',reach=1),dict(position='b',move='x',reach=.2)]
    r=reuse_metrics(decisions,[1,2,100])
    assert r['expected_encounters_per_game'] == pytest.approx(1.2)
    assert r['curve'][0]['expected_distinct_decisions'] == pytest.approx(1.2)
    assert r['curve'][1]['expected_distinct_decisions'] == pytest.approx(1.36)
    assert r['curve'][1]['expected_repeat_encounters'] == pytest.approx(1.04)
    assert exposure(1,0) == 0
    assert exposure(0,100) == 0
    assert exposure(1e-20,500) == pytest.approx(5e-18,rel=1e-12,abs=0)
    with pytest.raises(ValueError,match='Duplicate'):
        reuse_metrics(decisions+decisions)


def test_known_branching_entropy_and_depth(tmp_path):
    study,evidence=sample(tmp_path)
    g=study.original
    evaluator=Evaluator(g,True,evidence,chess_facts(g,True,evidence))
    result=scope_metrics(evaluator,{g.roots[0]:1},{k:' '.join(n.path) for k,n in g.nodes.items()},[1,2])
    assert result['reuse']['expected_encounters_per_game'] == pytest.approx(2.8)
    assert result['reuse']['curve'][1]['expected_distinct_decisions'] == pytest.approx(2.96)
    p=result['predictability']
    assert p['mean_entropy_bits'] == pytest.approx(entropy([.8,.2])/2)
    assert p['effective_replies'] == pytest.approx(2**(entropy([.8,.2])/2))
    assert p['expected_reply_information_bits'] == pytest.approx(entropy([.8,.2]))
    assert p['recorded_reply_coverage'] == 1
    assert result['position_profiles']['deviation']['scope_mass'] == pytest.approx(.2)
    assert result['position_profiles']['theory_leaf']['scope_mass'] == pytest.approx(.8)
    deviation=next(r for r in result['stopping_outcomes'] if r['type']=='deviation')
    assert deviation['position'] == position('e4 e5 Nf3 d6')
    for distribution in result['position_profiles']['all']['distributions'].values():
        assert sum(r['probability'] for r in distribution) == pytest.approx(1)


@pytest.mark.parametrize('white',[True,False])
def test_transposition_counts_one_shared_decision(tmp_path,white):
    g=graph(tmp_path,'1. Nf3 d5 2. g3 Nf6 3. Bg2 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    evidence={k:data(100,0,0,[(m,100//len(n.edges),0,0) for m in n.edges]) for k,n in g.nodes.items()}
    policy={g.roots[0]:{'g1f3':.4,'g2g3':.6}}
    if not white:
        from repertoire_score.graph import key, Graph, Node
        mapping={k:key(chess.Board(n.fen).mirror()) for k,n in g.nodes.items()}
        def mirror_move(m):
            move=chess.Move.from_uci(m)
            return chess.Move(chess.square_mirror(move.from_square),chess.square_mirror(move.to_square)).uci()
        evidence={mapping[k]:data(d['white'],0,0,[(mirror_move(r['uci']),r['white'],0,0) for r in d['moves']]) for k,d in evidence.items()}
        nodes={mapping[k]:Node(chess.Board(n.fen).mirror().fen(),[],edges={mirror_move(m):mapping[t] for m,t in n.edges.items()}) for k,n in g.nodes.items()}
        policy={mapping[k]:{mirror_move(m):p for m,p in moves.items()} for k,moves in policy.items()}
        g=Graph(nodes,[],[mapping[g.roots[0]]])
    evaluator=Evaluator(g,white,evidence,chess_facts(g,white,evidence),policy)
    result=scope_metrics(evaluator,{g.roots[0]:1},{k:'' for k in g.nodes},[1,2])
    assert result['reuse']['reachable_distinct_decisions'] == 5
    assert result['reuse']['expected_encounters_per_game'] == pytest.approx(3)
    assert result['reuse']['curve'][1]['expected_distinct_decisions'] == pytest.approx(3.96)
    bishops=[r for r in result['reuse']['decisions'] if r['san'] in ('Bg2','Bg7')]
    assert len(bishops) == 1 and bishops[0]['reach'] == pytest.approx(1)


@pytest.mark.parametrize('total,recorded,expected_coverage',[(100,50,.5),(100,0,0),(0,0,0)])
def test_unrecorded_and_zero_data_are_not_invented_replies(tmp_path,total,recorded,expected_coverage):
    g=graph(tmp_path,'1. e4 e5 *')
    evidence={k:data(total,0,0) for k in g.nodes}
    evidence[position('e4')]=data(total,0,0,[('e7e5',recorded,0,0)])
    result=scope_metrics(Evaluator(g,True,evidence,chess_facts(g,True,evidence)),
                         {g.roots[0]:1},{k:'' for k in g.nodes})
    p=result['predictability']
    assert p['recorded_reply_coverage'] == expected_coverage
    assert p['effective_replies'] == (1 if recorded else None)
    assert result['reuse']['expected_encounters_per_game'] == 1
    assert result['unresolved_opponent_distribution_mass'] == (0 if total else 1)


def test_no_opponent_opportunity_is_unavailable_not_perfect_predictability(tmp_path):
    g=graph(tmp_path,'1. e4 *')
    evidence={k:data(100,0,0) for k in g.nodes}
    result=scope_metrics(Evaluator(g,True,evidence,chess_facts(g,True,evidence)),
                         {g.roots[0]:1},{k:'' for k in g.nodes})
    assert result['predictability']['effective_replies'] is None
    assert result['predictability']['recorded_reply_coverage'] is None
    assert result['reuse']['reachable_distinct_decisions'] == 1
    assert all(row['expected_distinct_decisions']==1 for row in result['reuse']['curve'])


def test_board_features_and_weighted_skeletons():
    board=chess.Board('b5k1/8/2p1p3/3p4/2BP4/P7/P7/2KQ1B2 w - - 0 1')
    from repertoire_score.graph import key
    cats,flags=board_fingerprint(key(board),True)
    assert cats['queens']=='own queen only'
    assert cats['king_placement']=='opposite wings'
    assert flags['own_isolated_d_pawn'] and flags['own_doubled'] and flags['own_passed']
    assert flags['own_bishop_pair'] and not flags['opponent_bishop_pair']
    reverse=board_fingerprint(key(board),False)
    assert reverse[0]['queens']=='opponent queen only'
    assert reverse[1]['opponent_isolated_d_pawn']
    stops=[dict(position=key(board),reach=.25,line='custom'),dict(position=position(''),reach=.75,line='root')]
    p=position_profile(stops,True)
    assert p['features']['own_isolated_d_pawn']==.25
    assert p['effective_pawn_structures']==pytest.approx(2**entropy([.25,.75]))
    assert p['distinct_pawn_structures']==2
    assert sum(r['probability'] for r in p['distributions']['pawn_structure'])==1


@pytest.mark.parametrize('multiple_entries',[False,True])
def test_offline_report_alternatives_entries_and_source_guard(tmp_path,monkeypatch,multiple_entries):
    saved,cache=run_fixture(tmp_path,monkeypatch,common_entry=not multiple_entries,multiple_entries=multiple_entries)
    before=(tmp_path/'fixture.pgn').read_bytes()
    result=analyze(tmp_path/'white.json',cache)
    assert result['manifest']['network_requests']==0
    assert (tmp_path/'fixture.pgn').read_bytes()==before
    expected={'overall':saved['overall'],**{c['id']:c['score'] for c in saved['chapters']}}
    for scope in result['scopes']:
        assert scope['reuse']['expected_encounters_per_game']==pytest.approx(expected[scope['id']]['prepared_depth']['expected_moves'])
        assert scope['validation']['resolved_score']==pytest.approx(expected[scope['id']]['resolved_contribution'])
    assert 'Tarrasch Nf6 (alternative)' in render(result)
    assert '\u2014' not in render(result)
    result['report_filename']='white.character.md'
    overview=summary([result])
    assert 'Positions where preparation ends' in overview
    assert 'Chapter reply predictability' in overview
    assert 'white.character.md' in overview
    (tmp_path/'fixture.pgn').write_bytes(before+b'\n')
    with pytest.raises(ValueError,match='PGN changed'):
        analyze(tmp_path/'white.json',cache)
