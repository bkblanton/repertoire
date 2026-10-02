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
    g,evidence=sample(tmp_path)
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
    ranked = {r['position']:r for r in result['positions']}
    assert ranked[deviation['position']]['kind'] == 'unprepared_reply'
    assert ranked[deviation['position']]['to_move'] == 'white'
    assert ranked[deviation['position']]['reach'] == pytest.approx(.2)
    assert deviation['position'] not in evidence
    reply = ranked[deviation['position']]
    assert reply['database_score'] == pytest.approx(.3) and reply['games'] == 20
    assert reply['games_source'] == 'parent_move_rows' and reply['repertoire_score'] is None
    parent = ranked[position('e4 e5 Nf3')]
    assert parent['repertoire_score'] == pytest.approx(.22)  # continuing our preparation
    assert parent['database_score'] == pytest.approx(.38)  # unconstrained database games
    assert parent['games'] == 100
    leaf = ranked[position('e4 e5 Nf3 Nc6 Bb5')]
    assert leaf['kind'] == 'theory_leaf' and leaf['reach'] == pytest.approx(.8)
    assert leaf['repertoire_score'] == pytest.approx(.2) and leaf['games'] == 100

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
    positions = result['positions']
    assert len({r['position'] for r in positions}) == len(positions)
    assert all(r['repertoire_score']==(1 if white else 0) and r['games']==100 for r in positions)
    root = next(r for r in positions if r['position'] == g.roots[0])
    assert root['reach'] == pytest.approx(1)  # our 40/60 mixture does not split the position
    bishop_position = bishops[0]['position']
    shared = [r for r in positions if r['position'] == bishop_position]
    assert len(shared) == 1 and shared[0]['reach'] == pytest.approx(1)
    from repertoire_score.graph import resolve, topology, key
    from repertoire_score.model import prepare, empirical
    from repertoire_score.vulnerabilities import reaches
    transitions = resolve(g, white, policy)
    order = topology(transitions, g.roots)
    model = prepare(g, transitions, order, white, evidence)
    independent = reaches(model, order, empirical(model, white), {g.roots[0]: 1})
    assert {r['position']: r['reach'] for r in positions} == pytest.approx({k:v for k,v in independent.items() if v > 0})
    for row in positions:
        board = chess.Board(g.nodes[g.roots[0]].fen)
        if row['line'] != '(PGN root)':
            for token in row['line'].split():
                board.push_san(token.split('.')[-1])
        assert key(board) == row['position']



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
    ranked = {r['position']:r for r in result['positions']}
    assert ranked[position('e4')]['kind'] == ('opponent_reply' if total else 'unresolved_distribution')
    assert (position('e4 e5') in ranked) == bool(recorded)
    assert ranked[position('e4')]['games'] == total
    assert ranked[position('e4')]['repertoire_score'] == (1 if total else None)



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


def test_transposed_unprepared_reply_combines_reach_and_chapter_context(tmp_path):
    from repertoire_score.attribution import enrich
    g = graph(tmp_path, '1. Nf3 d5 2. g3 e6 3. Bg2 *\n\n1. g3 Nf6 2. Nf3 e6 3. Bg2 *')
    evidence = {k:data(100,0,0,[(m,100//len(n.edges),0,0) for m in n.edges]) for k,n in g.nodes.items()}
    a, b = position('Nf3 d5 g3'), position('g3 Nf6 Nf3')
    evidence[a] = data(20,0,80,[('g8f6',10,0,40),('e7e6',10,0,40)])
    evidence[b] = data(90,0,10,[('d7d5',70,0,5),('e7e6',20,0,5)])
    policy = {g.roots[0]:{'g1f3':.4,'g2g3':.6}}
    result = scope_metrics(Evaluator(g,True,evidence,chess_facts(g,True,evidence),policy),
                           {g.roots[0]:1}, {k:'' for k in g.nodes})
    target = position('Nf3 d5 g3 Nf6')
    assert target not in evidence and target not in g.nodes
    replies = [r for r in result['positions'] if r['position'] == target]
    assert len(replies) == 1 and replies[0]['reach'] == pytest.approx(.65)
    assert replies[0]['database_score'] == pytest.approx((.2*.2 + .45*(70/75))/.65)
    assert replies[0]['games'] == 125
    assert replies[0]['counts_white_draw_black'] == [80,0,45]
    assert sum(o['reach'] for o in replies[0]['unprepared_origins']) == pytest.approx(.65)
    board = chess.Board()
    import re
    for san in re.sub(r'\d+\.(?:\.\.)?', '', replies[0]['line']).split(): board.push_san(san)
    from repertoire_score.graph import key
    assert key(board) == target
    enriched = enrich({'scopes':[dict(id='overall',**result)]},g)
    attribution = next(r for r in enriched['scopes'][0]['positions'] if r['position']==target)['chapter_attribution']
    assert attribution['source_ids'] == [] and attribution['context_ids'] == ['1','2']
    assert attribution['basis'] == 'unprepared move'


def test_position_scores_and_parent_reply_counts_use_black_perspective(tmp_path):
    g = graph(tmp_path,'1. e4 c5 2. Nf3 Nc6 *')
    evidence = {k:data(50,20,30,[(m,50,20,30) for m in n.edges]) for k,n in g.nodes.items()}
    parent = position('e4 c5')
    evidence[parent] = data(60,10,30,[('g1f3',48,8,24),('d2d4',12,2,6)])
    evidence[position('e4 c5 Nf3 Nc6')] = data(20,20,60)
    result = scope_metrics(Evaluator(g,False,evidence,chess_facts(g,False,evidence)),
                           {g.roots[0]:1}, {k:'' for k in g.nodes})
    rows = {r['position']:r for r in result['positions']}
    assert rows[parent]['repertoire_score'] == pytest.approx(.8*.7+.2*.35)
    assert rows[parent]['database_score'] == pytest.approx(.35)
    assert rows[parent]['games'] == 100
    reply = position('e4 c5 d4')
    assert reply not in evidence
    assert rows[reply]['database_score'] == pytest.approx(.35)
    assert rows[reply]['games'] == 20 and rows[reply]['to_move'] == 'black'
    assert rows[reply]['repertoire_score'] is None
