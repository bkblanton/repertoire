import json
from types import SimpleNamespace
import chess
import numpy as np
import pytest
from repertoire_score.graph import parse, key, resolve, topology, conflicts, infer_entries
from repertoire_score.explorer import validate, Explorer
from repertoire_score.model import prepare, empirical
from repertoire_score.uncertainty import Posterior
from repertoire_score.evaluate import backward, forward, summarize, chapter_score, KNOWN, UNKNOWN, COMPLETED, LEAF, DEVIATION, OTHER


def graph(tmp_path, text):
    path = tmp_path / 'fixture.pgn'
    path.write_text(text, encoding='utf-8')
    return parse(path)


def position(moves):
    b = chess.Board()
    for m in moves.split():
        b.push_san(m)
    return key(b)


def data(w, d, b, moves=()):
    return dict(white=w, draws=d, black=b, moves=[dict(uci=m, white=a, draws=c, black=e) for m,a,c,e in moves])


def setup(g, color, evidence, policy=None):
    """Model, order, empirical sample, Posterior, empirical values and posterior-mean values."""
    t = resolve(g, color, policy or {})
    order = topology(t, g.roots)
    model = prepare(g, t, order, color, evidence)
    raw = empirical(model, color)
    values = backward(model, order, raw, 30)
    posterior = Posterior(model, order, color, [0.5]*3, 30)
    return model, order, raw, posterior, values, posterior.values


def test_forced_own_move_deviations_and_conservation(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nf3 *')
    e4, leaf = position('e4'), position('e4 e5 Nf3')
    # Own e4 and Nf3 are forced. No statistics for their popularity are used.
    evidence = {e4:data(60,0,40,[('e7e5',40,0,40),('c7c5',20,0,0)]), leaf:data(50,0,50)}
    m,o,r,s,v,p = setup(g, True, evidence)
    root = g.roots[0]
    assert v[root][KNOWN,0] == pytest.approx(.8*.5+.2*1)
    stops,_ = forward(m,o,r,{root:1})
    assert sum(x[0] for x in stops.values()) == pytest.approx(1)
    assert sum(stops[k,j][0]*(r[k][j][1] or 0) for k,j in stops) == pytest.approx(.6)
    assert np.allclose(p[root][UNKNOWN]+p[root][LEAF]+p[root][DEVIATION]+p[root][OTHER],1)
    assert np.allclose(sum(flow*s.sample[k][j][1] for (k,j),flow in forward(m,o,s.sample,{root:1})[0].items()),p[root][COMPLETED])
    evidence[e4] = data(40,0,60,[('e7e5',40,0,40),('c7c5',0,0,20)])
    assert setup(g, True, evidence)[4][root][KNOWN,0] == pytest.approx(.4)


def test_sparse_zero_leaf_and_zero_distribution(tmp_path):
    g = graph(tmp_path, '1. e4 e5 2. Nf3 *')
    evidence = {position('e4'):data(80,0,20,[('e7e5',70,0,10),('c7c5',10,0,10)]),position('e4 e5 Nf3'):data(0,0,0)}
    m,o,r,s,v,p = setup(g, True, evidence)
    summary = summarize(v[g.roots[0]],s.mixture({g.roots[0]: 1.}))
    assert summary['raw_empirical_score'] is None
    assert summary['conditional_bounds'] == pytest.approx([.1,.9])
    evidence[position('e4')] = data(0,0,0)
    m,o,r,s,v,p = setup(g, True, evidence)
    assert v[g.roots[0]][UNKNOWN,0] == 1
    assert len(m[position('e4')].branches) == 1
    assert m[position('e4')].mode == 'stop'


def test_unobserved_moves_have_posterior_mass(tmp_path):
    g = graph(tmp_path, '1. e4 e5 *')
    m,o,r,s,v,p = setup(g,True,{position('e4'):data(1,0,0,[('e7e5',1,0,0)]),position('e4 e5'):data(1,0,0)})
    assert v[g.roots[0]][UNKNOWN,0] == 0
    assert p[g.roots[0]][UNKNOWN].mean() > 0


def test_residual_not_deviation_and_invalid_counts(tmp_path):
    g = graph(tmp_path,'1. e4 e5 *')
    e = {position('e4'):data(8,1,1,[('e7e5',5,1,1)]),position('e4 e5'):data(5,0,5)}
    m,o,r,s,v,p = setup(g,True,e)
    assert v[g.roots[0]][OTHER,0] == pytest.approx(.3)
    assert v[g.roots[0]][DEVIATION,0] == 0
    with pytest.raises(ValueError,match='Inconsistent'):
        validate(data(0,0,0,[('e7e5',1,0,0)]),position('e4'))
    with pytest.raises(ValueError):
        validate(data(-1,0,0),position('e4'))
    with pytest.raises(ValueError):
        validate(dict(white=1,draws=0,black=0),position('e4'))


def test_duplicate_chapters_and_transposing_pgn(tmp_path):
    text = '1. Nf3 d5 2. g3 Nf6 *'
    g1 = graph(tmp_path,text)
    g2 = graph(tmp_path,text+'\n\n'+text)
    assert len(g1.nodes) == len(g2.nodes)
    for g in [g1,g2]:
        model = setup(g,True,{position('Nf3'):data(10,0,0,[('d7d5',10,0,0)]),position('Nf3 d5 g3'):data(10,0,0,[('g8f6',10,0,0)]),position('Nf3 d5 g3 Nf6'):data(3,2,5)})
        assert model[4][g.roots[0]][KNOWN,0] == pytest.approx(.4)
    # Same final position through another move order merges exactly.
    g = graph(tmp_path, '1. Nf3 d5 2. g3 Nf6 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    junction = position('Nf3 d5 g3 Nf6')
    assert junction == position('g3 Nf6 Nf3 d5')
    assert len(g.nodes[junction].chapters) == 2
    assert g.nodes[junction].edges == {'f1g2':position('g3 Nf6 Nf3 d5 Bg2')}


def test_transposition_incoming_mass_and_first_entry(tmp_path):
    from repertoire_score.depth import prepared_depth_values, summarize_depth, chapter_prepared_depth
    g = graph(tmp_path,'1. Nf3 d5 2. g3 Nf6 3. Bg2 *\n\n1. g3 Nf6 2. Nf3 d5 3. Bg2 *')
    root = g.roots[0]
    policy = {root:{'g1f3':.5,'g2g3':.5}}
    evidence = {position('Nf3'):data(10,0,0,[('d7d5',10,0,0)]),position('Nf3 d5 g3'):data(10,0,0,[('g8f6',10,0,0)]),
                position('g3'):data(10,0,0,[('g8f6',10,0,0)]),position('g3 Nf6 Nf3'):data(10,0,0,[('d7d5',10,0,0)]),
                position('Nf3 d5 g3 Nf6 Bg2'):data(3,2,5)}
    m,o,r,s,v,p = setup(g,True,evidence,policy)
    flow,_ = forward(m,o,r,{root:1})
    leaf = position('Nf3 d5 g3 Nf6 Bg2')
    assert sum(mass[0] for (k, _), mass in flow.items() if k == leaf) == pytest.approx(1)
    chapter = chapter_score(m,o,r,v,{root:1},[position('Nf3'),position('g3'),leaf],s)
    assert chapter['entry_probability'] == pytest.approx(1)
    assert chapter['first_entry_weights'][leaf] == 0
    assert chapter['raw_empirical_score'] == pytest.approx(.4)
    depths = prepared_depth_values(m,o,r)
    assert summarize_depth(depths,{root:1})['expected_moves'] == pytest.approx(3)
    assert chapter_prepared_depth(depths,[position('Nf3'),position('g3'),leaf],chapter)['expected_moves'] == pytest.approx(2)
    with (tmp_path/'fixture.pgn').open('a') as stream:
        stream.write('\n\n1. Nf3 d5 2. g3 Nf6 3. Bg2 *')
    duplicate = parse(tmp_path/'fixture.pgn')
    dm,do,dr,*_ = setup(duplicate,True,evidence,policy)
    assert summarize_depth(prepared_depth_values(dm,do,dr),{root:1})['expected_moves'] == pytest.approx(3)


def test_conflict_cycle_color_and_reproducibility(tmp_path):
    g = graph(tmp_path,'1. e4 (1. d4) *')
    assert len(conflicts(g,True)) == 1
    assert list(resolve(g,True,{})[g.roots[0]]) == ['e2e4']
    with pytest.raises(ValueError,match='Invalid policy'):
        resolve(g,True,{g.roots[0]:{'e2e4':1,'d2d4':1}})
    g = graph(tmp_path,'1. Nf3 Nf6 2. Ng1 Ng8 *')
    with pytest.raises(ValueError,match='cycle'):
        topology(resolve(g,True,{}),g.roots)
    g = graph(tmp_path,'*')
    e = {g.roots[0]:data(7,2,1)}
    white,black = setup(g,True,e),setup(g,False,e)
    assert white[4][g.roots[0]][KNOWN,0] == pytest.approx(.8)
    assert black[4][g.roots[0]][KNOWN,0] == pytest.approx(.2)
    assert np.array_equal(white[5][g.roots[0]],setup(g,True,e)[5][g.roots[0]])


def test_castling_and_legal_ep_identity():
    k = position('e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5')
    d = data(1,0,0,[('e1h1',1,0,0)])
    validate(d,k)
    assert d['moves'][0]['uci'] == 'e1g1'
    b = chess.Board(); b.push_san('e4')
    assert key(b).endswith(' -')
    assert key(b) == key(chess.Board(b.fen().rsplit(' ',2)[0]+' 50 99'))


def test_terminal_outcome_no_evidence(tmp_path):
    for fen, expected in [('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1',1),('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',.5),('7k/8/6K1/8/8/8/8/8 w - - 0 1',.5)]:
        g = graph(tmp_path,f'[SetUp "1"]\n[FEN "{fen}"]\n\n*')
        m,o,r,s,v,p = setup(g,True,{})
        assert v[g.roots[0]][KNOWN,0] == expected


def test_offline_cache_miss_is_error(tmp_path):
    explorer = Explorer(tmp_path,offline=True)
    with pytest.raises(ValueError,match='Offline cache miss'):
        explorer.get(key(chess.Board()))
    explorer.close()


def test_reject_variant_and_malformed_pgn(tmp_path):
    with pytest.raises(ValueError):
        graph(tmp_path,'[Variant "Atomic"]\n\n1. e4 *')
    with pytest.raises(ValueError):
        graph(tmp_path,'1. e4 e5 2. Bh6 *')


def test_leaf_results_are_not_parent_move_results(tmp_path):
    g = graph(tmp_path,'1. e4 e5 *')
    evidence = {position('e4'):data(100,0,0,[('e7e5',100,0,0)]),position('e4 e5'):data(0,0,100)}
    assert setup(g,True,evidence)[4][g.roots[0]][KNOWN,0] == 0


def test_entry_ambiguity_and_explicit_exclusions(tmp_path):
    text = '[ChapterURL "https://lichess.org/study/test/a"]\n\n1. e4 e5 (1... c5) *\n\n[ChapterURL "https://lichess.org/study/test/b"]\n\n1. e4 e6 *'
    g = graph(tmp_path,text)
    entry = infer_entries(g)['a']
    assert entry['status'] == 'inferred_multiple_frontiers'
    assert entry['positions'] == entry['candidates']
    assert len(entry['candidates']) == 2
    assert len(parse(tmp_path/'fixture.pgn',exclusions=['a']).chapters) == 1
    removed = parse(tmp_path/'fixture.pgn',exclusions=[f'a:{position("e4")}:c7c5'])
    assert position('e4 c5') not in removed.nodes


def test_move_order_preserves_mainline_before_variations(tmp_path):
    g = graph(tmp_path,'1. e4 (1. d4) e5 *')
    assert list(g.nodes[g.roots[0]].edges) == ['e2e4','d2d4']
    assert list(conflicts(g,True)[0]['choices']) == ['e2e4','d2d4']


def test_unlisted_opponent_move_reenters_known_theory(tmp_path):
    # First chapter ends its own variation at the transposition. The other
    # chapter supplies the continuation; opponent 2...d5 is not locally listed.
    g = graph(tmp_path,'1. Nf3 Nf6 2. g3 e6 *\n\n1. g3 d5 2. Nf3 Nf6 3. Bg2 *')
    t = resolve(g,True,{g.roots[0]:{'g1f3':.5,'g2g3':.5}})
    assert t[position('Nf3 Nf6 g3')]['d7d5'][0] == position('g3 d5 Nf3 Nf6')


def test_cli_fixture_end_to_end(tmp_path,monkeypatch):
    from repertoire_score import score as cli
    from repertoire_score.report.format import headline_delta
    white_row = f'| White | 80.00% | 70.00% | {headline_delta(.7, .8)} |'
    black_row = f'| Black | 20.00% | 30.00% | {headline_delta(.3, .2)} |'
    graph(tmp_path, '1. e4 e5 *')  # writes the fixture PGN
    evidence = {position(''):data(7,2,1,[('e2e4',7,2,1)]), position('e4'):data(80,0,20,[('e7e5',80,0,20)]),position('e4 e5'):data(6,2,2)}
    class FakeExplorer:
        def __init__(self,*args):
            self.filters = {}; self.provenance = {}
        def get(self,k):
            self.provenance[k] = {'retrieved_at':'synthetic','cache_key':'synthetic'}
            return evidence[k]
        def close(self):
            pass
    monkeypatch.setattr(cli,'Explorer',FakeExplorer)
    args = SimpleNamespace(config=None,pgn=str(tmp_path/'fixture.pgn'),color='white',output=str(tmp_path/'report'),command='run',cache=str(tmp_path/'cache'),offline=True,refresh=False,simulations=100,seed=1,prior=[.5]*3,sparse_threshold=30,tolerance=1)
    cli.analyze(args)
    result = json.loads((tmp_path/'report.json').read_text())
    assert result['overall']['raw_empirical_score'] == pytest.approx(.7)
    assert result['overall']['prepared_depth']['expected_moves'] == 1
    reference = result['starting_position_reference']
    assert reference['white_score'] == pytest.approx(.8)
    assert reference['black_score'] == pytest.approx(.2)
    assert reference['owner_score'] == pytest.approx(.8)
    assert reference['sample_count'] == 10
    assert result['chapters'][0]['entry_baseline']['raw_score'] == pytest.approx(.8)
    assert result['chapters'][0]['entry_baseline']['difference_pp'] == pytest.approx(-10)
    assert result['diagnostics']['sanity_checks_passed']
    assert len(result['chapters']) == 1
    assert len(result['prior_sensitivity']) == 2
    assert 'Approximate model-based 95% score interval' in (tmp_path/'report.md').read_text()
    assert '| White | 80.00% |' in (tmp_path/'report.md').read_text()
    assert '| Black |' not in (tmp_path/'report.md').read_text()
    assert white_row in (tmp_path/'report.md').read_text()
    assert '| Repertoire score |' in (tmp_path/'report.md').read_text()
    assert '| Posterior mean |' not in (tmp_path/'report.md').read_text()
    assert not (tmp_path/'report.study-description.md').exists()
    assert (tmp_path/'summary.md').exists()
    saved_white = (tmp_path/'report.json').read_bytes()
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'chapter_regions':{'1':{'anchors':[{'path':['e4']}], 'description':'Subject begins at e4'}}}))
    args.config = str(config)
    args.color = 'black'
    args.output = str(tmp_path/'black')
    cli.analyze(args)
    black = json.loads((tmp_path/'black.json').read_text())
    assert set(black['chapters'][0]['region']['positions']) == {position('e4'), position('e4 e5')}
    assert black['chapters'][0]['entries'][0]['position'] == position('e4')
    assert black['chapters'][0]['entries'][0]['conditional_first_entry_weight'] == 1
    assert black['overall']['prepared_depth']['expected_moves'] == 1
    assert black['chapters'][0]['score']['prepared_depth']['expected_moves'] == 1
    assert black_row in (tmp_path/'report.md').read_text(encoding='utf-8')
    assert not (tmp_path/'black.md').exists()
    # A one-color rerun must retain its companion and regenerate their summary.
    assert (tmp_path/'report.json').read_bytes() == saved_white
    registry = json.loads((tmp_path/'.report-index.json').read_text())
    assert registry == {'white': 'report.json', 'black': 'black.json'}
    summary = (tmp_path/'summary.md').read_text()
    assert summary.count('<summary>Chapter comparisons (1 chapter)</summary>') == 2
    assert 'Posterior mean' not in summary and 'Empirical score' not in summary
    assert '| White | 80.0% | 70.0% | -10.0%' in summary
    assert '| Black | 20.0% | 30.0% | +10.0%' in summary
    assert 'Posterior' not in summary and str(tmp_path) not in summary
    assert not (tmp_path/'black.study-description.md').exists()


def test_missing_distribution_does_not_imply_zero_entry_probability(tmp_path):
    g = graph(tmp_path,'1. e4 e5 (1... c5) *')
    evidence = {position('e4'):data(0,0,0),position('e4 e5'):data(5,0,5),position('e4 c5'):data(5,0,5)}
    m,o,r,s,v,p = setup(g,True,evidence)
    one = chapter_score(m,o,r,v,{g.roots[0]:1},[position('e4 e5')],s)
    assert one['entry_probability'] is None
    assert one['entry_probability_bounds'] == [0,1]
    assert one['raw_empirical_score'] == .5
    two = chapter_score(m,o,r,v,{g.roots[0]:1},[position('e4 e5'),position('e4 c5')],s)
    assert two['status'] == 'unresolved_first_entry_weights'


def test_prior_strength_is_independent_of_legal_move_count(tmp_path):
    # At a one-game node, total joint prior strength is 1.5, regardless
    # of the twenty legal black replies. Mean observed-row mass is 1.075/2.5.
    g = graph(tmp_path,'1. e4 e5 *')
    m,o,r,s,v,p = setup(g,True,{position('e4'):data(1,0,0,[('e7e5',1,0,0)]),position('e4 e5'):data(1,0,0)})
    k = position('e4')
    j = next(j for j,b in enumerate(m[k].branches) if b.move=='e7e5')
    assert s.sample[k][j][0] == pytest.approx(1.075/2.5)
