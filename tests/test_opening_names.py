import chess

from repertoire_score.board_cache import canonical
from repertoire_score.opening_names import names
from repertoire_score.openings import opening_identity


def key(line):
    board = chess.Board()
    for move in line.split():
        board.push_san(move)
    return canonical(board)


def test_dataset_names_canonical_positions():
    listed = names()
    assert listed[key('e4 e5 Nc3')] == dict(eco='C25', name='Vienna Game')
    assert listed[key('e4 e6')] == dict(eco='C00', name='French Defense')
    assert key('e4 e6 d4 d5 Nd2 Nf6 e5') not in listed
    assert all(opening_identity(value) for value in listed.values())
