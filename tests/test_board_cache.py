import random

import chess

from repertoire_score.board_cache import (
    STARTING_POSITION,
    after_fen,
    canonical,
    children,
    fen_number,
    fen_position,
    move_text,
    next_number,
    san,
    turn,
)


def test_cached_move_facts_match_python_chess_over_random_games():
    rng = random.Random(7)
    checked = 0
    for _ in range(8):
        board = chess.Board()
        for _ in range(100):
            moves = list(board.legal_moves)
            if not moves:
                break
            fen = board.fen(en_passant='legal')
            position = fen_position(fen)
            assert position == canonical(board)
            assert turn(position) == board.turn and fen_number(fen) == board.fullmove_number
            move = rng.choice(moves)
            expected_text = f'{board.fullmove_number}{"." if board.turn else "..."}{board.san(move)}'
            assert san(position, move.uci()) == board.san(move)
            assert move_text(position, fen_number(fen), move.uci()) == expected_text
            assert next_number(position, fen_number(fen)) == (board.fullmove_number + (not board.turn))
            board.push(move)
            assert children(position)[move.uci()] == canonical(board)
            assert after_fen(fen, move.uci()) == board.fen(en_passant='legal')
            checked += 1
    assert checked > 500
    assert STARTING_POSITION == canonical(chess.Board())


def test_halfmove_clock_and_numbers_for_castling_en_passant_and_promotion():
    cases = [
        ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 3 10', 'e1g1'),
        ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 3 10', 'e8c8'),
        ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 7', 'e5d6'),
        ('1n2k3/P7/8/8/8/8/8/4K3 w - - 5 40', 'a7b8q'),
        ('4k3/8/8/8/8/8/8/R3K3 b - - 12 30', 'e8d7'),
    ]
    for fen, uci in cases:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        position = fen_position(board.fen(en_passant='legal'))
        assert move_text(position, board.fullmove_number, uci).endswith(board.san(move))
        expected_fen = board.fen(en_passant='legal')
        board.push(move)
        assert after_fen(expected_fen, uci) == board.fen(en_passant='legal')
