"""Bounded, immutable board calculations shared by analysis stages.

Only position-dependent facts live here. Graph membership, move policies and
Explorer evidence are applied by callers, so a transposition in a different
repertoire cannot inherit another graph's continuations.
"""
from functools import lru_cache
from typing import NamedTuple

import chess


def canonical(board):
    return ' '.join(board.fen(en_passant='legal').split()[:4])


@lru_cache(maxsize=65536)
def terminal_white(position):
    board = chess.Board(position + ' 0 1')
    if board.is_checkmate():
        return float(board.turn != chess.WHITE)
    if board.is_stalemate() or board.is_insufficient_material():
        return .5
    return None


class Geometry(NamedTuple):
    turn: bool
    outcome: float | None
    moves: tuple[tuple[str, str], ...]


@lru_cache(maxsize=16384)
def geometry(position):
    board = chess.Board(position + ' 0 1')
    moves = []
    for move in board.legal_moves:
        board.push(move)
        moves.append((move.uci(), canonical(board)))
        board.pop()
    return Geometry(board.turn, terminal_white(position), tuple(moves))


def owner_outcome(position, color):
    value = terminal_white(position)
    return value if value is None or color == chess.WHITE else 1 - value
