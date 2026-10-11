"""Bounded, immutable board calculations shared by analysis stages.

Only position-dependent facts live here. Graph membership, move policies and
Explorer evidence are applied by callers, so a transposition in a different
repertoire cannot inherit another graph's continuations.
"""

from collections.abc import Iterable
from functools import cache, lru_cache
from typing import NamedTuple

import chess

from .schema import Position


def canonical(board: chess.Board) -> Position:
    return ' '.join(board.fen(en_passant='legal').split()[:4])


def position_of(node: str) -> Position:
    """The position of a scoring node. Inside a repetition loop, a node also records how often each loop
    position has occurred (see graph.unroll); everywhere else the node is the position itself."""
    return node.partition('#')[0]


@lru_cache(maxsize=65536)
def terminal_white(position: Position) -> float | None:
    board = chess.Board(position + ' 0 1')
    if board.is_checkmate():
        return float(board.turn != chess.WHITE)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    return None


class Geometry(NamedTuple):
    turn: bool
    outcome: float | None
    moves: tuple[tuple[str, str], ...]


@lru_cache(maxsize=16384)
def geometry(position: Position) -> Geometry:
    board = chess.Board(position + ' 0 1')
    moves = []
    for move in board.legal_moves:
        board.push(move)
        moves.append((move.uci(), canonical(board)))
        board.pop()
    return Geometry(board.turn, terminal_white(position), tuple(moves))


def owner_outcome(position: Position, color: bool) -> float | None:
    value = terminal_white(position)
    return value if value is None or color == chess.WHITE else 1 - value


# Cheap position facts. Building a chess.Board costs far more than these string and
# cache lookups, and the analyses ask for the same few thousand positions repeatedly.

STARTING_POSITION = canonical(chess.Board())


def turn(position: Position) -> bool:
    """Side to move in a canonical position (or a full FEN): True for White."""
    return position.split(' ', 2)[1] == 'w'


def fen_position(fen: str) -> Position:
    """The canonical four-field position of a full FEN with legal en passant."""
    return ' '.join(fen.split()[:4])


def fen_number(fen: str) -> int:
    return int(fen.split()[5])


@cache
def children(position: Position) -> dict[str, Position]:
    """Legal UCI moves mapped to the canonical position each one reaches."""
    return dict(geometry(position).moves)


@cache
def san(position: Position, uci: str) -> str:
    return chess.Board(position + ' 0 1').san(chess.Move.from_uci(uci))


def move_text(position: Position, number: int, uci: str) -> str:
    """A move with its number, as python-chess prints it from a board at that full-move number."""
    return f'{number}{"." if turn(position) else "..."}{san(position, uci)}'


def next_number(position: Position, number: int) -> int:
    """The full-move number after a move from `position`; it increases after Black moves."""
    return number if turn(position) else number + 1


def route_line(position: Position, number: int, moves: Iterable[str]) -> tuple[str, Position, int]:
    """Numbered move text along `moves` from `position` at full-move `number`, with the position and number reached."""
    text = []
    for move in moves:
        text.append(move_text(position, number, move))
        position, number = children(position)[move], next_number(position, number)
    return ' '.join(text), position, number


def after_fen(fen: str, uci: str) -> str:
    """The full FEN after a move, matching python-chess's fen(en_passant='legal') without building a board."""
    fields = fen.split()
    position = ' '.join(fields[:4])
    notation = san(position, uci)
    # A pawn move or capture resets the halfmove clock; pawn moves are the SAN moves that start with a file.
    halfmove = 0 if notation[0] in 'abcdefgh' or 'x' in notation else int(fields[4]) + 1
    return f'{children(position)[uci]} {halfmove} {next_number(position, int(fields[5]))}'
