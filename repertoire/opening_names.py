"""Opening names from the lichess-org/chess-openings dataset, which the Explorer also uses."""

import csv
from functools import cache
from pathlib import Path

import chess

from .board_cache import canonical

DIRECTORY = Path(__file__).parent / 'data' / 'chess-openings'
SOURCE = dict(
    repository='https://github.com/lichess-org/chess-openings',
    commit='a6189a30dc273ccb21fc2536a9a2fefd5592a67a',
    license='CC0-1.0',
)


@cache
def names():
    """The ECO code and name of every listed position, keyed by canonical position."""
    result = {}
    for path in sorted(DIRECTORY.glob('*.tsv')):
        with path.open(encoding='utf-8', newline='') as file:
            for row in csv.DictReader(file, delimiter='\t'):
                board = chess.Board()
                for token in row['pgn'].split():
                    if not token.endswith('.'):
                        board.push_san(token)
                k = canonical(board)
                if k in result:
                    raise ValueError(f'Duplicate opening name position: {row["pgn"]}')
                result[k] = dict(eco=row['eco'], name=row['name'])
    if not result:
        raise FileNotFoundError(f'No opening names in {DIRECTORY}')
    return result
