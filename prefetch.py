"""Warm the cache without selecting any repertoire policy."""
import argparse
import chess
from repertoire_score.graph import parse
from repertoire_score.explorer import Explorer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn")
    parser.add_argument("color", choices=["white", "black"])
    args = parser.parse_args()
    graph = parse(args.pgn)
    explorer = Explorer(".cache/explorer")
    positions = [k for k, n in graph.nodes.items() if
                 (not n.edges or chess.Board(n.fen).turn != (args.color == "white")) and
                 not chess.Board(n.fen).is_checkmate() and not chess.Board(n.fen).is_stalemate() and
                 not chess.Board(n.fen).is_insufficient_material()]
    try:
        for i, k in enumerate(positions):
            data = explorer.get(k)
            if i % 10 == 0 or i+1 == len(positions):
                print(f"{args.color}: cached {i+1}/{len(positions)}; sample={data['white']+data['draws']+data['black']}", flush=True)
    finally:
        explorer.close()
