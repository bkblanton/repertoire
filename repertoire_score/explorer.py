"""Sequential, validated, authenticated Explorer access with persistent cache."""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

import chess
import httpx

from .board_cache import children

ENDPOINT = "https://explorer.lichess.org/lichess"
DEFAULT_FILTERS = {
    "variant": "standard",
    "speeds": "blitz,rapid,classical",
    "ratings": "0,1000,1200,1400,1600,1800,2000,2200,2500",
    "since": "1952-01",
    "until": "3000-12",
}

_cache_observer = ContextVar('explorer_cache_observer', default=None)


@contextmanager
def observe_cache():
    """Track actual table dependencies, including misses, without storing data."""
    paths = set()
    token = _cache_observer.set(paths)
    try:
        yield paths
    finally:
        _cache_observer.reset(token)


class CacheMiss(ValueError):
    """An offline read found no cached table; distinct from a table with zero games."""


def add_token_option(parser):
    parser.add_argument(
        '--token-file', help='File containing a Lichess API token; overrides LICHESS_TOKEN for this run'
    )


def apply_token_file(parser, args):
    """Load --token-file into this process's environment; the token is never written anywhere."""
    if not args.token_file:
        return
    try:
        token = Path(args.token_file).read_text(encoding='utf-8-sig').strip()
    except OSError as exc:
        parser.error(f'cannot read token file: {exc}')
    if not token:
        parser.error(f'token file is empty: {args.token_file}')
    os.environ['LICHESS_TOKEN'] = token


def counts(row):
    result = [row.get(k) for k in ("white", "draws", "black")]
    if any(type(v) is not int or v < 0 for v in result):
        raise ValueError("Missing, negative, or noninteger Explorer result counts")
    return result


def validate(data, position):
    parent = counts(data)
    legal = children(position)
    if not isinstance(data.get("moves"), list):
        raise ValueError("Explorer response missing moves")
    seen, sums = set(), [0, 0, 0]
    for row in data["moves"]:
        move = row.get("uci")
        if not isinstance(move, str) or move not in legal:
            # Explorer can encode castling as king-to-rook, even for standard chess.
            try:
                move = chess.Board(position + " 0 1").parse_uci(move).uci()
            except (ValueError, TypeError):
                raise ValueError(f"Illegal Explorer move: {move}") from None
            row["uci"] = move
        if move not in legal or move in seen:
            raise ValueError(f"Illegal or duplicate Explorer move: {move}")
        seen.add(move)
        for i, v in enumerate(counts(row)):
            sums[i] += v
    residual = [p - s for p, s in zip(parent, sums)]
    if min(residual) < 0:
        raise ValueError(f"Inconsistent Explorer totals at {position}: {parent} < {sums}")
    return residual


class Explorer:
    def __init__(self, cache, filters=None, offline=False, refresh=False, delay=1.0, retries=6):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.filters = dict(DEFAULT_FILTERS if filters is None else filters)
        self.offline, self.refresh, self.delay, self.retries = offline, refresh, delay, retries
        token = os.environ.get("LICHESS_TOKEN", "").strip()
        if not offline and not token:
            raise ValueError("Set LICHESS_TOKEN in the environment")
        self.client = httpx.Client(headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=45)
        self.provenance = {}
        self.last_request = 0

    def get(self, position):
        query = dict(self.filters, fen=position + " 0 1", moves=len(children(position)), topGames=0, recentGames=0)
        identity = {"endpoint": ENDPOINT, "query": query}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / (digest + ".json")
        observer = _cache_observer.get()
        if observer is not None:
            observer.add(path.resolve())
        if path.exists() and not self.refresh:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry["identity"] != identity:
                raise ValueError("Cache identity mismatch")
            validate(entry["data"], position)
            self.provenance[position] = {"cache_key": digest, "retrieved_at": entry["retrieved_at"]}
            return entry["data"]
        if self.offline:
            raise CacheMiss(f"Offline cache miss: {position}")
        for attempt in range(self.retries + 1):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.client.get(ENDPOINT, params=query)
            except httpx.TransportError as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"Explorer transport failure ({type(exc).__name__}) at {position}") from None
                time.sleep(min(60, 2**attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.retries:
                    raise RuntimeError(f"Explorer HTTP {response.status_code} after bounded retries")
                retry = response.headers.get("Retry-After", "")
                wait = max(60 if response.status_code == 429 else 2**attempt, float(retry) if retry.isdigit() else 0)
                print(f"Explorer HTTP {response.status_code}; backing off {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"Explorer HTTP {response.status_code}; response is not statistical evidence")
            data = response.json()
            validate(data, position)
            entry = {"identity": identity, "retrieved_at": datetime.now(UTC).isoformat(), "data": data}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(entry), encoding="utf-8")
            temporary.replace(path)
            self.provenance[position] = {"cache_key": digest, "retrieved_at": entry["retrieved_at"]}
            return data
        raise AssertionError("Retry loop exhausted")

    def close(self):
        self.client.close()
