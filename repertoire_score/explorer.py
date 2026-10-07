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


# Lichess asks clients to wait a full minute after HTTP 429.
RATE_LIMIT_WAIT = 60
# How long one request keeps retrying through server errors and dropped connections, long enough
# to ride out a Wi-Fi reconnect or a short Lichess outage during an unattended run.
OUTAGE_PATIENCE = 30 * 60


class Backoff:
    """Waits between attempts at one request. Rate limits are temporary by definition, so they are
    retried indefinitely; outages are retried with exponential waits until `patience` seconds."""

    def __init__(self, label, patience=OUTAGE_PATIENCE):
        self.label, self.patience = label, patience
        self.attempt, self.waited = 0, 0

    def rate_limited(self, retry_after=''):
        wait = max(RATE_LIMIT_WAIT, int(retry_after) if retry_after.isdigit() else 0)
        print(f'{self.label}: rate-limited (HTTP 429); waiting {wait}s', flush=True)
        time.sleep(wait)

    def unavailable(self, reason, retry_after=''):
        if self.waited >= self.patience:
            raise RuntimeError(f'{self.label} still unavailable after {self.waited // 60} minutes of retries: {reason}')
        wait = max(min(60, 2**self.attempt), int(retry_after) if retry_after.isdigit() else 0)
        wait = min(wait, max(1, self.patience - self.waited))
        self.attempt += 1
        self.waited += wait
        print(f'{self.label}: {reason}; retrying in {wait}s', flush=True)
        time.sleep(wait)


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
    def __init__(self, cache, filters=None, offline=False, refresh=False, delay=1.0, patience=OUTAGE_PATIENCE):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.filters = dict(DEFAULT_FILTERS if filters is None else filters)
        self.offline, self.refresh, self.delay, self.patience = offline, refresh, delay, patience
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
        backoff = Backoff("Explorer", self.patience)
        while True:
            time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.client.get(ENDPOINT, params=query)
            except httpx.TransportError as exc:
                backoff.unavailable(f"connection failed ({type(exc).__name__})")
                continue
            retry_after = response.headers.get("Retry-After", "")
            if response.status_code == 429:
                backoff.rate_limited(retry_after)
                continue
            if response.status_code >= 500:
                backoff.unavailable(f"HTTP {response.status_code}", retry_after)
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

    def close(self):
        self.client.close()
