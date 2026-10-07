"""Sequential, validated, authenticated Explorer access with persistent cache."""

import hashlib
import json
import os
import time
from collections import deque
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

    def __init__(self, label, patience=OUTAGE_PATIENCE, note=''):
        self.label, self.patience = label, patience
        self.note = f' ({note})' if note else ''
        self.attempt, self.waited = 0, 0

    def rate_limited(self, retry_after=''):
        wait = max(RATE_LIMIT_WAIT, int(retry_after) if retry_after.isdigit() else 0)
        print(f'{self.label}: rate-limited (HTTP 429); waiting {wait}s{self.note}', flush=True)
        time.sleep(wait)

    def unavailable(self, reason, retry_after=''):
        if self.waited >= self.patience:
            raise RuntimeError(f'{self.label} still unavailable after {self.waited // 60} minutes of retries: {reason}')
        wait = max(min(60, 2**self.attempt), int(retry_after) if retry_after.isdigit() else 0)
        wait = min(wait, max(1, self.patience - self.waited))
        self.attempt += 1
        self.waited += wait
        print(f'{self.label}: {reason}; retrying in {wait}s{self.note}', flush=True)
        time.sleep(wait)


def duration(seconds):
    """A rough human duration: 45s, 12m, 3h 05m."""
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f'{seconds}s'
    hours, minutes = divmod(round(seconds / 60), 60)
    return f'{hours}h {minutes:02d}m' if hours else f'{minutes}m'


class Progress:
    """Counts network fetches and estimates the time left from the recent rate, backoff included."""

    WINDOW = 50
    INTERVAL = 15

    def __init__(self, total, label):
        self.total, self.label, self.done = total, label, 0
        self.finished = deque([time.monotonic()], maxlen=self.WINDOW + 1)
        self.printed = self.finished[0]

    def status(self):
        return f'{self.done}/{self.total} fetched'

    def remaining(self):
        elapsed = self.finished[-1] - self.finished[0]
        if self.done < 2 or elapsed <= 0:
            return None
        return (self.total - self.done) * elapsed / (len(self.finished) - 1)

    def advance(self):
        self.done += 1
        now = time.monotonic()
        self.finished.append(now)
        if self.done == self.total or now - self.printed >= self.INTERVAL:
            self.printed = now
            left = self.remaining()
            estimate = f', ~{duration(left)} left' if left and self.done < self.total else ''
            print(f'{self.label}: {self.status()}{estimate}', flush=True)


def survey(explorer, positions, label, fetching=None):
    """Say how many of `positions` are cached and, when they will be fetched, how long that may take.

    Returns the tables that are not cached. `fetching` defaults to whether the Explorer is online.
    """
    missing = [k for k in positions if not explorer.cached(k)]
    if fetching is None:
        fetching = not explorer.offline
    if not missing:
        print(f'{label}: {len(positions)} Explorer tables, all cached', flush=True)
    elif not fetching:
        print(f'{label}: {len(missing)} of {len(positions)} Explorer tables are not cached', flush=True)
    else:
        # The request delay is a floor; rate limits make real runs slower.
        floor = duration(len(missing) * max(explorer.delay, 0))
        print(
            f'{label}: {len(positions)} Explorer tables: {len(positions) - len(missing)} cached, '
            f'{len(missing)} to fetch (at least {floor})',
            flush=True,
        )
    return missing


def fetch_missing(explorer, positions, label):
    """Request only the tables that are not cached, with progress; cached tables are not read."""
    missing = survey(explorer, list(dict.fromkeys(positions)), label)
    progress = Progress(len(missing), label)
    for k in missing:
        explorer.note = progress.status()
        explorer.get(k)
        progress.advance()
    explorer.note = ''


def collect(explorer, positions, label):
    """Read every table in order, saying up front how many need the network and how long that may take."""
    positions = list(dict.fromkeys(positions))
    missing = survey(explorer, positions, label)
    progress, pending = Progress(len(missing), label), set(missing)
    evidence = {}
    for k in positions:
        if k in pending and not explorer.offline:
            explorer.note = progress.status()
            evidence[k] = explorer.get(k)
            progress.advance()
        else:
            evidence[k] = explorer.get(k)
    explorer.note = ''
    return evidence


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
        # Shown with backoff messages, so a long rate-limit wait still says how far the run has got.
        self.note = ''

    def locate(self, position):
        query = dict(self.filters, fen=position + " 0 1", moves=len(children(position)), topGames=0, recentGames=0)
        identity = {"endpoint": ENDPOINT, "query": query}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return query, identity, digest, self.cache / (digest + ".json")

    def cached(self, position):
        """Whether `get` would be answered without a request."""
        return not self.refresh and self.locate(position)[3].exists()

    def get(self, position):
        query, identity, digest, path = self.locate(position)
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
        backoff = Backoff("Explorer", self.patience, self.note)
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
