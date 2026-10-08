"""Sequential, validated, authenticated Explorer access with persistent cache."""

import hashlib
import json
import os
import time
from argparse import ArgumentParser, Namespace
from collections import deque
from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import chess
import httpx

from .board_cache import children
from .schema import JsonObject, Position

ENDPOINT = "https://explorer.lichess.org/lichess"
DEFAULT_FILTERS = {
    "variant": "standard",
    "speeds": "blitz,rapid,classical",
    "ratings": "0,1000,1200,1400,1600,1800,2000,2200,2500",
    "since": "1952-01",
    "until": "3000-12",
}

_cache_observer: ContextVar[set[Path] | None] = ContextVar('explorer_cache_observer', default=None)


@contextmanager
def observe_cache() -> Iterator[set[Path]]:
    """Track actual table dependencies, including misses, without storing data."""
    paths: set[Path] = set()
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

    def __init__(self, label: str, patience: int = OUTAGE_PATIENCE, note: str = '') -> None:
        self.label, self.patience = label, patience
        self.note = f' ({note})' if note else ''
        self.attempt, self.waited = 0, 0

    def rate_limited(self, retry_after: str = '') -> None:
        wait = max(RATE_LIMIT_WAIT, int(retry_after) if retry_after.isdigit() else 0)
        print(f'{self.label}: rate-limited (HTTP 429); waiting {wait}s{self.note}', flush=True)
        time.sleep(wait)

    def unavailable(self, reason: str, retry_after: str = '') -> None:
        if self.waited >= self.patience:
            raise RuntimeError(f'{self.label} still unavailable after {self.waited // 60} minutes of retries: {reason}')
        wait = max(min(60, 2**self.attempt), int(retry_after) if retry_after.isdigit() else 0)
        wait = min(wait, max(1, self.patience - self.waited))
        self.attempt += 1
        self.waited += wait
        print(f'{self.label}: {reason}; retrying in {wait}s{self.note}', flush=True)
        time.sleep(wait)


def duration(seconds: float) -> str:
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

    def __init__(self, total: int, label: str) -> None:
        self.total, self.label, self.done = total, label, 0
        self.finished = deque([time.monotonic()], maxlen=self.WINDOW + 1)
        self.printed = self.finished[0]

    def status(self) -> str:
        return f'{self.done}/{self.total} fetched'

    def remaining(self) -> float | None:
        elapsed = self.finished[-1] - self.finished[0]
        if self.done < 2 or elapsed <= 0:
            return None
        return (self.total - self.done) * elapsed / (len(self.finished) - 1)

    def fetch(self, explorer: 'Explorer', position: Position) -> JsonObject:
        """One network fetch. If it is interrupted or fails, say how far the run got before stopping."""
        explorer.note = self.status()
        try:
            data = explorer.get(position)
        except BaseException:
            print(f'{self.label}: stopped at {self.status()}; tables fetched so far stay cached', flush=True)
            raise
        finally:
            explorer.note = ''
        self.advance()
        return data

    def advance(self) -> None:
        self.done += 1
        now = time.monotonic()
        self.finished.append(now)
        if self.done == self.total or now - self.printed >= self.INTERVAL:
            self.printed = now
            left = self.remaining()
            estimate = f', ~{duration(left)} left' if left and self.done < self.total else ''
            print(f'{self.label}: {self.status()}{estimate}', flush=True)


def survey(
    explorer: 'Explorer', positions: Collection[Position], label: str, fetching: bool | None = None
) -> list[Position]:
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


def fetch_missing(explorer: 'Explorer', positions: Iterable[Position], label: str) -> None:
    """Request only the tables that are not cached, with progress; cached tables are not read."""
    missing = survey(explorer, list(dict.fromkeys(positions)), label)
    progress = Progress(len(missing), label)
    for k in missing:
        progress.fetch(explorer, k)


def collect(explorer: 'Explorer', positions: Iterable[Position], label: str) -> dict[Position, JsonObject]:
    """Read every table in order, saying up front how many need the network and how long that may take."""
    positions = list(dict.fromkeys(positions))
    missing = survey(explorer, positions, label)
    progress, pending = Progress(len(missing), label), set(missing)
    evidence: dict[Position, JsonObject] = {}
    for k in positions:
        fetching = k in pending and not explorer.offline
        evidence[k] = progress.fetch(explorer, k) if fetching else explorer.get(k)
    return evidence


def add_token_option(parser: ArgumentParser) -> None:
    parser.add_argument(
        '--token-file', help='File containing a Lichess API token; overrides LICHESS_TOKEN for this run'
    )


def apply_token_file(parser: ArgumentParser, args: Namespace) -> None:
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


def counts(row: Mapping[str, Any]) -> list[int]:
    result = [row.get(k) for k in ("white", "draws", "black")]
    if any(type(v) is not int or v < 0 for v in result):
        raise ValueError("Missing, negative, or noninteger Explorer result counts")
    return cast(list[int], result)


def validate(data: JsonObject, position: Position) -> list[int]:
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
    def __init__(
        self,
        cache: str | Path,
        filters: Mapping[str, str] | None = None,
        offline: bool = False,
        refresh: bool = False,
        delay: float = 1.0,
        patience: int = OUTAGE_PATIENCE,
    ) -> None:
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.filters = dict(DEFAULT_FILTERS if filters is None else filters)
        self.offline, self.refresh, self.delay, self.patience = offline, refresh, delay, patience
        token = os.environ.get("LICHESS_TOKEN", "").strip()
        if not offline and not token:
            raise ValueError("Set LICHESS_TOKEN in the environment")
        if not offline:
            # A run stopped mid-write leaves a temporary file; the table itself was never cached.
            for leftover in self.cache.glob("*.tmp"):
                leftover.unlink(missing_ok=True)
        self.client = httpx.Client(headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=45)
        self.provenance: dict[Position, dict[str, str]] = {}
        self.last_request: float = 0
        # Shown with backoff messages, so a long rate-limit wait still says how far the run has got.
        self.note = ''

    def locate(self, position: Position) -> tuple[dict[str, Any], dict[str, Any], str, Path]:
        query = dict(self.filters, fen=position + " 0 1", moves=len(children(position)), topGames=0, recentGames=0)
        identity = {"endpoint": ENDPOINT, "query": query}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return query, identity, digest, self.cache / (digest + ".json")

    def cached(self, position: Position) -> bool:
        """Whether `get` would be answered without a request."""
        return not self.refresh and self.locate(position)[3].exists()

    def get(self, position: Position) -> JsonObject:
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

    def close(self) -> None:
        self.client.close()
