"""Single-file hash-bucketed index for fast random-access lookups into the Lichess evaluation export.

The Lichess evaluation export (lichess_db_eval.jsonl.zst) stores hundreds of millions of positions in no useful order,
so searching it for even a few new positions requires decompressing the entire 22 GB file. This module converts the
export once into a single binary archive (`lichess_db_eval.idx`) partitioned into `num_buckets` (default 65,536)
independent Zstandard-compressed buckets keyed by CRC32 of the 3-field FEN prefix (pieces, side to move, castling).

Looking up a handful of new positions after editing a repertoire then reads and decompresses only the buckets those
positions hash into, taking seconds instead of minutes, while keeping a single file on disk.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import time
import zlib
from collections.abc import Collection, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, BinaryIO

import zstandard

from .explorer import Progress, duration
from .schema import JsonObject, Position

MAGIC = b'REPEVAL1'
VERSION = 1
HEADER = struct.Struct('<8sIIQQQ')  # magic, version, num_buckets, table_offset, metadata_offset, metadata_length
ENTRY = struct.Struct('<QQ')  # offset, length
DEFAULT_BUCKETS = 65536
BATCH_FRAMES = 8
MAX_FRAME_DECOMPRESSED = 128 * 1024 * 1024
# Mirrors evals.LINE_KEY: pieces, side to move and castling rights, leaving en passant to python-chess.
LINE_KEY = re.compile(rb'\{"fen":"([^ "]+ [wb] [^ "]+) ')
PZSTD_SKIPPABLE_MAGIC = 0x184D2A50
ZSTD_FRAME_MAGIC = 0xFD2FB528


def bucket_for(prefix: bytes, num_buckets: int) -> int:
    """Deterministic bucket index for a 3-field FEN prefix (`pieces turn castling`)."""
    return (zlib.crc32(prefix) & 0xFFFFFFFF) % num_buckets


def index_path_for(export_path: str | Path) -> Path:
    """The sibling index path for an evaluation export file, or the path itself if already an `.idx`."""
    path = Path(export_path)
    if path.suffix == '.idx':
        return path
    if path.name.endswith('.jsonl.zst'):
        return path.with_name(path.name[: -len('.jsonl.zst')] + '.idx')
    return path.with_suffix('.idx')


def pzstd_frames(path: Path) -> list[tuple[int, int]] | None:
    """Locate independent pzstd frames `(payload_offset, payload_size)` if the entire file uses pzstd framing."""
    size = path.stat().st_size
    if size < 16:
        return None
    frames: list[tuple[int, int]] = []
    with path.open('rb') as raw:
        pos = 0
        while pos < size:
            raw.seek(pos)
            header = raw.read(16)
            if len(header) < 16:
                return None
            magic, payload_len, frame_size = struct.unpack('<III', header[:12])
            if magic != PZSTD_SKIPPABLE_MAGIC or payload_len != 4 or frame_size == 0:
                return None
            if struct.unpack('<I', header[12:16])[0] != ZSTD_FRAME_MAGIC:
                return None
            if pos + 12 + frame_size > size:
                return None
            frames.append((pos + 12, frame_size))
            pos += 12 + frame_size
    return frames if pos == size and frames else None


def _split_dimensions(num_buckets: int) -> tuple[int, int]:
    """Factor `num_buckets` into `(coarse, fine)` for 2-pass external partitioning."""
    if num_buckets < 1:
        raise ValueError(f'num_buckets must be at least 1: {num_buckets}')
    root = math.isqrt(num_buckets)
    for coarse in range(min(root, 256), 0, -1):
        if num_buckets % coarse == 0:
            return coarse, num_buckets // coarse
    return 1, num_buckets


def _route_block(
    data: bytes,
    start: int,
    end: int,
    num_buckets: int,
    fine: int,
    buffers: list[list[bytes]],
) -> int:
    """Partition complete JSONL lines in `data[start:end]` into coarse shard `buffers`; returns line count."""
    if end <= start:
        return 0
    keys = LINE_KEY.findall(data, start, end)
    lines = data[start:end].split(b'\n')
    if lines and not lines[-1]:
        lines.pop()
    crc = zlib.crc32
    if len(lines) == len(keys):
        for key, line in zip(keys, lines):
            buffers[((crc(key) & 0xFFFFFFFF) % num_buckets) // fine].append(line)
        return len(lines)
    count = 0
    for line in lines:
        match = LINE_KEY.match(line)
        if match:
            buffers[((crc(match.group(1)) & 0xFFFFFFFF) % num_buckets) // fine].append(line)
            count += 1
    return count


def _flush_buffers(
    out: BinaryIO,
    cctx: zstandard.ZstdCompressor,
    buffers: list[list[bytes]],
    chunks: list[tuple[int, int, int]],
) -> None:
    """Compress each non-empty coarse buffer as one zstd blob into `out` and record `(coarse_id, offset, length)`."""
    for coarse_id, lines in enumerate(buffers):
        if not lines:
            continue
        compressed = cctx.compress(b'\n'.join(lines) + b'\n')
        offset = out.tell()
        out.write(compressed)
        chunks.append((coarse_id, offset, len(compressed)))
        lines.clear()


def _pass1_frames_task(
    args: tuple[int, str, list[tuple[int, int]], bool, bool, str, int, int],
) -> tuple[int, str, list[tuple[int, int, int]], bytes, bytes, int, int]:
    """Pass 1 worker: decompress a contiguous slice of pzstd frames and partition into coarse zstd chunks."""
    task_id, export_str, frame_slice, has_prev, has_next, part_str, num_buckets, fine = args
    coarse = num_buckets // fine
    dctx = zstandard.ZstdDecompressor()
    cctx = zstandard.ZstdCompressor(level=1)
    buffers: list[list[bytes]] = [[] for _ in range(coarse)]
    chunks: list[tuple[int, int, int]] = []
    head_fragment = b''
    tail = b''
    lines_routed = 0
    bytes_read = 0
    first_frame = True

    with open(export_str, 'rb') as src, open(part_str, 'wb') as out:
        for i, (offset, fsize) in enumerate(frame_slice):
            src.seek(offset)
            compressed = src.read(fsize)
            bytes_read += fsize + 12
            raw = dctx.decompress(compressed, max_output_size=MAX_FRAME_DECOMPRESSED)
            data = tail + raw
            if first_frame:
                if has_prev:
                    nl = data.find(b'\n')
                    if nl == -1:
                        # Entire frame is still part of the leading line; carry forward in `head_fragment`.
                        head_fragment += data
                        tail = b''
                        continue
                    head_fragment += data[:nl]
                    start = nl + 1
                else:
                    start = 0
                first_frame = False
            else:
                start = 0
            end = data.rfind(b'\n') + 1
            if end > start:
                lines_routed += _route_block(data, start, end, num_buckets, fine, buffers)
                tail = data[end:]
            else:
                tail = data[start:]
            if (i + 1) % BATCH_FRAMES == 0:
                _flush_buffers(out, cctx, buffers, chunks)
        if not has_next and tail:
            # Final line of the file with no trailing newline.
            lines_routed += _route_block(tail, 0, len(tail), num_buckets, fine, buffers)
            tail = b''
        _flush_buffers(out, cctx, buffers, chunks)

    return task_id, part_str, chunks, head_fragment, tail, lines_routed, bytes_read


def _pass1_stream(
    export_path: Path,
    part_path: Path,
    num_buckets: int,
    fine: int,
    chunk_size: int = 1 << 25,
) -> tuple[list[tuple[int, int, int]], int]:
    """Single-process streaming Pass 1 for plain (non-pzstd) `.zst` exports."""
    coarse = num_buckets // fine
    cctx = zstandard.ZstdCompressor(level=1)
    buffers: list[list[bytes]] = [[] for _ in range(coarse)]
    chunks: list[tuple[int, int, int]] = []
    lines_routed = 0
    total = export_path.stat().st_size
    started = printed = time.monotonic()

    with export_path.open('rb') as src, part_path.open('wb') as out:
        reader = zstandard.ZstdDecompressor().stream_reader(src, read_size=1 << 22)
        tail = b''
        blocks = 0
        while block := reader.read(chunk_size):
            data = tail + block
            end = data.rfind(b'\n') + 1
            if end > 0:
                lines_routed += _route_block(data, 0, end, num_buckets, fine, buffers)
                tail = data[end:]
            else:
                tail = data
            blocks += 1
            if blocks % 4 == 0:
                _flush_buffers(out, cctx, buffers, chunks)
            if time.monotonic() - printed >= Progress.INTERVAL:
                printed = time.monotonic()
                share = src.tell() / total if total else 1.0
                elapsed = time.monotonic() - started
                left = f', ~{duration(elapsed * (1 - share) / share)} left' if 0 < share < 1 else ''
                print(f'Index pass 1/2: {100 * share:.0f}% read{left}', flush=True)
        if tail:
            lines_routed += _route_block(tail, 0, len(tail), num_buckets, fine, buffers)
        _flush_buffers(out, cctx, buffers, chunks)
    return chunks, lines_routed


def _pass2_shard_task(
    args: tuple[int, list[tuple[str, int, int]], list[bytes], str, int, int],
) -> tuple[int, str, list[int]]:
    """Pass 2 worker: read one coarse shard's chunks, split into `fine` buckets, write zstd-3 blobs to `out_str`."""
    coarse_id, chunk_locs, extra_lines, out_str, num_buckets, fine = args
    dctx = zstandard.ZstdDecompressor()
    cctx = zstandard.ZstdCompressor(level=3)
    sub_buckets: list[list[bytes]] = [[] for _ in range(fine)]
    crc = zlib.crc32

    def route_sub(data: bytes) -> None:
        end = len(data)
        if not end:
            return
        keys = LINE_KEY.findall(data, 0, end)
        lines = data.split(b'\n')
        if lines and not lines[-1]:
            lines.pop()
        if len(lines) == len(keys):
            for key, line in zip(keys, lines):
                sub_buckets[((crc(key) & 0xFFFFFFFF) % num_buckets) % fine].append(line)
            return
        for line in lines:
            match = LINE_KEY.match(line)
            if match:
                sub_buckets[((crc(match.group(1)) & 0xFFFFFFFF) % num_buckets) % fine].append(line)

    # Group reads by file so each worker opens each part file at most once per shard.
    by_file: dict[str, list[tuple[int, int]]] = {}
    for part_str, offset, length in chunk_locs:
        by_file.setdefault(part_str, []).append((offset, length))
    for part_str, spans in by_file.items():
        with open(part_str, 'rb') as src:
            for offset, length in spans:
                src.seek(offset)
                raw = dctx.decompress(src.read(length))
                route_sub(raw)
    if extra_lines:
        route_sub(b'\n'.join(extra_lines) + b'\n')

    lengths = [0] * fine
    with open(out_str, 'wb') as out:
        for fine_id, lines in enumerate(sub_buckets):
            if not lines:
                continue
            compressed = cctx.compress(b'\n'.join(lines) + b'\n')
            out.write(compressed)
            lengths[fine_id] = len(compressed)
            lines.clear()

    return coarse_id, out_str, lengths


def build_index(
    export_path: str | Path,
    target_path: str | Path | None = None,
    *,
    num_buckets: int = DEFAULT_BUCKETS,
    workers: int | None = None,
    source_modified: str | None = None,
) -> JsonObject:
    """Convert `export_path` into a single-file hash-bucketed archive at `target_path`."""
    from .evals import modified

    source = Path(export_path)
    if not source.is_file():
        raise FileNotFoundError(f'No evaluation export at {source}')
    target = index_path_for(source) if target_path is None else Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    modified_iso = modified(source) if source_modified is None else source_modified
    coarse, fine = _split_dimensions(num_buckets)
    cpu = os.cpu_count() or 1
    pass1_workers = max(1, min(cpu - 2 if cpu >= 4 else cpu, 6)) if workers is None else max(1, workers)
    pass2_workers = max(1, min(pass1_workers, 4))

    tmp_dir = target.parent / f'.{target.stem}.index-tmp'
    tmp_target = target.with_name(target.name + '.tmp')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    try:
        frames = pzstd_frames(source)
        shard_chunks: list[list[tuple[str, int, int]]] = [[] for _ in range(coarse)]
        boundary_lines: list[list[bytes]] = [[] for _ in range(coarse)]
        total_lines = 0

        if frames and pass1_workers > 1 and len(frames) >= pass1_workers * 2:
            num_tasks = min(len(frames), max(pass1_workers * 4, 32))
            step = math.ceil(len(frames) / num_tasks)
            slices = [frames[i : i + step] for i in range(0, len(frames), step)]
            tasks = [
                (
                    tid,
                    str(source),
                    sl,
                    tid > 0,
                    tid + 1 < len(slices),
                    str(tmp_dir / f'part_{tid:02d}.tmp'),
                    num_buckets,
                    fine,
                )
                for tid, sl in enumerate(slices)
            ]
            print(
                f'Index pass 1/2: partitioning {source.name} ({len(frames):,} pzstd frames) '
                f'with {pass1_workers} workers',
                flush=True,
            )
            heads: dict[int, bytes] = {}
            tails: dict[int, bytes] = {}
            bytes_done = 0
            p1_start = p1_printed = time.monotonic()
            with ProcessPoolExecutor(max_workers=pass1_workers) as pool:
                futures = [pool.submit(_pass1_frames_task, t) for t in tasks]
                for future in as_completed(futures):
                    tid, part_str, chunks, head_frag, tail_frag, n_lines, n_bytes = future.result()
                    heads[tid] = head_frag
                    tails[tid] = tail_frag
                    total_lines += n_lines
                    bytes_done += n_bytes
                    for cid, off, length in chunks:
                        shard_chunks[cid].append((part_str, off, length))
                    if time.monotonic() - p1_printed >= Progress.INTERVAL:
                        p1_printed = time.monotonic()
                        share = bytes_done / stat.st_size if stat.st_size else 1.0
                        elapsed = time.monotonic() - p1_start
                        left = f', ~{duration(elapsed * (1 - share) / share)} left' if 0 < share < 1 else ''
                        print(f'Index pass 1/2: {100 * share:.0f}% partitioned{left}', flush=True)
            # Stitch the handful of lines that crossed contiguous slice boundaries.
            crc = zlib.crc32
            for tid in range(len(slices) - 1):
                stitched = tails[tid] + heads[tid + 1]
                if not stitched:
                    continue
                for line in stitched.split(b'\n'):
                    if not line:
                        continue
                    match = LINE_KEY.match(line)
                    if match:
                        cid = ((crc(match.group(1)) & 0xFFFFFFFF) % num_buckets) // fine
                        boundary_lines[cid].append(line)
                        total_lines += 1
            print(f'Index pass 1/2: 100% partitioned ({total_lines:,} lines)', flush=True)
        else:
            print(f'Index pass 1/2: partitioning {source.name} (streaming)', flush=True)
            part_path = tmp_dir / 'part_00.tmp'
            chunks, total_lines = _pass1_stream(source, part_path, num_buckets, fine)
            for cid, off, length in chunks:
                shard_chunks[cid].append((str(part_path), off, length))
            print(f'Index pass 1/2: 100% partitioned ({total_lines:,} lines)', flush=True)

        # Pass 2: split each coarse shard into `fine` buckets and append sequentially to `tmp_target`.
        table: list[tuple[int, int]] = [(0, 0)] * num_buckets
        non_empty_shards = [cid for cid in range(coarse) if shard_chunks[cid] or boundary_lines[cid]]
        print(
            f'Index pass 2/2: writing {num_buckets:,} buckets across {len(non_empty_shards):,} shards '
            f'with {pass2_workers} workers',
            flush=True,
        )
        p2_start = p2_printed = time.monotonic()
        done_shards = 0

        def append_shard(out: BinaryIO, cid: int, shard_file: str, lengths: list[int]) -> None:
            shard_path = Path(shard_file)
            base_bucket = cid * fine
            try:
                with shard_path.open('rb') as src:
                    for fine_id, length in enumerate(lengths):
                        if not length:
                            continue
                        offset = out.tell()
                        remaining = length
                        while remaining > 0:
                            buf = src.read(min(remaining, 1 << 20))
                            if not buf:
                                raise EOFError(f'Truncated shard output: {shard_path}')
                            out.write(buf)
                            remaining -= len(buf)
                        table[base_bucket + fine_id] = (offset, length)
            finally:
                shard_path.unlink(missing_ok=True)

        with tmp_target.open('wb') as out:
            out.write(b'\x00' * HEADER.size)
            if pass2_workers > 1 and len(non_empty_shards) >= 4:
                with ProcessPoolExecutor(max_workers=pass2_workers) as pool:
                    pending_futs = []
                    # Keep at most 2 * pass2_workers shards in flight so temporary disk stays tiny.
                    shard_iter = iter(non_empty_shards)

                    def submit_next() -> bool:
                        try:
                            cid = next(shard_iter)
                        except StopIteration:
                            return False
                        out_str = str(tmp_dir / f'shard_{cid:03d}.tmp')
                        pending_futs.append(
                            pool.submit(
                                _pass2_shard_task,
                                (cid, shard_chunks[cid], boundary_lines[cid], out_str, num_buckets, fine),
                            )
                        )
                        return True

                    for _ in range(pass2_workers * 2):
                        if not submit_next():
                            break
                    while pending_futs:
                        finished = next(as_completed(pending_futs))
                        pending_futs.remove(finished)
                        cid, shard_file, lengths = finished.result()
                        append_shard(out, cid, shard_file, lengths)
                        done_shards += 1
                        submit_next()
                        if time.monotonic() - p2_printed >= Progress.INTERVAL:
                            p2_printed = time.monotonic()
                            share = done_shards / len(non_empty_shards)
                            elapsed = time.monotonic() - p2_start
                            left = f', ~{duration(elapsed * (1 - share) / share)} left' if 0 < share < 1 else ''
                            print(f'Index pass 2/2: {100 * share:.0f}% written{left}', flush=True)
            else:
                for cid in non_empty_shards:
                    out_str = str(tmp_dir / f'shard_{cid:03d}.tmp')
                    _, shard_file, lengths = _pass2_shard_task(
                        (cid, shard_chunks[cid], boundary_lines[cid], out_str, num_buckets, fine)
                    )
                    append_shard(out, cid, shard_file, lengths)
                    done_shards += 1

            table_offset = out.tell()
            table_bytes = bytearray(num_buckets * ENTRY.size)
            for i, (off, length) in enumerate(table):
                ENTRY.pack_into(table_bytes, i * ENTRY.size, off, length)
            out.write(table_bytes)
            metadata_offset = out.tell()
            metadata: JsonObject = dict(
                source_name=source.name,
                source_size=stat.st_size,
                source_modified=modified_iso,
                hash='crc32',
                num_buckets=num_buckets,
                lines=total_lines,
            )
            meta_bytes = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
            out.write(meta_bytes)
            out.seek(0)
            out.write(HEADER.pack(MAGIC, VERSION, num_buckets, table_offset, metadata_offset, len(meta_bytes)))

        tmp_target.replace(target)
        elapsed_total = time.monotonic() - started
        print(
            f'Index complete: {target} ({target.stat().st_size / 1e9:.2f} GB, {total_lines:,} lines, '
            f'{num_buckets:,} buckets) in {duration(elapsed_total)}',
            flush=True,
        )
        return dict(metadata, path=str(target), index_size=target.stat().st_size, seconds=elapsed_total)
    finally:
        tmp_target.unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


class EvalIndex:
    """Random-access reader over a single-file hash-bucketed evaluation archive."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open('rb') as raw:
            header_bytes = raw.read(HEADER.size)
            if len(header_bytes) != HEADER.size:
                raise ValueError(f'Corrupt evaluation index (truncated header): {self.path}')
            magic, version, num_buckets, table_offset, meta_offset, meta_len = HEADER.unpack(header_bytes)
            if magic != MAGIC or version != VERSION:
                raise ValueError(f'Unsupported evaluation index format: {self.path}')
            raw.seek(meta_offset)
            self.metadata: JsonObject = json.loads(raw.read(meta_len).decode('utf-8'))
            self.num_buckets = num_buckets
            self.table_offset = table_offset

    @property
    def source_identity(self) -> tuple[str, int, str]:
        return (
            str(self.metadata['source_name']),
            int(self.metadata['source_size']),
            str(self.metadata['source_modified']),
        )

    @property
    def retrieved_at(self) -> str:
        return str(self.metadata['source_modified'])

    def matches_export(self, export_path: Path) -> bool:
        """Whether this index was built from `export_path` (matching name, size and modification time)."""
        from .evals import modified

        if not export_path.is_file():
            return False
        stat = export_path.stat()
        return self.source_identity == (export_path.name, stat.st_size, modified(export_path))

    def _read_table(self, raw: BinaryIO) -> bytes:
        raw.seek(self.table_offset)
        size = self.num_buckets * ENTRY.size
        data = raw.read(size)
        if len(data) != size:
            raise ValueError(f'Corrupt evaluation index (truncated bucket table): {self.path}')
        return data

    def lookup(
        self,
        wanted: Collection[Position],
        *,
        label: str = 'Evaluation index',
    ) -> Iterator[tuple[Position, Any]]:
        """Yield `(position, Evaluation)` for every evaluation in the index matching `wanted`."""
        from .evals import from_export, position_of, wanted_lines

        wanted_set = set(wanted)
        if not wanted_set:
            return
        by_bucket: dict[int, set[bytes]] = {}
        for pos in wanted_set:
            prefix = pos.rsplit(' ', 1)[0].encode()
            by_bucket.setdefault(bucket_for(prefix, self.num_buckets), set()).add(prefix)

        retrieved_at = self.retrieved_at
        bucket_ids = sorted(by_bucket)
        found = 0
        started = printed = time.monotonic()

        with self.path.open('rb') as raw:
            table_bytes = self._read_table(raw)
            spans: list[tuple[int, int, int, set[bytes]]] = []
            for bid in bucket_ids:
                offset, length = ENTRY.unpack_from(table_bytes, bid * ENTRY.size)
                if length:
                    spans.append((bid, offset, length, by_bucket[bid]))

            if not spans:
                return

            # Parallelize zstd decompression only for large multi-bucket queries.
            if len(spans) >= 128:
                total_spans = len(spans)

                def decode_batch(batch: list[tuple[int, bytes, set[bytes]]]) -> list[tuple[Position, Any]]:
                    dctx = zstandard.ZstdDecompressor()
                    hits: list[tuple[Position, Any]] = []
                    for _, comp, prefixes in batch:
                        data = dctx.decompress(comp)
                        for line in wanted_lines(data, len(data), prefixes):
                            item = json.loads(line)
                            if (k := position_of(item['fen'])) in wanted_set:
                                hits.append((k, from_export(item, retrieved_at)))
                    return hits

                workers = min(os.cpu_count() or 1, 6)
                batch_size = 16
                done = 0
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    pending = []
                    idx = 0

                    def read_batch() -> list[tuple[int, bytes, set[bytes]]]:
                        nonlocal idx
                        batch = []
                        end_idx = min(idx + batch_size, total_spans)
                        while idx < end_idx:
                            bid, offset, length, prefixes = spans[idx]
                            raw.seek(offset)
                            batch.append((bid, raw.read(length), prefixes))
                            idx += 1
                        return batch

                    for _ in range(workers * 2):
                        if idx < total_spans:
                            pending.append(pool.submit(decode_batch, read_batch()))
                    while pending:
                        fut = pending.pop(0)
                        for pair in fut.result():
                            found += 1
                            yield pair
                        done = min(total_spans, done + batch_size)
                        if idx < total_spans:
                            pending.append(pool.submit(decode_batch, read_batch()))
                        if time.monotonic() - printed >= Progress.INTERVAL:
                            printed = time.monotonic()
                            share = done / total_spans
                            elapsed = time.monotonic() - started
                            left = f', ~{duration(elapsed * (1 - share) / share)} left' if 0 < share < 1 else ''
                            print(f'{label}: {100 * share:.0f}% read, {found:,} positions found{left}', flush=True)
                return

            dctx = zstandard.ZstdDecompressor()
            for _, offset, length, prefixes in spans:
                raw.seek(offset)
                data = dctx.decompress(raw.read(length))
                for line in wanted_lines(data, len(data), prefixes):
                    item = json.loads(line)
                    if (k := position_of(item['fen'])) in wanted_set:
                        found += 1
                        yield k, from_export(item, retrieved_at)


def read_index_metadata(path: str | Path) -> Mapping[str, Any]:
    """Read only the metadata block of an index file."""
    return EvalIndex(path).metadata
