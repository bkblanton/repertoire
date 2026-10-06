"""Export the White and Black Lichess studies to local PGN files."""
import argparse
import json
import os
from pathlib import Path
import re
import time

import httpx

from .graph import parse

SOURCES = 'studies.json'
DIRECTORY = 'studies'
ENDPOINT = 'https://lichess.org/api/study/{}.pgn'
STUDY = re.compile(r'^(?:https?://)?(?:www\.)?lichess\.org/study/([A-Za-z0-9]{8})(?:/[A-Za-z0-9]{8})?/?(?:[?#].*)?$')


def study_id(value):
    """Accept a bare study ID or a study/chapter URL; chapter links export the whole study."""
    value = value.strip()
    if re.fullmatch(r'[A-Za-z0-9]{8}', value):
        return value
    match = STUDY.match(value)
    if not match:
        raise ValueError(f'Not a Lichess study URL or ID: {value}')
    return match.group(1)


def load_sources(path=SOURCES):
    sources = json.loads(Path(path).read_text(encoding='utf-8'))
    missing = [color for color in ('white', 'black') if not sources.get(color)]
    if missing:
        raise ValueError(f'{path} needs a study URL for {", ".join(missing)}')
    return {color: sources[color] for color in ('white', 'black')}


def default_paths(directory=DIRECTORY):
    return {color: Path(directory) / f'{color}.pgn' for color in ('white', 'black')}


def comparable(text):
    # Lichess stamps the export date into every chapter's Date header.
    return [line for line in text.replace('\r\n', '\n').strip().split('\n') if not line.startswith('[Date ')]


def download(client, study, retries=3):
    url = ENDPOINT.format(study_id(study))
    params = dict(clocks='false', comments='true', variations='true')
    for attempt in range(retries + 1):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError as exc:
            if attempt == retries:
                raise RuntimeError(f'Study export transport failure ({type(exc).__name__}) for {study}') from None
            time.sleep(2**attempt)
            continue
        if response.status_code == 429 and attempt < retries:
            print('Lichess HTTP 429; backing off 60s', flush=True)
            time.sleep(60)
            continue
        if response.status_code in (401, 403, 404):
            raise RuntimeError(f'Lichess HTTP {response.status_code} for {study}: the study must exist and the '
                               'token needs the study:read scope for private studies')
        if response.status_code != 200:
            raise RuntimeError(f'Lichess HTTP {response.status_code} for {study}')
        return response.text
    raise AssertionError('Retry loop exhausted')


def fetch(sources=SOURCES, directory=DIRECTORY, client=None):
    """Download both studies; leave a file untouched when only its export date changed."""
    studies = load_sources(sources) if isinstance(sources, (str, Path)) else sources
    token = os.environ.get('LICHESS_TOKEN', '').strip()
    if not token:
        raise ValueError('Set LICHESS_TOKEN in the environment to export studies')
    paths = default_paths(directory)
    Path(directory).mkdir(parents=True, exist_ok=True)
    owned = client is None
    client = client or httpx.Client(headers={'Authorization': f'Bearer {token}'}, timeout=60)
    try:
        for color, study in studies.items():
            text = download(client, study)
            path = paths[color]
            temporary = path.with_suffix('.pgn.tmp')
            temporary.write_text(text, encoding='utf-8', newline='\n')
            try:
                # Validate before replacing the previous export.
                chapters = len(parse(temporary).chapters)
            except ValueError:
                temporary.unlink()
                raise
            if path.is_file() and comparable(path.read_text(encoding='utf-8-sig')) == comparable(text):
                temporary.unlink()
                print(f'{color}: {chapters} chapters, unchanged ({path})', flush=True)
            else:
                temporary.replace(path)
                print(f'{color}: {chapters} chapters, updated ({path})', flush=True)
    finally:
        if owned:
            client.close()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', default=SOURCES, help='JSON file with white and black study URLs')
    parser.add_argument('--directory', default=DIRECTORY)
    args = parser.parse_args()
    try:
        fetch(args.sources, args.directory)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f'Study export failed: {exc}\n')


if __name__ == '__main__':
    main()
