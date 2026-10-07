"""Fetch every Explorer table both repertoires need, so every later stage can run offline."""

import argparse
from pathlib import Path

from . import studies
from .context import DEFAULT_CACHE
from .explorer import DEFAULT_FILTERS, Explorer, add_token_option, apply_token_file, fetch_missing, survey
from .graph import parse
from .score import config_path, inspect_repertoire, load_config, parent_positions, plan_repertoire, required_positions


def required_tables(pgn, color, config):
    """Scoring tables, then own-move parent tables, for one repertoire PGN. No table is read."""
    graph = parse(pgn, config.get('exclude', []))
    plan = plan_repertoire(graph, color, config, inspect_repertoire(graph, color))
    return list(dict.fromkeys([*required_positions(plan, color), *parent_positions(plan, color)]))


def fetch(pgns, configs, cache=DEFAULT_CACHE, dry_run=False, extra=None):
    """Cache the tables for {color: pgn} with {color: config path or None}.

    Colors with the same Explorer filters share one pass, so the run has a single count and estimate.
    `extra` adds {color: positions}, such as saved comparisons. `dry_run` reports what would be fetched
    and makes no requests.
    """
    groups = {}
    for color, pgn in pgns.items():
        config = load_config(configs.get(color))
        filters = dict(DEFAULT_FILTERS, **config.get('filters', {}))
        group = groups.setdefault(tuple(sorted(filters.items())), dict(filters=filters, colors=[], positions=[]))
        group['colors'].append(color)
        group['positions'] += required_tables(pgn, color == 'white', config)
        group['positions'] += (extra or {}).get(color, [])
    for group in groups.values():
        label = ' and '.join(group['colors'])
        positions = list(dict.fromkeys(group['positions']))
        explorer = Explorer(cache, group['filters'], offline=dry_run)
        try:
            if dry_run:
                survey(explorer, positions, label, fetching=True)
            else:
                fetch_missing(explorer, positions, label)
        finally:
            explorer.close()


def main():
    from . import compare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('white_pgn', nargs='?', help='White PGN; omit both to use the exported studies')
    parser.add_argument('black_pgn', nargs='?')
    parser.add_argument('--studies', default=studies.DIRECTORY, help='Folder of exported study PGNs')
    parser.add_argument('--white-config', help='Configuration (default: configs/white.json, created if missing)')
    parser.add_argument('--black-config', help='Configuration (default: configs/black.json, created if missing)')
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument('--dry-run', action='store_true', help='Count the tables to fetch and estimate the time')
    parser.add_argument('--comparisons', default=compare.REGISTRY, help='Saved comparisons whose tables to fetch')
    add_token_option(parser)
    args = parser.parse_args()
    apply_token_file(parser, args)
    if (args.white_pgn is None) != (args.black_pgn is None):
        parser.error('give both PGN paths, or neither to use the exported studies')
    if args.white_pgn is None:
        pgns = studies.default_paths(args.studies)
    else:
        pgns = dict(white=Path(args.white_pgn), black=Path(args.black_pgn))
    configs = dict(white=config_path('white', args.white_config), black=config_path('black', args.black_config))
    try:
        # Saved comparisons share the pass, as in a build.
        extra = compare.registry_tables(compare.load_registry(args.comparisons), pgns, configs)
        fetch(pgns, configs, args.cache, args.dry_run, extra)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f'Fetch failed: {exc}\nTables fetched so far are cached; rerun to continue.\n')


if __name__ == '__main__':
    main()
