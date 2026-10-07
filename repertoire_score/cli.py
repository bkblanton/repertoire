"""The `repertoire` command: one entry point that dispatches to each stage's own parser."""

import argparse
import importlib
import sys

# Subcommand, module, one-line help. Each module keeps its own main() and argument parser.
COMMANDS = [
    ('build', 'build', 'Export the studies, fetch Explorer tables, score both colors and write every report'),
    ('export', 'studies', 'Export the White and Black Lichess studies to local PGN files'),
    ('fetch', 'fetch', 'Fetch the Explorer tables both repertoires need; slow, resumable, and estimated'),
    ('score', 'score', 'Inspect or score one repertoire PGN'),
    ('report', 'render', 'Render the Markdown reports from saved results, offline'),
    ('compare', 'compare', 'Compare a candidate study or PGN with your repertoire, alternative by alternative'),
    ('vulnerabilities', 'vulnerabilities', 'Rank strengths and vulnerabilities from saved scores'),
    ('preparation', 'preparation', 'Stopping outcomes, prepared-depth distributions and entry routes'),
    ('character', 'character', 'Position reach, gaps, reuse, reply variety and position profiles'),
    ('ratings', 'ratings', 'Opponent rating contexts for chapters and lines'),
    ('openings', 'openings', 'Opening names, reach and first-entry cohorts'),
    ('insights', 'report_insights', 'Gap priorities, branch-score spread and gain intervals'),
    ('correlations', 'position_correlations', 'Prepared depth versus continuation gain'),
    ('rating-correlations', 'rating_correlations', 'Opponent rating versus score within parent positions'),
]


def parser():
    result = argparse.ArgumentParser(
        prog='repertoire',
        description='Score chess opening repertoires with Lichess Opening Explorer statistics.',
        epilog="Run 'repertoire <command> --help' for a command's options.",
    )
    commands = result.add_subparsers(title='commands', metavar='<command>')
    for name, _, summary in COMMANDS:
        # Options are parsed by the stage itself; this parser only lists the commands.
        commands.add_parser(name, help=summary, add_help=False)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    modules = {name: module for name, module, _ in COMMANDS}
    if not argv or argv[0] not in modules:
        parser().parse_args(argv or ['--help'])
        return
    name, rest = argv[0], argv[1:]
    module = importlib.import_module(f'repertoire_score.{modules[name]}')
    # Stage parsers take their usage name from argv[0].
    previous = sys.argv
    sys.argv = [f'repertoire {name}', *rest]
    try:
        module.main()
    except KeyboardInterrupt:
        # Every stage saves its work atomically, so stopping is safe and a rerun continues.
        print('\nStopped. Work saved so far is kept; rerun the same command to continue.', file=sys.stderr)
        raise SystemExit(130) from None
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
