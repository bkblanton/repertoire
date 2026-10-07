"""The definitions and evidence glossary at the end of the full report."""
from importlib.resources import files

from .bundle import FAMILIES
from .format import display_source, escape
from .markdown import section

# One glossary entry per line, each with its own `def-*` anchor that tables link to.
DEFINITIONS = files(__package__).joinpath('definitions.md').read_text(encoding='utf-8').splitlines()


def methods(bundles):
    text = section('## Definitions and evidence', 'methods')
    text += DEFINITIONS
    text += ['<details>', '<summary>Saved analysis files and validation</summary>', '']
    for b in bundles:
        r = b['report']; m = r['manifest']; color = r['color'].title()
        text += [f'### {color} evidence', '',
                 f"Analyzed {m['created_at']}. PGN: `{escape(display_source(m['input_path'], b['path']))}`.", '',
                 f"Input SHA-256: `{m['input_sha256']}`. Score SHA-256: `{b['digest']}`.", '',
                 f"{m['positions']:,} graph positions; {m['evaluated_positions']:,} evaluated positions; "
                 f"Owner W/D/L prior {m['prior']}; sparse threshold {m['sparse_threshold']}.", '',
                 'Data files: ' + ' | '.join(f'[{label}]({path.name})' for label, path in
                     [('Scores', b['path'])] + [(family.title(), b['path'].with_suffix(f'.{family}.json')) for family in FAMILIES if family in b]) + '.', '']
        checks = ['Scoring sanity checks passed' if r.get('diagnostics', {}).get('sanity_checks_passed') else 'Scoring sanity checks unavailable']
        if 'vulnerabilities' in b:
            checks.append(f"vulnerability candidate-child queries: {b['vulnerabilities']['manifest']['candidate_child_queries']}")
        if 'preparation' in b or 'character' in b:
            checks.append('preparation and character use cached evidence only')
        text += ['; '.join(checks) + '. Companion score hashes and filters were checked before assembly.', '']
    return text + ['</details>', '']
