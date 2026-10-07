"""The comparison page: whether a candidate study's alternatives improve your repertoire, rendered from saved JSON."""

import os
from pathlib import Path

from .format import (
    centipawn_delta,
    display,
    escape,
    line,
    number,
    opponent_rating,
    percentage,
    score_points,
)
from .links import analysis_url
from .markdown import section, table

# Opponent-rating gaps at least this large between compared moves get a caveat.
RATING_GAP = 100


def relative(target, page):
    try:
        return Path(os.path.relpath(target, Path(page).parent)).as_posix()
    except ValueError:
        return Path(target).as_posix()


def linked(route, position):
    return f'[{line(route)}]({analysis_url(position, route)})' if route else '(starting position)'


def last_move(route):
    return line(route.split()[-1]) if route.split() else 'the start'


def change(value, digits=2):
    """A score difference in percentage points; whole-repertoire changes are often tiny, so they get a digit more."""
    if value is not None and digits == 2 and abs(100 * value) < 0.1:
        digits = 3
    return score_points(None if value is None else 100 * value, digits, signed=True)


def headline(after, before):
    """A whole-repertoire change, with its centipawn translation as in the main report's headlines."""
    cp = centipawn_delta(after, before)
    return change(None if after is None or before is None else after - before) + (
        '' if cp is None else f' ({number(cp, 0, signed=True)} cp)'
    )


def bounds(interval):
    return ' to '.join(score_points(v, 2, signed=True) for v in interval) if interval else 'unavailable'


def reply_label(route):
    """The opponent move a decision answers, such as 2...Nf6."""
    return escape(route.split()[-1]) if route.split() else 'the start'


def uncertain(bounds):
    return bounds is not None and bounds[0] <= 0 <= bounds[1]


def option_name(option, page):
    """The move, then where it comes from: your chapter pages or the candidate's Lichess chapters."""
    if option['source'] == 'repertoire':
        sources = ', '.join(
            f"[{c['page']}]({relative(Path(page).parent.parent / 'chapters' / (c['page'] + '.md'), page)})"
            for c in option['repertoire_chapters']
        )
        return f"{escape(option['label'])} (yours{': ' + sources if sources else ''})"
    names = ', '.join(
        f"[{escape(c['name'])}]({c['url']})" if c['url'] else escape(c['name']) for c in option['chapters']
    )
    return f"{escape(option['label'])} ({names})"


def chosen(decision, plan):
    return next(o for o in decision['options'] if o[plan])


def verdict(data):
    """One sentence: which alternatives to adopt and where to keep your moves, grouped by move."""
    adopt, keep = {}, {}
    for d in sorted(data['decisions'], key=lambda d: -(d['reach'] or 0)):
        if d['parent'] is not None:
            continue
        best = chosen(d, 'improving')
        target = adopt if best['index'] else keep
        name = escape(best['label']) if best['move'] else 'no preparation'
        target.setdefault(name, []).append(reply_label(d['line']))
    adopted = [f'{move} against {join(replies)}' for move, replies in adopt.items()]
    kept = [f'{move} against {join(replies)}' for move, replies in keep.items()]
    if not adopted:
        return '**Keep your repertoire:** no alternative in the candidate improves it.'
    text = f"**Adopt {join(adopted)}**" + (f"; keep {join(kept)}" if kept else '') + '.'
    unsure = [
        f"{escape(chosen(d, 'improving')['label'])} against {reply_label(d['line'])}"
        for d in data['decisions']
        if d['parent'] is None
        and chosen(d, 'improving')['index']
        and uncertain(chosen(d, 'improving')['overall_interval'])
    ]
    if unsure:
        text += f" The gain from {join(unsure)} is within its 95% interval of no change."
    return text


def join(items):
    if len(items) == 1:
        return items[0]
    # Items that already contain "and" need a serial comma to stay readable.
    last = ', and ' if len(items) > 2 or any(' and ' in item for item in items) else ' and '
    return ', '.join(items[:-1]) + last + items[-1]


def scenario_table(data):
    scenarios, color = data['scenarios'], data['color'].title()
    entry = data['entry']
    has_entry = bool(entry['line'])
    top = [d for d in data['decisions'] if d['parent'] is None]
    adopted = sum(1 for d in top if chosen(d, 'improving')['index'])
    rows = []
    for key, name in (
        ('current', 'Your repertoire'),
        ('improving', f'Improving alternatives only ({adopted} of {len(top)})'),
        ('all', f'All alternatives ({len(top)} of {len(top)})'),
    ):
        s = scenarios[key]
        first = key == 'current'
        row = [name]
        if has_entry:
            row += [percentage(s['entry_score']), '' if first else change(s['entry_change'])]
        row += [
            percentage(s['score']),
            '' if first else headline(s['score'], scenarios['current']['score']),
            '' if first else bounds(s['interval']),
        ]
        rows.append(row)
    headers = ['Scenario']
    if has_entry:
        headers += [f"After {line(entry['line'])}", 'Change']
    headers += [f'{color} repertoire', 'Change', '95% interval of change']
    return table(headers, rows)


def decision_rows(data, page):
    rows = []
    order = sorted([d for d in data['decisions'] if d['parent'] is None], key=lambda d: (-(d['reach'] or 0), d['line']))
    nested = {}
    for d in data['decisions']:
        if d['parent'] is not None:
            nested.setdefault(d['parent'][0], []).append(d)
    sequence = []

    def add(d, depth):
        sequence.append((d, depth))
        for child in nested.get(d['id'], []):
            add(child, depth + 1)

    for d in order:
        add(d, 0)
    by_id = {d['id']: d for d in data['decisions']}
    for d, depth in sequence:
        for i, option in enumerate(d['options']):
            where = ''
            if i == 0:
                where = linked(d['line'], d['position']) + f" · [details](#decision-{d['id'] + 1})"
                if depth:
                    parent = by_id[d['parent'][0]]['options'][d['parent'][1]]
                    where = f"Inside {escape(parent['label'])}: {where}"
            mark = []
            if option['improving']:
                mark.append('Adopt' if option['index'] else 'Keep')
                if option['index'] and uncertain(option['overall_interval']):
                    mark[-1] += ' (uncertain)'
            if option['all'] and option['index'] and len(d['options']) > 2:
                mark.append('In all alternatives')
            rows.append(
                [
                    where,
                    percentage(d['reach']) if i == 0 and d['reach'] is not None else '',
                    option_name(option, page),
                    percentage(option['score']),
                    '' if i == 0 else change(option['change'], 1),
                    '' if i == 0 else change(option['overall_change']),
                    '' if i == 0 else bounds(option['interval']),
                    '<br>'.join(mark),
                ]
            )
    color = data['color'].title()
    return table(
        [
            'Decision point',
            'Position reach',
            'Option',
            'Repertoire score here',
            'Change here',
            f'{color} repertoire change',
            '95% interval here',
            'Verdict',
        ],
        rows,
    )


def preparation_table(data):
    scenarios = data['scenarios']
    metrics = [scenarios[k]['entry_metrics'] or {} for k in ('current', 'improving', 'all')]
    return table(
        ['Measure', 'Your repertoire', 'Improving only', 'All alternatives'],
        [
            ['Expected prepared own moves', *[number(m.get('expected_moves')) for m in metrics]],
            ['Own moves to know', *[f"{m['distinct_decisions']:,}" if m else 'unavailable' for m in metrics]],
            ['Equivalent gap reach', *[percentage(m.get('equivalent_gap_reach')) for m in metrics]],
            ['Largest single gap', *[percentage(m.get('largest_gap_reach')) for m in metrics]],
            ['Effective opponent replies', *[number(m.get('effective_replies')) for m in metrics]],
            ['Opponent rating of score evidence', *[opponent_rating(m.get('opponent_rating')) for m in metrics]],
        ],
    )


def caveats(data):
    notes, weaker, stronger = [], [], []
    for d in sorted(data['decisions'], key=lambda d: -(d['reach'] or 0)):
        options = d['options']
        yours = (options[0]['metrics'] or {}).get('opponent_rating') or {}
        for option in options[1:]:
            theirs = (option['metrics'] or {}).get('opponent_rating') or {}
            if yours.get('mean') is None or theirs.get('mean') is None:
                continue
            gap = theirs['mean'] - yours['mean']
            if abs(gap) >= RATING_GAP:
                (weaker if gap < 0 else stronger).append(
                    f"{escape(option['label'])} after {reply_label(d['line'])} "
                    f"({opponent_rating(theirs)} against {opponent_rating(yours)})"
                )
    for rows, side in ((weaker, 'lower'), (stronger, 'higher')):
        if rows:
            notes.append(
                f"Candidate scores come from {side}-rated opponents than yours for {join(rows)}. Scores are not "
                'adjusted for rating, so part of each difference may reflect the '
                'opposition rather than the moves.'
            )
    by_id = {d['id']: d for d in data['decisions']}
    for group in data['interactions']:
        if abs(group['together'] - group['separately']) < 5e-5:
            continue
        names = join(
            [
                f"{escape(chosen(by_id[i], group['plan'])['label'])} after {reply_label(by_id[i]['line'])}"
                for i in group['decisions']
            ]
        )
        scenario = 'improving only' if group['plan'] == 'improving' else 'all alternatives'
        notes.append(
            f"In {scenario}, {names} share positions through transpositions: together they change the "
            f"repertoire by {change(group['together'], 3)}, against {change(group['separately'], 3)} separately."
        )
    competing = [d for d in data['decisions'] if len(d['options']) > 2]
    for d in competing:
        best = chosen(d, 'all')
        notes.append(
            f"After {line(d['line'])}, the candidate chapters compete; all alternatives uses the higher-scoring "
            f"{escape(best['label'])}."
        )
    if data['scenarios']['improving']['superseded_overrides']:
        notes.append(
            f"{len(data['scenarios']['improving']['superseded_overrides'])} explicit policy overrides in your "
            'configuration give way to adopted candidate lines.'
        )
    return [item for note in notes for item in (f'- {note}', '')]


def decision_section(d, data, page):
    color = data['color'].title()
    yours = d['options'][0]
    text = section(f"### After {line(d['line'])}", f"decision-{d['id'] + 1}")
    games = '' if d['games'] is None else f" {d['games']:,} database games;"
    text += [
        f"[Analysis board]({analysis_url(d['position'], d['line'])}) ·{games} database score "
        f"{percentage(d['database_score'])}." + ('' if d['reach'] is None else f" Reach {percentage(d['reach'])}."),
        '',
    ]
    for option in d['options'][1:]:
        if option['score'] is None or yours['score'] is None:
            continue
        sentence = (
            f"{escape(option['label'])} scores {percentage(option['score'])} here against "
            f"{percentage(yours['score'])} for {escape(yours['label'])} ({change(option['change'], 1)})."
        )
        if option['database_score'] is not None and yours['database_score'] is not None:
            sentence += (
                f" In the database alone {escape(option['label'])} scores {percentage(option['database_score'])} and "
                f"{escape(yours['label'])} {percentage(yours['database_score'])}, so preparation adds "
                f"{change(option['score'] - option['database_score'], 1)} and "
                f"{change(yours['score'] - yours['database_score'], 1)}."
            )
        text += [sentence, '']
    options = d['options']

    def metric(key, fmt):
        return [fmt((o['metrics'] or {}).get(key)) for o in options]

    text += table(
        ['Measure', *[option_name(o, page) for o in options]],
        [
            ['Repertoire score here', *[percentage(o['score']) for o in options]],
            [f'{color} repertoire change', *['' if not o['index'] else change(o['overall_change']) for o in options]],
            [
                '95% interval of the change here',
                *['' if not o['index'] else bounds(o['interval']) for o in options],
            ],
            [
                'Database score of the move',
                *[
                    'n/a'
                    if o['database_score'] is None
                    else f"{percentage(o['database_score'])} ({o['database_games']:,} games)"
                    for o in options
                ],
            ],
            ['Expected prepared own moves', *metric('expected_moves', number)],
            ['Own moves to know', *metric('distinct_decisions', lambda v: 'unavailable' if v is None else f'{v:,}')],
            ['Equivalent gap reach', *metric('equivalent_gap_reach', percentage)],
            ['Largest single gap', *metric('largest_gap_reach', percentage)],
            ['Effective opponent replies', *metric('effective_replies', number)],
            ['Opponent rating of score evidence', *metric('opponent_rating', opponent_rating)],
            ['Sparse evidence reach', *metric('sparse_mass', percentage)],
        ],
    )
    for option in options:
        if not option['replies']:
            continue
        rows = [
            [
                escape(r['san']),
                percentage(r['share']),
                'Yes' if r['prepared'] else 'No',
                percentage(r['score']),
                f"{r['games']:,}",
            ]
            for r in option['replies']
        ]
        body = [
            *table(['Reply', 'Share', 'Prepared', 'Score after', 'Games'], rows),
        ]
        gaps = (option['metrics'] or {}).get('gaps', [])
        if gaps:
            body += [
                f"Where {escape(option['label'])} preparation ends most often, as a share of games reaching this "
                'decision point:',
                '',
                *table(
                    ['Line', 'Share of games', 'Database score'],
                    [[linked(g['line'], g['position']), percentage(g['reach']), percentage(g['score'])] for g in gaps],
                ),
            ]
        summary = f"Replies to {escape(option['label'])}" + (' (yours)' if option['source'] == 'repertoire' else '')
        text += ['<details>', f'<summary>{summary}</summary>', '', *body, '</details>', '']
    return text


def source_argument(source):
    """How to name a candidate on the command line, without publishing local folders outside the repository."""
    if source['kind'] == 'lichess':
        return source['input']
    try:
        value = Path(os.path.relpath(source['path'])).as_posix()
        if value.startswith('..'):
            value = Path(source['path']).name
    except ValueError:
        value = Path(source['path']).name
    return f'"{value}"' if ' ' in value else value


def method_section(data, page):
    evidence, validation = data['evidence'], data['validation']
    sources = []
    for s in data['sources']:
        name = s['input'] if s['kind'] == 'lichess' else Path(s['path']).name
        sources.append(f"- `{escape(name)}`: {s['chapters']} chapters, SHA-256 `{s['sha256'][:16]}`")
    checks = ['your saved score reproduced exactly']
    checks.append(
        'improving choice adds up across groups'
        if validation.get('improving_additive')
        else 'improving choice is not additive across groups'
    )
    if 'adopt_pgn_reproduces_selection' in validation:
        checks.append(
            'adopt PGN reproduces the improving repertoire'
            if validation['adopt_pgn_reproduces_selection']
            else 'adopt PGN differs from the improving repertoire'
        )
    inputs = ' '.join(source_argument(s) for s in data['sources'])
    command = f"uv run repertoire compare {inputs} --color {data['color']} --name {data['name']}"
    if data['entry']['basis'] == 'given':
        command += f' --entry "{data["entry"]["line"]}"'
    return [
        '<details>',
        '<summary>Method, evidence and definitions</summary>',
        '',
        '- <a id="def-decision"></a>**Decision point:** each candidate chapter is followed along its first moves '
        'until it plays a different move from your repertoire, or adds a move where you have none. Chapters '
        'choosing the same move there share one option; competing moves are separate options. When chapters '
        'that share an option later disagree, that board is a nested decision point inside the option.',
        '- <a id="def-scenarios"></a>**Scenarios:** an adopted option places those candidate lines before your '
        'chapters, so they decide every board they record, and your preparation continues wherever they end or '
        'transpose. Every other candidate line is left out. Elsewhere, competing moves are decided by score, as '
        'in your repertoire. **Improving only** chooses at every decision point the option, including your own '
        'move, that gives the highest repertoire score. **All alternatives** always switches, and where '
        'candidate chapters compete it takes the higher-scoring candidate move.',
        '- <a id="def-linked"></a>**Linked decision points:** two decision points are chosen together when one '
        "option's changed positions can be reached from the other's (a transposition), or one is nested in the "
        f'other. Each linked group is searched over every combination (up to {256} per group); separate groups add up.',
        '- <a id="def-comparison-scores"></a>**Scores and changes:** repertoire scores count wins plus half of '
        'draws, from your side, conditional on reaching the position shown. Change here is the score at the '
        'decision point; repertoire change is the whole-color score with only that option adopted. Intervals are '
        'paired approximate 95% model intervals over the same cached tables; they ignore historical-game overlap '
        'and the selection of the best option, which favors options that scored well by chance. (uncertain) marks '
        'an adopted option whose interval includes no change.',
        '- <a id="def-comparison-measures"></a>**Preparation measures** follow the main report: '
        '[expected prepared depth](../report.md#def-depth), own moves to know (distinct reachable own decisions), '
        '[equivalent gap reach](../report.md#def-gap-reach), [effective replies](../report.md#def-effective-replies) '
        'and [opponent rating](../report.md#def-rating). Opponent ratings describe the games behind the score and '
        'do not adjust it.',
        '',
        f"Evidence: {evidence['tables']:,} Explorer tables, {evidence['fetched']:,} fetched for this comparison; "
        f"{evidence['scenarios_evaluated']:,} hypothetical repertoires scored. "
        f"Checks: {'; '.join(checks)}. Your repertoire scored "
        f"`{Path(data['manifest']['input_path']).name}` (SHA-256 `{data['manifest']['input_sha256'][:16]}`).",
        '',
        'Candidate sources:',
        '',
        *sources,
        '',
        f'Regenerate with `{command}`.',
        '',
        '</details>',
        '',
    ]


def render(data, page):
    with display(digits=1):
        return '\n'.join(_render(data, page))


def _render(data, page):
    color = data['color'].title()
    entry = data['entry']
    sources = ', '.join(
        f"[{escape(c['name'])}]({c['url']})" if c['url'] else escape(c['name']) for c in data['chapters']
    )
    report = relative(Path(page).parent.parent / 'report.md', page)
    summary = relative(Path(page).parent.parent / 'summary.md', page)
    top = [d for d in data['decisions'] if d['parent'] is None]
    text = [
        f"# {escape(data['title'])} vs your {color} repertoire",
        '',
        f"Candidate: {len(data['chapters'])} chapter{'s' if len(data['chapters']) != 1 else ''} "
        f"({sources}), {len(top)} decision point{'s' if len(top) != 1 else ''}"
        + (
            f" · compared after **{line(entry['line'])}** ({percentage(entry['reach'])} of {color} games)"
            if entry['line']
            else ''
        )
        + f" · {data['created_at'][:10]} · [Your report]({report}) · [Summary]({summary})",
        '',
        verdict(data),
        '',
        *scenario_table(data),
        "Scores count draws as half a point, from your side. Improving only keeps your move wherever the candidate "
        "does not score better. [Definitions](#def-scenarios).",
        '',
    ]
    notes = caveats(data)
    if notes:
        text += ['**Read with care:**', '', *notes]
    text += section('## Decision points', 'decision-points')
    text += [
        'One row per option at each board where the candidate plays a different move from you, most reached '
        'first. Changes are against your move. [Definitions](#def-decision).',
        '',
        *decision_rows(data, page),
    ]
    text += section(f"## Preparation after {line(entry['line'])}" if entry['line'] else '## Preparation', 'preparation')
    text += [
        'How much you would need to know, and how often games leave preparation, from the comparison position.',
        '',
        *preparation_table(data),
    ]
    if data['outputs'].get('adopt_pgn'):
        adopt = relative(data['outputs']['adopt_pgn'], page)
        text += [
            f"[Adopt PGN]({adopt}): the candidate lines of the improving choice, ready to import into your study as "
            'chapters. Where they compete with your chapters, the build plays the higher-scoring move.',
            '',
        ]
    text += section('## Each decision point', 'details')
    for d, _ in [
        (d, 0) for d in sorted(data['decisions'], key=lambda d: (d['parent'] is not None, -(d['reach'] or 0)))
    ]:
        text += decision_section(d, data, page)
    text += section('## Method and evidence', 'method')
    text += method_section(data, page)
    return text
