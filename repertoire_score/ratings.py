"""Cache-only opponent rating contexts for chapters and individual lines.

Explorer averageRating describes the move maker. Never use our own move rows
as opponent ratings, or average repeatedly over the visited decision points.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import chess

from . import SCHEMA_VERSION
from .explorer import Explorer, counts
from .graph import key, parse
from .preparation import Evaluator, chess_facts


def rating(known=0., moment=0., *, basis, **details):
    if not -1e-9 <= known <= 1 + 1e-9:
        raise AssertionError('Rating coverage outside [0, 1]')
    known = min(1., max(0., float(known)))
    return dict(mean=moment / known if known else None,
                known_coverage=known, missing_coverage=1 - known,
                weighted_rating_sum=float(moment), basis=basis, **details)


def usable(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def reply_rating(data, move):
    """The specific preceding opponent move, with no child lookup."""
    row = next((r for r in (data or {}).get('moves', []) if r['uci'] == move), None)
    n = sum(counts(row)) if row else 0
    value = row.get('averageRating') if row else None
    known = float(n > 0 and usable(value))
    return rating(known, value if known else 0., basis='preceding opponent move row',
                  rated_observations=n if known else 0, observations=n)


def response_rating(data):
    """Opponent to move: games weight the move-maker ratings in this table."""
    total = sum(counts(data)) if data else 0
    known, moment = 0, 0.
    for row in (data or {}).get('moves', []):
        n = sum(counts(row))
        if usable(row.get('averageRating')):
            known += n
            moment += n * row['averageRating']
    return rating(known / total if total else 0., moment / total if total else 0.,
                  basis='current opponent response rows', rated_observations=known,
                  observations=total)


def mixture(parts, basis, *, origins=None):
    """Mix normalized rating contexts by model probability, never sample count."""
    parts = list(parts)
    mass = sum(p for p, _ in parts)
    if mass < 0 or any(p < 0 for p, _ in parts):
        raise ValueError('Negative rating mixture probability')
    result = rating(sum(p * r['known_coverage'] for p, r in parts) / mass if mass else 0.,
                    sum(p * r['weighted_rating_sum'] for p, r in parts) / mass if mass else 0.,
                    basis=basis)
    if origins is not None:
        result['origins'] = origins
    return result


def unavailable(basis='unavailable opponent rating'):
    return rating(basis=basis)


def stop_key(row):
    return '|'.join(str(row.get(k) or '') for k in ('parent_position', 'move', 'type'))


class Context:
    """Flow-aware local ratings and once-per-game stopping-evidence moments."""
    def __init__(self, evaluator, starts, initial=None):
        self.e = evaluator
        self.starts = starts
        self.initial = initial or {}
        self.reach = evaluator.reaches(starts)
        arrivals = defaultdict(list)
        for k, p in starts.items():
            if p > 0:
                arrivals[k].append((p, (initial or {}).get(k, unavailable('no preceding opponent move')),
                                    None, None))
        for k, mass in self.reach.items():
            if mass <= 0: continue
            for move, p, target in evaluator.edges[k]:
                if evaluator.facts[k]['turn'] != evaluator.color:
                    arrivals[target].append((mass * p, reply_rating(evaluator.evidence.get(k), move), k, move))
        self.local = {}
        for k, mass in self.reach.items():
            if mass <= 0: continue
            if evaluator.facts[k]['turn'] != evaluator.color:
                self.local[k] = response_rating(evaluator.evidence.get(k))
            else:
                parts = arrivals[k]
                if not math.isclose(sum(p for p, *_ in parts), mass, abs_tol=1e-9):
                    raise AssertionError('Own-turn rating arrival flow not conserved')
                self.local[k] = mixture([(p, r) for p, r, _, _ in parts], 'incoming opponent moves under scope policy',
                    origins=[dict(parent_position=parent, move=move, weight=p / mass,
                                  mean=r['mean'], known_coverage=r['known_coverage'],
                                  **({'first_entry_origins': r['origins']} if parent is None and r.get('origins') else {}))
                             for p, r, parent, move in parts if p > 0])
                if k == key(chess.Board()):
                    self.local[k]['reason'] = 'no_preceding_opponent_move'
        self.moments = {}

    def stop(self, k, move, kind):
        if kind in ('no_recorded_continuation', 'unresolved_distribution'):
            return unavailable('unidentified continuation or missing distribution')
        if move:
            return reply_rating(self.e.evidence.get(k), move)
        return self.local.get(k, unavailable())

    def continuation(self, k, incoming=None):
        """Own-turn stopping ratings depend on the exact incoming opponent edge."""
        if self.e.facts[k]['turn'] == self.e.color and not self.e.edges[k]:
            return incoming or self.local.get(k, unavailable())
        if k in self.moments: return self.moments[k]
        parts = [(p, self.stop(k, move, kind)) for move, p, kind, _, _ in self.e.stops[k]]
        for move, p, target in self.e.edges[k]:
            incoming = reply_rating(self.e.evidence.get(k), move) if self.e.facts[k]['turn'] != self.e.color else None
            parts.append((p, self.continuation(target, incoming)))
        if not math.isclose(sum(p for p, _ in parts), 1., abs_tol=1e-9):
            raise AssertionError('Stopping rating probability not conserved')
        self.moments[k] = mixture(parts, 'continuation stopping evidence')
        return self.moments[k]

    def score_evidence(self):
        return mixture([(p, self.continuation(k, self.initial.get(k, unavailable('no preceding opponent move'))))
                        for k, p in self.starts.items()], 'scope stopping evidence')

    def inventory(self):
        # A starting-board row is still an individual position. Keep its local
        # response average, without creating a whole-repertoire rating mean.
        positions = {k: dict(local=r, **({'continuation': self.continuation(k)}
                          if k != key(chess.Board()) else {})) for k, r in self.local.items()}
        stops, deviations = {}, defaultdict(list)
        for k, mass in self.reach.items():
            if mass <= 0: continue
            for move, p, kind, _, _ in self.e.stops[k]:
                r = self.stop(k, move, kind)
                stops[stop_key(dict(parent_position=k, move=move, type=kind))] = r
                if move:
                    target = self.e.facts[k]['after'][move][0]
                    deviations[target].append((mass * p, r, k, move))
        for target, parts in deviations.items():
            mass = sum(p for p, *_ in parts)
            r = mixture([(p, r) for p, r, _, _ in parts], 'incoming unprepared opponent moves',
                        origins=[dict(parent_position=k, move=m, weight=p / mass,
                                      mean=r['mean'], known_coverage=r['known_coverage']) for p, r, k, m in parts])
            positions[target] = dict(local=r, continuation=r)
        return dict(positions=positions, stops=stops)


def first_entries(evaluator, roots, entries):
    """Stop flow at its first entry and retain the last opponent move mixture."""
    evaluator.evaluate(roots)
    flow = dict.fromkeys(evaluator.values, 0.)
    incoming = defaultdict(list)
    for k, p in roots.items():
        flow[k] += p
        if k in entries:
            incoming[k].append((p, unavailable('entry is a PGN root'), None, None))
    for k in reversed(evaluator.values):
        if k in entries or flow[k] <= 0: continue
        for move, p, target in evaluator.edges[k]:
            mass = flow[k] * p
            flow[target] += mass
            if target in entries:
                r = (reply_rating(evaluator.evidence.get(k), move)
                     if evaluator.facts[k]['turn'] != evaluator.color
                     else response_rating(evaluator.evidence.get(target)))
                incoming[target].append((mass, r, k, move))
    entry_mass = sum(flow.get(k, 0) for k in entries)
    weights = {k: flow.get(k, 0) / entry_mass for k in entries if flow.get(k, 0) > 0} if entry_mass else {}
    initial = {}
    for k in entries:
        if evaluator.facts[k]['turn'] != evaluator.color:
            initial[k] = response_rating(evaluator.evidence.get(k))
        else:
            parts = incoming[k]
            mass = sum(p for p, *_ in parts)
            initial[k] = mixture([(p, r) for p, r, _, _ in parts], 'first-entry incoming opponent moves',
                origins=[dict(parent_position=parent, move=move, weight=p / mass, mean=r['mean'],
                              known_coverage=r['known_coverage']) for p, r, parent, move in parts if p > 0])
    return weights, initial


def move_context(evidence, color, position, move):
    if chess.Board(position + ' 0 1').turn != color:
        return reply_rating(evidence.get(position), move)
    board = chess.Board(position + ' 0 1'); board.push_uci(move)
    return response_rating(evidence.get(key(board)))


def comparison_fields(reply, parent):
    """Compare the opponent move maker with the same player's parent cohort."""
    available = reply.get('mean') is not None and parent.get('mean') is not None
    return dict(parent_mean=parent.get('mean'), parent_known_coverage=parent['known_coverage'],
                difference_vs_parent=reply['mean'] - parent['mean'] if available else None,
                comparison_coverage=reply['known_coverage'] if available else 0.)


def comparison_mixture(parts):
    """Paired differences use arrival flow; both cohorts must be available."""
    parts = list(parts)
    total = sum(p for p, _ in parts)
    paired = sum(p * r['comparison_coverage'] for p, r in parts)
    def average(field):
        return (sum(p * r['comparison_coverage'] * r[field] for p, r in parts
                    if r['comparison_coverage'] > 0) / paired) if paired else None
    return dict(parent_mean=average('parent_mean'),
                parent_known_coverage=average('parent_known_coverage') if paired else 0.,
                difference_vs_parent=average('difference_vs_parent'),
                comparison_coverage=paired / total if total else 0.)


def add_reply_differences(result, evidence):
    """Enrich individual replies using cached parents without changing ratings."""
    parents = {}
    color = result['color'] == 'white'

    def direct(parent, move):
        if parent is None:
            return comparison_fields(unavailable(), unavailable())
        if parent not in parents: parents[parent] = response_rating(evidence.get(parent))
        return comparison_fields(reply_rating(evidence.get(parent), move), parents[parent])

    def arrivals(origins):
        return comparison_mixture([(origin['weight'],
            arrivals(origin['first_entry_origins']) if origin.get('first_entry_origins')
            else direct(origin['parent_position'], origin['move'])) for origin in origins])

    for scope in result['scopes']:
        for position, fields in scope.get('positions', {}).items():
            if chess.Board(position + ' 0 1').turn == color:
                # Copy so shared stopping/chapter means are not modified.
                fields['local'] = dict(fields['local'], **arrivals(fields['local'].get('origins', [])))
        for identity, r in list(scope.get('moves', {}).items()):
            position, move = identity.rsplit('|', 1)
            if chess.Board(position + ' 0 1').turn != color:
                scope['moves'][identity] = dict(r, **direct(position, move))
        for identity, r in list(scope.get('stops', {}).items()):
            parent, move, kind = identity.rsplit('|', 2)
            if move:
                scope['stops'][identity] = dict(r, **direct(parent, move))
            elif kind == 'theory_leaf' and chess.Board(parent + ' 0 1').turn == color:
                local = scope.get('positions', {}).get(parent, {}).get('local', {})
                scope['stops'][identity] = dict(r, **arrivals(local.get('origins', [])))
    result['manifest']['reply_difference_basis'] = (
        'Reply move-maker mean minus parent game-weighted opponent response mean; '
        'transposed arrivals mix paired differences by policy reach. No score adjustment.')
    result['validation']['reply_differences_use_cached_parents'] = True
    return result


def analyze(path, cache='.cache/explorer'):
    path = Path(path)
    score_bytes = path.read_bytes(); saved = json.loads(score_bytes)
    m = saved['manifest']; source = Path(m['input_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != m['input_sha256']:
        raise ValueError('PGN changed since scoring; regenerate scores first')
    supporting, hashes = {}, {}
    for family in ('preparation', 'character', 'vulnerabilities'):
        p = path.with_suffix(f'.{family}.json')
        data_bytes = p.read_bytes(); data = json.loads(data_bytes)
        if data['color'] != saved['color'] or any(data['manifest'].get(k) != expected for k, expected in (
                ('report_sha256', hashlib.sha256(score_bytes).hexdigest()),
                ('input_sha256', m['input_sha256']), ('filters', m['filters']))):
            raise ValueError(f'{family} belongs to a different score snapshot')
        supporting[family] = data; hashes[family] = hashlib.sha256(data_bytes).hexdigest()
    graph = parse(source, m['configuration'].get('exclude', []))
    color = saved['color'] == 'white'; evidence, missing = {}, []
    explorer = Explorer(cache, m['filters'], offline=True)
    try:
        for k in graph.nodes:
            try: evidence[k] = explorer.get(k)
            except ValueError as exc:
                if not str(exc).startswith('Offline cache miss:'): raise
                missing.append(k)
            if k in m['evidence'] and explorer.provenance.get(k) != m['evidence'][k]:
                raise ValueError('Saved score evidence changed; regenerate scores first')
        # Only read existing cached alternatives. A miss remains unavailable.
        for scope in [supporting['vulnerabilities']['overall'], *supporting['vulnerabilities']['chapters']]:
            for row in scope.get('all_signed_rows', []):
                alt = row.get('alternative')
                if row['kind'] != 'own' or not alt: continue
                board = chess.Board(row['position'] + ' 0 1'); board.push_uci(alt['move'])
                k = key(board)
                if k in evidence or k in missing: continue
                try: evidence[k] = explorer.get(k)
                except ValueError as exc:
                    if not str(exc).startswith('Offline cache miss:'): raise
                    missing.append(k)
    finally:
        explorer.close()
    facts = chess_facts(graph, color, evidence)
    policy = m['configuration'].get('policy', {}); roots = m['root_weights']
    prep_scopes = {s['id']: s for s in supporting['preparation']['scopes']}
    chapters = {c['id']: c for c in saved['chapters']}
    output = []

    def evaluator(chapter):
        return Evaluator(graph, color, evidence, facts, policy, chapter=chapter, sparse=m['sparse_threshold'])

    for sid, prep in prep_scopes.items():
        cid = None if sid == 'overall' else sid
        e = evaluator(cid)
        initial, entry_weights = {}, {}
        if cid:
            c = chapters[cid]
            entry_weights, initial = first_entries(e, roots, {r['position'] for r in c['entries']})
            if c['score'].get('entry_probability') is not None:
                # The model's saved normalized entry weights are authoritative.
                expected = {k: p for k, p in prep['starts'].items() if p > 0}
                if entry_weights.keys() != expected.keys() or any(not math.isclose(p, entry_weights[k], abs_tol=1e-9) for k, p in expected.items()):
                    raise AssertionError('Rating first-entry flow differs from saved chapter weights')
        scope = dict(id=sid, policy_basis=prep['policy_basis'], positions={}, stops={}, moves={})
        if not prep['starts']:
            scope['status'] = 'unresolved entry weights'; output.append(scope); continue
        context = Context(e, prep['starts'], initial)
        value = e.evaluate(prep['starts'])
        expected = saved['overall'] if not cid else chapters[cid]['score']
        if not math.isclose(value[0], expected['resolved_contribution'], abs_tol=1e-10) or not math.isclose(value[1], expected['unresolved_mass'], abs_tol=1e-10):
            raise AssertionError('Rating context did not reproduce saved score')
        scope.update(context.inventory()); scope['status'] = 'available'
        if cid:
            scope['score_evidence'] = context.score_evidence()
            scope['entry_baseline'] = mixture([(p, initial[k]) for k, p in prep['starts'].items()], 'weighted local first-entry ratings')
            scope['entries'] = initial
        for k, mass in context.reach.items():
            if mass <= 0: continue
            for row in evidence.get(k, {}).get('moves', []):
                scope['moves'][k + '|' + row['uci']] = move_context(evidence, color, k, row['uci'])
            for move, _, _ in e.edges[k]:
                scope['moves'].setdefault(k + '|' + move, move_context(evidence, color, k, move))
        # Independent forward enumeration verifies that deeper paths get no extra weight.
        enumerated = mixture([(context.reach[k] * p, context.stop(k, move, kind))
            for k in context.reach for move, p, kind, _, _ in e.stops[k]], 'enumerated stopping evidence')
        moment = context.score_evidence()
        for field in ('known_coverage', 'weighted_rating_sum'):
            if not math.isclose(enumerated[field], moment[field], abs_tol=1e-8):
                raise AssertionError(f'Forward and backward stopping rating moments differ: {sid} {field}: {enumerated[field]} vs {moment[field]}')
        # Chapter pawn groups may summarize stopping cohorts; overall groups may not.
        if cid:
            from .character import board_fingerprint
            groups = defaultdict(list)
            for row in prep['stops']:
                value = board_fingerprint(row['position'], color)[0]['pawn_structure']
                groups[value].append((row['reach'], scope['stops'][stop_key(row)]))
            scope['pawn_groups'] = {k: mixture(parts, 'chapter pawn-structure stopping evidence') for k, parts in groups.items()}
        output.append(scope)

    if hashlib.sha256(source.read_bytes()).hexdigest() != m['input_sha256'] or path.read_bytes() != score_bytes:
        raise ValueError('Source or saved score changed during rating analysis')
    return add_reply_differences(dict(color=saved['color'], scopes=output, manifest=dict(schema_version=SCHEMA_VERSION, 
        created_at=datetime.now(timezone.utc).isoformat(), report_sha256=hashlib.sha256(score_bytes).hexdigest(),
        input_sha256=m['input_sha256'], input_path=str(source), filters=m['filters'],
        supporting_sha256=hashes, evidence=explorer.provenance,
        cache_sha256={k: hashlib.sha256((Path(cache) / (p['cache_key'] + '.json')).read_bytes()).hexdigest()
                      for k, p in explorer.provenance.items()},
        cache_only=True, network_requests=0, uncached_positions=missing),
        validation=dict(scores_reproduced=True, first_entry_weights_reproduced=True,
                        stopping_rating_moments_reproduced=True, source_pgn_unchanged=True,
                        no_repertoire_rating_averages=True, starting_position_local_context_retained=True)), evidence)


def attach(bundle):
    """Attach display-only contexts to loaded rows without rewriting analyses."""
    scopes = {s['id']: s for s in bundle['ratings']['scopes']}

    def position(row, scope, field='position'):
        row['opponent_rating'] = scope.get('positions', {}).get(row.get(field), {}).get('local')

    def stops(row, scope):
        identity = dict(row)
        if identity.get('type') in ('leaf', 'terminal'):
            identity['type'] = 'theory_leaf'
        row['opponent_rating'] = scope.get('stops', {}).get(stop_key(identity))

    for family in ('character', 'preparation', 'vulnerabilities'):
        data = bundle.get(family, {})
        items = data.get('scopes', []) if family != 'vulnerabilities' else [data.get('overall', {}), *data.get('chapters', [])]
        for item in items:
            sid = item.get('id', 'overall'); scope = scopes.get(sid, {})
            if family == 'character':
                for row in item.get('positions', []): position(row, scope)
                for row in item.get('reuse', {}).get('decisions', []): position(row, scope)
                for row in item.get('predictability', {}).get('positions', []):
                    position(row, scope)
                    for reply in row.get('replies', []):
                        reply['opponent_rating'] = scope.get('moves', {}).get(row['position'] + '|' + reply['move'])
                for profile in item.get('position_profiles', {}).values():
                    for rows in profile.get('distributions', {}).values():
                        for row in rows: position(row, scope, 'example_position')
                    if sid != 'overall':
                        for row in profile.get('distributions', {}).get('pawn_structure', []):
                            row['group_opponent_rating'] = scope.get('pawn_groups', {}).get(row['value'])
            elif family == 'preparation':
                for row in item.get('stops', []): stops(row, scope)
            else:
                rows = list(item.get('all_signed_rows', [])) + list(item.get('strengths', []))
                for more in item.get('rankings', {}).values(): rows.extend(more)
                for row in rows:
                    row['opponent_rating'] = scope.get('moves', {}).get(row['position'] + '|' + row['move'])
                    if row.get('alternative'):
                        alt = row['alternative']
                        alt['opponent_rating'] = scope.get('moves', {}).get(row['position'] + '|' + alt['move'])
    for row in bundle['report']['events']: stops(row, scopes.get('overall', {}))
    for chapter in bundle['report']['chapters']:
        scope = scopes.get(chapter['id'], {})
        chapter['opponent_ratings'] = {k: scope.get(k) for k in ('score_evidence', 'entry_baseline')}
        for row in chapter['entries']:
            row['opponent_rating'] = scope.get('entries', {}).get(row['position'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+')
    parser.add_argument('--cache', default='.cache/explorer')
    args = parser.parse_args()
    try:
        for value in args.reports:
            path = Path(value)
            result = analyze(path, args.cache)
            path.with_suffix('.ratings.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
            from .render import update_report_outputs
            update_report_outputs(path)
            print(f'{result["color"]}: opponent rating ledger generated; network requests: 0', flush=True)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'Rating analysis failed: {exc}\n')


if __name__ == '__main__':
    main()
