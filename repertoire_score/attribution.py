"""Chapter provenance for displayed moves, positions and PGN edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import chess

from .graph import key, parse


ATTRIBUTION_NOTE = ('Chapter attribution follows the exact position or final recorded move, so a representative route can combine chapters. '
                    'Shared sources list every contributing chapter. Unprepared replies name their parent context; '
                    'transpositions name the chapters containing the reached position.')


def write_text(path, text):
    """Replace complete outputs atomically, tolerating brief Windows sync locks."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     prefix='.chapter-attribution-',suffix='.tmp',delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        for attempt in range(8):
            try:
                os.replace(temporary,path)
                break
            except OSError:
                if attempt == 7: raise
                time.sleep(min(.1*2**attempt,1.0))
    finally:
        temporary.unlink(missing_ok=True)


class Attribution:
    def __init__(self, graph):
        self.graph = graph
        self.catalog = [{k:c.get(k) for k in ('id','name','url')} for c in graph.chapters]
        self.order = {c['id']:i for i,c in enumerate(graph.chapters)}
        self.cache = {}

    def ordered(self, ids):
        return sorted(set(ids), key=lambda cid:(self.order.get(cid,len(self.order)),cid))

    def position_or_move(self, position, move=None):
        position = ' '.join(position.split()[:4])
        identity = position,move
        if identity in self.cache: return self.cache[identity]
        node = self.graph.nodes.get(position)
        context = self.ordered(node.chapters) if node else []
        sources, transpositions = [], []
        if move is None:
            sources, basis = context, 'position'
        elif node and node.provenance.get(move):
            sources, basis = self.ordered(node.provenance[move]), 'recorded move'
        else:
            basis = 'unprepared move'
            if node:
                board = chess.Board(node.fen)
                board.push_uci(move)
                target = self.graph.nodes.get(key(board))
                if target:
                    transpositions, basis = self.ordered(target.chapters), 'transposition'
        result = dict(source_ids=sources, context_ids=context, transposition_ids=transpositions, basis=basis)
        self.cache[identity] = result
        return result


def enrich(report, graph):
    """Add provenance only; leave every score, probability and ordering unchanged."""
    attribution = Attribution(graph)
    report['chapter_catalog'] = attribution.catalog
    if 'events' in report:
        for row in report['events']:
            row['chapter_attribution'] = attribution.position_or_move(row['parent_position'],row.get('move'))
        for chapter in report['chapters']:
            for entry in chapter['entries']:
                entry['chapter_attribution'] = attribution.position_or_move(entry['position'])
    elif 'proposals' in report:
        for scope in report['scopes']:
            for row in scope['stops']:
                row['chapter_attribution'] = attribution.position_or_move(row['parent_position'],row.get('move'))
        for proposal in report['proposals']:
            proposal['chapter_attribution'] = dict(source_ids=attribution.ordered(e['chapter_id'] for e in proposal['edits']),
                context_ids=[],transposition_ids=[],basis='PGN subtree edits')
    elif 'scopes' in report:
        for scope in report['scopes']:
            if 'reuse' not in scope: continue
            for row in scope['reuse']['decisions']:
                row['chapter_attribution'] = attribution.position_or_move(row['position'],row['move'])
            for row in scope['predictability']['positions']:
                row['chapter_attribution'] = attribution.position_or_move(row['position'])
                for reply in row['replies']:
                    reply['chapter_attribution'] = attribution.position_or_move(row['position'],reply['move'])
            examples = {}
            for row in scope['stopping_outcomes']:
                row['chapter_attribution'] = attribution.position_or_move(row['parent_position'],row.get('move'))
                examples[row['position'],row['line']] = row['chapter_attribution']
            for profile in scope['position_profiles'].values():
                for distribution in profile['distributions'].values():
                    for row in distribution:
                        row['example_chapter_attribution'] = examples[row['example_position'],row['example_line']]
    elif 'nonoverlapping_change_contributions' in report:
        for row in report['nonoverlapping_change_contributions']:
            row['chapter_attribution'] = attribution.position_or_move(row['position'])
    else:
        def visit(item):
            if isinstance(item,list):
                for child in item: visit(child)
            elif isinstance(item,dict):
                if item.get('kind') in ('own','opponent') and 'position' in item and 'move' in item:
                    item['chapter_attribution'] = attribution.position_or_move(item['position'],item['move'])
                    if item.get('alternative'):
                        alt = item['alternative']
                        alt['chapter_attribution'] = attribution.position_or_move(item['position'],alt['move'])
                for field,child in list(item.items()):
                    if field != 'chapter_attribution': visit(child)
        visit(report.get('overall',{}))
        visit(report.get('chapters',[]))
    return report


def chapter_text(row, catalog):
    attribution = row.get('chapter_attribution')
    if attribution is None: return 'Not attributed'
    lookup = {c['id']:c for c in catalog}
    def names(ids):
        result = []
        for cid in ids:
            chapter = lookup.get(cid,{'name':cid})
            label = chapter['name'].replace('|','&#124;').replace('[','&#91;').replace(']','&#93;').replace('\n',' ')
            result.append(f"[{label}]({chapter['url']})" if chapter.get('url') else label)
        return '; '.join(result) or 'None'
    if attribution['source_ids']: return names(attribution['source_ids'])
    if attribution['transposition_ids']: return 'Transposition into: '+names(attribution['transposition_ids'])
    if attribution['basis'] == 'unprepared move':
        return 'Unprepared'+('; parent: '+names(attribution['context_ids']) if attribution['context_ids'] else '')
    return 'None'


def vulnerability_summary(results, top=10):
    from .vulnerabilities import pct,delta,opponent_tables,table
    text = ['# Repertoire vulnerability summary', '',
            'Largest weighted deficits, separately for opponent replies and our moves. The benchmarks differ and the rows overlap, '
            'so do not sum them or interpret drag as an achievable improvement. See each full report for definitions, sample caveats, '
            'all chapters and exact evidence.', '', ATTRIBUTION_NOTE, '']
    for report,path in results:
        catalog = report['chapter_catalog']
        text += [f"## {report['color'].title()}", '',
                 f"Repertoire score: {pct(report['overall_score'])}; {report['color']} starting baseline: "
                 f"{pct(report['starting_baseline_score'])}; difference: {delta(report['overall_delta_pp'])}.", '',
                 f"[Full overall and chapter report]({path.with_suffix('.vulnerabilities.md').name})", '']
        text += opponent_tables(report['overall']['rankings']['opponent'],top,catalog=catalog)
        text += ['### Our selected moves','']+table(report['overall']['rankings']['own'],top,own=True,catalog=catalog)
    return '\n'.join(text)


def improvement_markdown(report, text):
    """Add source cells to the established improvement report without changing its narrative."""
    changes = {r['representative_line']:r for r in report['nonoverlapping_change_contributions']}
    output,had_column,in_table = [],False,False
    for line in text.splitlines():
        if line.startswith('| First changed continuation |'):
            had_column = 'Source chapters' in line
            in_table = True
            if not had_column: line = line.replace('| First changed continuation |','| First changed continuation | Source chapters |',1)
        elif in_table and line.startswith('|---'):
            if not had_column: line = '|---'+line
        elif in_table and line.startswith('|'):
            cells = line.split('|')
            row = changes.get(cells[1].strip())
            if row:
                value = ' '+chapter_text(row,report['chapter_catalog'])+' '
                if had_column: cells[2] = value
                else: cells.insert(2,value)
                line = '|'.join(cells)
        else:
            in_table = False
        output.append(line)
    return '\n'.join(output)+'\n'


def regenerate(paths):
    """Refresh saved attribution and Markdown without rerunning any analysis."""
    from .render import render_reports
    from .vulnerabilities import markdown as vulnerability_markdown
    from .preparation import render as preparation_render, summary as preparation_summary
    from .character import render as character_render, summary as character_summary
    paths = [Path(p) for p in paths]
    stages = []
    correlation_updates,improvement_updates = {},[]
    # Validate every requested source and companion before any write.
    for path in paths:
        report = json.loads(path.read_text(encoding='utf-8'))
        manifest = report['manifest']
        source = Path(manifest['input_path'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
            raise ValueError(f'PGN changed since scoring; regenerate scores first: {source}')
        graph = parse(source,manifest['configuration'].get('exclude',[]))
        companions = []
        old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        correlation_path = path.parent/'depth-delta-correlation.json'
        if correlation_path.exists():
            correlation = correlation_updates.setdefault(correlation_path,json.loads(correlation_path.read_text(encoding='utf-8')))
            provenance = correlation.get('provenance',{}).get(report['color'],{})
            if provenance.get('report_path') == str(path.resolve()) and provenance.get('report_sha256') == old_hash:
                provenance['_refresh_path'] = str(path)
        improvement_path = path.parent/f"{report['color']}-improvements.json"
        if improvement_path.exists():
            improvement = json.loads(improvement_path.read_text(encoding='utf-8'))
            if improvement.get('input_sha256') == manifest['input_sha256']:
                improvement_updates.append((improvement_path,enrich(improvement,graph)))
        for kind in ('vulnerabilities','preparation','character'):
            companion_path = path.with_suffix(f'.{kind}.json')
            if not companion_path.exists(): continue
            companion = json.loads(companion_path.read_text(encoding='utf-8'))
            if companion['manifest']['input_sha256'] != manifest['input_sha256'] or companion['manifest']['report_sha256'] != old_hash:
                raise ValueError(f'Saved companion differs from source scores: {companion_path}')
            companions.append((kind,companion_path,enrich(companion,graph)))
        stages.append((path,enrich(report,graph),companions))
    vulnerability_results,preparation_results,character_results = [],[],[]
    for path,report,companions in stages:
        data = json.dumps(report,indent=2,allow_nan=False)
        write_text(path,data)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for kind,companion_path,result in companions:
            result['manifest']['report_sha256'] = digest
            write_text(companion_path,json.dumps(result,indent=2,allow_nan=False))
            renderer = {'vulnerabilities':vulnerability_markdown,'preparation':preparation_render,'character':character_render}[kind]
            write_text(companion_path.with_suffix('.md'),renderer(result))
            if kind == 'vulnerabilities': vulnerability_results.append((result,path))
            elif kind == 'preparation': preparation_results.append((result,path))
            else: character_results.append(result)
        print(f"Updated {report['color']} chapter attribution: score report and {len(companions)} companion reports",flush=True)
    directory = paths[0].parent
    render_reports(paths,directory/'summary.md')
    if vulnerability_results: write_text(directory/'vulnerabilities.md',vulnerability_summary(vulnerability_results))
    if preparation_results: write_text(directory/'preparation.md',preparation_summary(preparation_results))
    if character_results: write_text(directory/'character.md',character_summary(character_results))
    for path,correlation in correlation_updates.items():
        changed = False
        for provenance in correlation.get('provenance',{}).values():
            source = provenance.pop('_refresh_path',None)
            if source:
                provenance['report_sha256'] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
                changed = True
        if changed: write_text(path,json.dumps(correlation,indent=2,allow_nan=False))
    for path,report in improvement_updates:
        write_text(path,json.dumps(report,indent=2,allow_nan=False))
        markdown_path = path.with_suffix('.md')
        if markdown_path.exists():
            write_text(markdown_path,improvement_markdown(report,markdown_path.read_text(encoding='utf-8')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+',type=Path,help='Saved score reports; refresh their existing companions too')
    args = parser.parse_args()
    regenerate(args.reports)


if __name__ == '__main__': main()
