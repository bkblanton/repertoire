"""JSON-friendly evidence ledger and Markdown reports."""
import numpy as np
import chess
from .model import score
from .explorer import counts


def starting_position_reference(data, color, provenance):
    results = counts(data)
    return {
        "position": chess.STARTING_FEN,
        "sample_count": sum(results),
        "counts_white_draw_black": results,
        "white_score": score(results, chess.WHITE),
        "black_score": score(results, chess.BLACK),
        "owner_score": score(results, color),
        "basis": "Empirical standard starting-position score under the same Explorer filters, without forcing repertoire moves",
        "provenance": provenance,
    }


def reference_markdown(reference, color):
    return [
        "Starting-position reference under the same database filters, before forcing any repertoire moves:", "",
        "| Perspective | Starting-position score |",
        "|---|---:|",
        f"| {color.title()} | {pct(reference['owner_score'])} |", "",
        f"Based on {reference['sample_count']:,} games, counting a draw as half a point. This is a population reference, not a causal estimate of improvement from the repertoire.", "",
    ]


def events(graph, model, raw_sample, post_sample, raw_flow, post_flow, color, prior_strength):
    result = []
    for (k, j), mass in raw_flow.items():
        b = model[k].branches[j]
        raw_score = raw_sample[k][j][1]
        draws = post_sample[k][j][1]
        pmass = post_flow[(k, j)]
        sample = sum(b.counts)
        unresolved = b.fixed_score is None and sample == 0
        n = graph.nodes[k]
        board = chess.Board(n.fen)
        path = list(n.path)
        if b.move:
            move = chess.Move.from_uci(b.move)
            path.append(board.san(move))
            board.push(move)
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
        result.append({"parent_position": k, "position": board.fen(en_passant="legal"), "move": b.move,
                       "representative_path_san": path, "chapters": sorted(n.chapters), "type": b.kind,
                       "unresolved": unresolved, "score_status": "prior-only; no direct observations" if unresolved else "deterministic" if b.fixed_score is not None else "observed",
                       "sample_count": sample, "counts_white_draw_black": b.counts,
                       "probability": float(mass[0]), "posterior_probability_mean": float(pmass.mean()),
                       "raw_score": raw_score, "posterior_score_mean": float(np.mean(draws)),
                       "posterior_score_interval_95": interval,
                       "contribution": float(mass[0]*raw_score) if raw_score is not None else None,
                       "posterior_contribution_mean": float(np.mean(pmass*draws)),
                       "uncertainty_priority": float(pmass.mean())*(interval[1]-interval[0]),
                       "contribution_sd": float(np.std(pmass*draws)),
                       "prior_fraction": 0 if b.fixed_score is not None else
                       (prior_strength/len(model[k].branches) if model[k].mode == "opponent" else prior_strength)/
                       (sample+(prior_strength/len(model[k].branches) if model[k].mode == "opponent" else prior_strength)),
                       "evidence": "deterministic chess outcome" if b.fixed_score is not None else
                       "parent move-result table" if model[k].mode == "opponent" else
                       "no usable opponent distribution" if b.kind == "unresolved_distribution" else "position result counts"})
    return result


def pct(v):
    return "undefined" if v is None else f"{100*v:.3f}%"


def bounds(v):
    return f"{100*v[0]:.3f}% to {100*v[1]:.3f}%"


def overall_delta(report, precision=3):
    score = report['overall'].get('raw_empirical_score')
    baseline = report.get('starting_position_reference', {}).get('owner_score')
    if score is None or baseline is None:
        return 'undefined'
    return f'{100*(score-baseline):+.{precision}f}'


def depth_text(summary):
    depth = summary.get('prepared_depth')
    if depth is None:
        return 'not calculated'
    if depth['expected_moves'] is not None:
        return f"{depth['expected_moves']:.2f}"
    if 'conditional_bounds' in depth:
        low, high = depth['conditional_bounds']
        return f'unresolved ({low:.2f} to {high:.2f})'
    return 'undefined'


def chapter_table(chapters, concise=False):
    lines = [
        "Entry probability is chapter reach: the chance of first reaching any qualifying position, counting each modeled game once across all entry routes. Repertoire score is conditional on that entry. Entry baseline uses the same first-entry weights. Delta is repertoire score minus entry baseline, in percentage points (pp). Chapters can overlap; their scores and entry probabilities are not additive.", "",
        "Expected prepared depth is the number of your remaining prepared moves, averaged over first entries and opponent replies. It includes an available move at entry and preparation supplied by other chapters after transpositions; entry itself earns no extra move.", "",
        "| Chapter | Entry probability | Entry baseline | Repertoire score | Delta (pp) | Expected prepared depth (moves) |" + ("" if concise else " Approx. 95% interval | Unresolved | Sparse sensitivity |"),
        "|---|---:|---:|---:|---:|---:|" + ("" if concise else "---|---:|---|"),
    ]
    for c in chapters:
        s = c["score"]
        baseline = c.get("entry_baseline", {})
        difference = baseline.get("difference_pp")
        delta = "undefined" if difference is None else f"{difference:+.3f}"
        prefix = f"| {c['name']} | {pct(s.get('entry_probability'))} | {pct(baseline.get('raw_score'))} |"
        row = f"{prefix} {pct(s.get('raw_empirical_score'))} | {delta} | {depth_text(s)} |"
        if not concise:
            row += (f" {bounds(s['posterior']['credible_interval_95'])} | {pct(s['unresolved_mass'])} | {bounds(s['sparse_sensitivity'])} |"
                    if "posterior" in s else " | | |")
        lines.append(row)
    return lines


def markdown(report):
    overall = report["overall"]
    filters = report["manifest"]["filters"]
    lines = [f"# {report['color'].title()} repertoire score", "", 
             f"Population: rated {filters.get('speeds', 'configured speeds')} games in the Lichess Opening Explorer. Rating groups: {filters.get('ratings', 'configured ratings')}. Date filter: {filters.get('since', 'start')} through {filters.get('until', 'end')}.", "",
             "Scores are from the repertoire owner's perspective. Probabilities are modeled repertoire probabilities, not observed historical sequence frequencies.", "",
             f"Repertoire score: **{pct(overall['raw_empirical_score'])}**. Conditional bounds with unresolved evidence: **{bounds(overall['conditional_bounds'])}**.",
             f"Overall delta versus the {report['color'].title()} starting-position baseline: **{overall_delta(report)} pp** (repertoire score minus baseline).",
             f"Expected prepared depth from the repertoire root: **{depth_text(overall)} of your moves**. No depth cutoff or discount is used.",
             f"Approximate model-based 95% credible interval: **{bounds(overall['posterior']['credible_interval_95'])}**.",
             f"Posterior interpretation: {overall['posterior']['label']}.", "",
             "The posterior point estimate and interval explicitly complete missing scores using the configured prior. Missing opponent distributions stop unresolved, without uniform move assumptions. Conservative bounds retain all missing evidence. Intervals do not include unknown overlap among games at different positions, population mismatch, or repertoire-selection bias.", "",
             f"Empirical stopping mass: theory leaves {pct(overall['masses']['theory_leaf'])}; deviations {pct(overall['masses']['deviation'])}; other stops {pct(overall['masses']['other_stop'])}; unresolved {pct(overall['unresolved_mass'])}.",
             f"Posterior mean unresolved mass: {pct(overall['posterior']['unresolved_mass_mean'])}. Posterior mean conditional score bounds: {bounds(overall['posterior']['conditional_bounds_mean'])}. These bounds are distinct from the credible interval.",
             f"Sparse threshold: fewer than {report['manifest']['sparse_threshold']} observations. Sparse mass: {pct(overall['sparse_mass'])}; sparse-score sensitivity: {bounds(overall['sparse_sensitivity'])}.", "",
             "## Chapters", "",
             "Entry probability is separate from score conditional on first entry. Complete merged theory applies after entry. Chapters may overlap; do not add or average these rows to derive the overall score.", "",
             *chapter_table(report["chapters"])]
    if report.get('chapter_transitions'):
        lines += ['', '## Chapter transitions', '',
                  'Conditional probability of reaching the destination at or after first entering the source chapter, before the model stops. Shared or simultaneous entry counts; this does not require a later move or imply that the source chapter has ended. Rows overlap and must not be added. Zero-probability and unresolved source rows remain in JSON.', '',
                  '| Source chapter | Destination chapter | Conditional probability | Joint probability from repertoire root |',
                  '|---|---|---:|---:|']
        for row in report['chapter_transitions']:
            probability = row['conditional_probability']
            if probability is not None and probability > 0:
                lines.append(f"| {row['source_name']} | {row['destination_name']} | {pct(probability)} | {pct(row['joint_probability'])} |")
            elif row.get('status') == 'unresolved_destination_reach':
                lines.append(f"| {row['source_name']} | {row['destination_name']} | unresolved ({bounds(row['conditional_bounds'])}) | undefined |")
    lines += ["", "## Exact chapter entries", "",
              'Paths below are representative move orders for exact positions, not an exhaustive route list. First-entry weights exclude paths that already entered the chapter; a later position can still be an entry through another route.', '']
    for c in report["chapters"]:
        lines += [f"### {c['name']}", "", f"Entry definition: {c['entry_status']}", ""]
        if c.get('region'):
            region = c['region']
            lines += [f"Subject definition: {region['description']}. Region: {len(region['positions'])} exact positions. Anchors:", '']
            lines += [f"- `{anchor}`" for anchor in region['anchors']]
            lines += ['', 'Possible first-entry positions:', '']
        for e in c["entries"]:
            lines.append(f"- {' '.join(e['path']) or '(PGN root)'}; first-entry weight {pct(e.get('conditional_first_entry_weight'))}; canonical position `{e['position']}`")
    lines += ["", "## Largest stopping contributions", "", "| Type | Representative SAN path | Probability | Contribution | Observations |", "|---|---|---:|---:|---:|"]
    for e in sorted(report["events"], key=lambda e:e["posterior_contribution_mean"], reverse=True)[:15]:
        lines.append(f"| {e['type']} | {' '.join(e['representative_path_san'])} | {pct(e['probability'])} | {pct(e['posterior_contribution_mean'])} | {e['sample_count']} |")
    lines += ["", "## Largest uncertainty priorities", "", "Priority is posterior mean reach times stopping-score interval width. It is a review heuristic, not additive variance attribution.", "", "| Type | Representative SAN path | Priority (score points) | Observations |", "|---|---|---:|---:|"]
    for e in sorted(report["events"], key=lambda e:e["uncertainty_priority"], reverse=True)[:15]:
        lines.append(f"| {e['type']} | {' '.join(e['representative_path_san'])} | {100*e['uncertainty_priority']:.4f} | {e['sample_count']} |")
    lines += ["", "## Prior sensitivity", "", "The total joint-table prior strength is controlled per opponent position, divided over legal moves and any observed residual bucket.", ""]
    for p in report["prior_sensitivity"]:
        lines.append(f"- Prior {p['prior']}: posterior mean {pct(p['overall']['posterior']['mean'])}; interval {bounds(p['overall']['posterior']['credible_interval_95'])}.")
    lines += ["", "## Run details", "", f"- Input SHA-256: `{report['manifest']['input_sha256']}`",
              f"- Random seed: {report['manifest']['seed']}; simulations: {report['manifest']['simulations']}; prior owner W/D/L: {report['manifest']['prior']}.",
              f"- Own-move policy: {report['manifest']['configuration'].get('policy_description', 'explicit configuration in the companion JSON' if report['manifest']['configuration'].get('policy') else 'unambiguous PGN moves')}.",
              f"- Distinct graph positions: {report['manifest']['positions']}; evaluated positions: {report['manifest']['evaluated_positions']}.",
              "- Full stopping-event ledger, sample counts, prior influence, cache keys and retrieval timestamps are in the companion JSON.",
              "- Sanity checks: node and terminal probability conservation; weighted terminal contributions reproduce the root value.", ""]
    if "starting_position_reference" in report:
        lines[6:6] = reference_markdown(report["starting_position_reference"], report['color'])
    return "\n".join(lines)


def population_text(report):
    filters = report['manifest']['filters']
    speeds = filters.get('speeds', 'configured speeds').replace(',', ', ')
    ratings = filters.get('ratings', '')
    bands = 'all ratings' if ratings == '0,1000,1200,1400,1600,1800,2000,2200,2500' else f"rating groups {ratings or 'as configured'}"
    since, until = filters.get('since'), filters.get('until')
    dates = 'all available dates' if since == '1952-01' and until == '3000-12' else f"{since or 'start'} to {until or 'end'}"
    return f"Lichess rated {speeds}; {bands}; {dates}."


def evidence_date(report):
    dates = sorted({p['retrieved_at'][:10] for p in report['manifest'].get('evidence', {}).values()
                    if p.get('retrieved_at', '')[:4].isdigit()})
    return (dates[0] if dates[0] == dates[-1] else f'{dates[0]} to {dates[-1]}') if dates else report['manifest'].get('created_at', '')[:10]


def study_description(report):
    """Plain text with simple bullets, suitable for a Lichess study description."""
    def number(value):
        return 'unresolved' if value is None else f'{100*value:.2f}%'
    overall = report['overall']
    reference = report.get('starting_position_reference', {}).get('owner_score')
    lines = [f"{report['color'].title()} repertoire", '',
             f"Repertoire score: {number(overall['raw_empirical_score'])}. Starting-position reference: {number(reference)}. Overall delta: {overall_delta(report, 2)} pp.",
             f"Expected prepared depth from the repertoire root: {depth_text(overall)} of my moves.",
             population_text(report), f"Database snapshot: {evidence_date(report)}.", '',
             "Scores count wins as 1 and draws as 0.5. The model assumes I play my prepared moves and opponents follow database move frequencies, stopping at the end of preparation or the first deviation.", '',
             "Chapter results: entry probability / repertoire score / entry baseline / difference in percentage points (pp) / expected prepared depth in my moves.", '']
    for c in report['chapters']:
        baseline = c.get('entry_baseline', {})
        delta = baseline.get('difference_pp')
        difference = 'unresolved' if delta is None else f'{delta:+.2f} pp'
        name = ' '.join(c['name'].split())
        lines.append(f"- {name}: {number(c['score'].get('entry_probability'))} / {number(c['score'].get('raw_empirical_score'))} / {number(baseline.get('raw_score'))} / {difference} / {depth_text(c['score'])} moves")
    lines += ['', 'Entry probability is the modeled chance of reaching any qualifying chapter position, including later transpositions, counting each game once per chapter. Chapters may overlap, so entry probabilities are not additive. Each chapter score and its baseline use the same first-entry weights. The model stops at a deviation or theory leaf. A positive difference means a higher modeled score than the entry baseline. These are population estimates, not personal forecasts or proof of improvement.']
    lines += ['', 'Expected prepared depth counts my remaining prepared moves from first entry, including an available move at entry and continuations in other chapters. Shared prefixes are counted once per modeled game. It uses no cutoff or discount and measures usable preparation, not difficulty.']
    if overall.get('unresolved_mass', 0) > 0:
        lines += ['', f"Unresolved probability: {number(overall['unresolved_mass'])}. Overall score bounds: {bounds(overall['conditional_bounds'])}."]
    return '\n'.join(lines) + '\n'


def summary_markdown(reports, report_paths):
    """Combined summary built solely from current JSON results, without stale prose."""
    lines = ['# Repertoire summary', '',
             'Scores are from the repertoire owner\'s perspective, with a win worth 1 point and a draw worth half a point.', '',
             '| Repertoire | Starting-position reference | Repertoire score | Delta (pp) | Expected prepared depth (moves) |', '|---|---:|---:|---:|---:|']
    for r in reports:
        lines.append(f"| {r['color'].title()} | {pct(r.get('starting_position_reference', {}).get('owner_score'))} | {pct(r['overall']['raw_empirical_score'])} | {overall_delta(r)} | {depth_text(r['overall'])} |")
    lines += ['', 'Overall delta is repertoire score minus the starting-position baseline for the same color, in percentage points (pp). The model follows the selected repertoire moves and uses database frequencies for opponent replies. Population differences and sparse evidence limit what these comparisons establish; detailed uncertainty remains in the full reports.', '']
    for r, path in zip(reports, report_paths):
        lines += [f"## {r['color'].title()} chapters ({len(r['chapters'])})", '',
                  population_text(r), f"Database snapshot: {evidence_date(r)}.", '', *chapter_table(r['chapters'], concise=True), '',
                  f"[Full report]({path.with_suffix('.md').resolve().as_posix()}) | [Study description]({path.with_suffix('.study-description.md').resolve().as_posix()}) | [JSON]({path.resolve().as_posix()})", '']
    return '\n'.join(lines)
