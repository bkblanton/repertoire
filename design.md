# Design: Expected Score of a Chess Repertoire

## Objective

Build a Python application that reads a Lichess study PGN and estimates:

- The expected score of the overall repertoire.
- Each chapter’s expected score conditional on entering it.
- The probability of reaching theory leaves versus leaving theory.
- Uncertainty caused by limited database evidence.
- Which positions and deviations most affect the estimate.

Use Lichess Opening Explorer statistics for a configurable player population. Treat our selected repertoire moves as certain and opponent moves as probabilistic.

The result is a population-based model estimate, not a prediction calibrated to the individual user or a causal measure of opening quality.

## Inputs

Required:

- Study PGN.
- Repertoire color.
- Lichess token, supplied through an environment variable.
- Explorer filters: ratings, time controls, and date range.

Optional configuration:

- Our move choices where the repertoire contains alternatives.
- Chapter entry positions or paths.
- Prior parameters, simulation count, random seed, and cache policy.
- Reporting tolerance for uncertainty, expressed in score percentage points.

Analyze White and Black repertoires separately. Combine them only when explicit color weights are supplied.

## Repertoire representation

Parse every chapter and variation with `python-chess`. Reject malformed games, illegal moves, and unsupported variants. Preserve chapter IDs, names, and provenance for positions and moves.

Build a shared graph of canonical positions. For standard chess, the key includes:

- Piece placement.
- Side to move.
- Castling rights.
- Legally available en passant.

Exclude move counters for ordinary opening transpositions.

Merge identical positions and deduplicate edges. A position ending one chapter is not an overall theory leaf if another chapter supplies a continuation.

At our turn, select the first recorded move: main PGN variation before side variations, and earlier chapters before later chapters. Explicit configured moves or mixture weights summing to one override that default for the overall policy. Record resolved conflicts in diagnostics. Never assign probability one to multiple alternatives.

All included PGN variations are assumed to represent repertoire content. Allow explicit exclusions for illustrative lines or annotated mistakes; do not infer exclusions from prose comments.

At an opponent node with theory continuations, an unlisted move that immediately reaches a known position counts as a transposition into theory. Otherwise, it is a deviation, and evaluation stops there. Do not search through unknown positions for later re-entry.

A position with no continuation in the merged repertoire is a theory leaf.

Detect reachable graph cycles after resolving our policy. For the initial implementation, fail with an actionable cycle report. Do not silently treat a repeated position as a draw. Exact repetition handling requires history beyond the canonical position key.

## Probability and score model

All scores are from the repertoire owner’s perspective:

\[
S=\frac{W+\frac12D}{W+D+L}
\]

At our turn, follow the selected move with probability one.

At an opponent node, use the observed move frequency within the selected explorer population. Do not renormalize over repertoire moves alone: deviations retain their full probability.

For position \(s\):

\[
V(s)=
\begin{cases}
S(s), & \text{theory leaf}\\
V(s_a), & \text{our selected move }a\\
\sum_{a\in K(s)}p(a\mid s)V(s_a)
+\sum_{a\in D(s)}p(a\mid s)S(s,a),
& \text{opponent turn}
\end{cases}
\]

Here, \(K(s)\) contains theory continuations and \(D(s)\) contains deviations.

Use position-level results for theory leaves. Use the parent explorer response’s move-specific results for deviations. These are different statistical populations and must not be substituted interchangeably.

Recognize checkmate, stalemate, and insufficient-material outcomes directly instead of requiring database evidence.

Compute values backward through the graph and probability mass forward. Sum incoming mass at transpositions before propagating it. Cache a shared position’s value without discarding alternative routes into it.

For each stopping event, record:

\[
\text{contribution}=\text{absolute probability}\times\text{score}
\]

Absolute path probability is the product of opponent move probabilities, with our moves contributing a factor of one. Label it as modeled repertoire probability, not observed historical sequence frequency.

## Explorer client and data integrity

Use authenticated requests to:

`https://explorer.lichess.org/lichess`

Request enough move rows to cover all legal moves. Disable example-game payloads. Keep filters identical throughout a run.

Implement:

- Sequential requests, timeouts, bounded retries, and rate-limit backoff.
- Persistent caching keyed by canonical position, endpoint, and all filters.
- Response validation and resumable runs.
- A run manifest containing configuration, retrieval timestamps, and data provenance.
- No token logging or inclusion in reports.

Compare parent win/draw/loss totals with the sum of returned move rows.

If complete move coverage leaves valid nonnegative residual counts, retain them as a distinct “no recorded continuation” stopping bucket. Do not label this bucket an opponent deviation. Reject unexplained inconsistencies.

Distinguish API failures from successful zero-data responses. Neither becomes a zero score or zero probability.

## Sparse data and uncertainty

Preserve repertoire probability regardless of sample size. A rare database position may be common under our forced move choices.

Represent a stopping score using a Dirichlet posterior over wins, draws, and losses:

\[
(\theta_W,\theta_D,\theta_L)
\sim
\operatorname{Dirichlet}
(W+\alpha_W,D+\alpha_D,L+\alpha_L)
\]

\[
S=\theta_W+\frac12\theta_D
\]

Use configurable priors. A weak default of \((0.5,0.5,0.5)\) is acceptable, but document it and expose prior sensitivity. Do not automatically back off to the parent’s score.

Propagate uncertainty with Monte Carlo:

1. Sample stopping-score parameters.
2. Sample opponent move probabilities.
3. Evaluate overall and chapter scores.
4. Summarize the resulting distributions.

At opponent nodes, sample a joint move-by-result table and derive move probabilities and deviation scores from it. This preserves their shared local evidence. Give the prior a controlled total strength so positions with more legal moves do not accidentally receive much stronger smoothing.

Represent legal but unobserved moves explicitly or in an auditable unresolved bucket. Never silently assume they are impossible.

Within each simulation, reuse the same sampled quantity wherever the same evidence is referenced, including transpositions. Do not sample individual game outcomes: the target is uncertainty in expected score.

Explorer aggregates do not expose all overlap between historical games at different positions. Label intervals as approximate, model-based credible intervals. They do not cover all population mismatch, player dependence, or repertoire-selection bias.

## Missing evidence and conservative bounds

Treat zero-data terminal scores as unresolved by default. Any prior-only estimate must be explicitly labeled.

If an opponent node has no usable move distribution, its downstream repertoire value is unresolved. Do not replace it with the node’s historical score or assume uniform opponent play.

Preserve unresolved probability mass. For known contribution \(K\) and unresolved mass \(U\), report:

\[
E\in[K,K+U]
\]

These are conditional bounds holding the resolved inputs fixed, not confidence intervals.

Also report a sensitivity range obtained by assigning all flagged sparse stopping events scores between zero and one. Their combined probability determines the width of this range.

Keep three concepts distinct:

- Posterior uncertainty from modeled sampling variation.
- Sensitivity to priors and sparse evidence.
- Completely unresolved probability mass.

## Chapter semantics

A chapter’s score means:

> Expected repertoire score conditional on first entering this chapter, preferring that chapter's first own moves and following the overall policy elsewhere in the merged repertoire.

Do not assume the chapter’s PGN root is its meaningful opening entry. Many chapters repeat moves from the initial position.

Support explicit entry positions or paths and chapter subject regions. Otherwise use all first chapter-unique positions as region anchors, including chapter-owned descendants. If none can be reached under the chapter policy, fall back to its first mainline opponent reply or PGN root. Find first arrival across all routes under the chapter comparison policy. Use that same policy for the score, entry weights, baseline, prepared depth, chapter transitions and vulnerabilities. Retain every chapter, including alternatives excluded by overall chapter order. Label alternative comparisons and report region reach under the overall policy separately. For comparisons of whole alternative systems, configure the same subject anchor for both. Compatible continuations split across chapters remain merged.

For multiple entry routes, use first-entry probabilities and their conditional weights. Do not count a later return to the same chapter as another entry.

Report chapter entry probability separately from its conditional score.

Chapters may overlap, so their contributions need not sum to the overall score. Compute the overall score independently from the repertoire root. If additive chapter attribution is requested, require an exclusive attribution rule and retain an “outside chapters” category.

Custom-FEN roots without a connecting path can receive conditional scores, but not absolute reach probabilities without supplied root weights.

## Outputs

Produce machine-readable JSON and a concise human-readable report containing:

- Raw empirical estimate when fully defined.
- Posterior mean and approximate 95% credible interval.
- Theory-leaf, deviation, other-stop, and unresolved probability masses.
- Overall and chapter-specific sparse-data sensitivity.
- Largest score contributions and largest uncertainty contributors.
- Sample counts, prior influence, and evidence provenance.
- Policy conflicts, chapter-entry ambiguities, and unsupported cycles.

For each terminal event, include its position, representative move path, event type, chapter references, probability, score estimate, uncertainty, and weighted contribution.

Do not call a deviation beneficial or harmful without defining a comparison baseline. Its absolute contribution is nonnegative; its effect relative to a baseline may have either sign.

## Validation and acceptance

Use synthetic explorer fixtures for deterministic tests. Cover:

- Forced own moves despite low database popularity.
- Opponent deviations that raise or lower expected score.
- Transpositions within and across chapters.
- Duplicate chapters and shared leaves.
- Conflicting own moves.
- Sparse, zero-data, incomplete, and inconsistent responses.
- Chapter overlap and multiple entry routes.
- Cycle detection and color reversal.

Required invariants:

- Every evaluated node conserves probability.
- Terminal and unresolved probability masses sum to one.
- Weighted stopping contributions reproduce the root value.
- Duplicating a chapter does not change the overall result.
- Equivalent transposing PGNs produce equivalent results.
- Monte Carlo results are reproducible with a fixed seed.

Keep parsing, graph construction, explorer access, statistical estimation, evaluation, and reporting separate. The evaluator must run entirely against cached or synthetic data without network access.

## Planned extension: average opponent rating

This is an implementation plan, dated 2026-10-01. The rating extension has not been implemented or used to regenerate reports yet. It supplements the current consolidated report and summary; it does not replace the score model.

### Scope limit

Calculate and report opponent-rating averages only for individual positions, lines, and chapters. Do not add White-repertoire, Black-repertoire, combined-study, or whole-study baseline rating averages, including in JSON. Globally ranked tables may still show a separate rating for each line. Whole-graph policy flow is used only to establish the incoming context of those lines and chapter entries.

### Position-level source rule

Use Lichess `averageRating` only when the player making that move is our opponent. The field describes the move maker, as explained by the [Lichess Explorer maintainer](https://lichess.org/forum/lichess-feedback/opening-explorer-tweaks).

- **Our turn:** read the cached parent move row for the opponent move that reached this position. Never use averageRating on our own move to estimate the opponent's rating.
- **Opponent's turn:** take the game-count-weighted average of the opponent's recorded move rows at the current position. For row m, n_m is white wins + draws + black wins. The mean is sum(n_m * rating_m) / sum(n_m), over positive-count rows with a usable rating.
- **Specific opponent-reply rows:** use that exact reply's parent-row rating. This applies to prepared replies, unprepared replies, and moves that immediately transpose into preparation.
- **Positions reached by several routes:** merge all incoming opponent move rows with weights from the active repertoire policy's arrival flow. Use route weights, not the representative PGN line or historical counts pooled across parents. Include every contributing source and its coverage in JSON.
- **A chapter entry with our side to move:** reconstruct the incoming edge mixture from the first-entry traversal under that chapter's comparison policy. Ordinary unconditional reach at the entry, or one PGN parent, is insufficient.
- **A chapter or line root with our side to move and no preceding move:** the local rating is unavailable. Do not substitute the rating of our first move. That chapter or line continuation can still have an opponent-rating estimate from its later stopping evidence.

Examples checked against the current cache:

| White position | Source | Average opponent rating |
|---|---|---:|
| 1. e4 c6 2. d4 d5 3. exd5 cxd5 4. Bd3 Nf6, White to move | Cached Black Nf6 row before the reply | 1935 |
| 1. e4 e5 2. Nc3 Nc6 3. g3, Black to move | Count-weighted Black replies at this board | About 1760 |

These examples require no child-position queries. The first reached board is outside preparation; its opponent rating is already in its parent's table.

### Chapter and line averages and their meaning

Keep two explicit quantities in JSON:

1. **Position opponent rating:** the local estimate from the turn-based rule above. Use this for position, decision, and move-context tables.
2. **Score-evidence opponent rating:** the average opponent rating attached to the stopping evidence used by a chapter or a particular line continuation. Use this beside chapter and line scores only.

Compute the second quantity from the same stopping outcomes and probabilities used by the empirical score. For stopping outcome i, let q_i be its modeled probability, r_i its available opponent-rating estimate, and c_i its rating coverage:

- Known rating mass K = sum(q_i * c_i).
- Weighted rating sum M = sum(q_i * c_i * r_i).
- Average opponent rating = M / K when K > 0; otherwise unavailable.
- Missing rating mass = 1 - K, for a normalized chapter or line scope.

Do not average ratings across every visited node or decision. That would give longer branches more weight and count the same modeled game repeatedly. Opponent-turn leaves use their cached response mean; own-turn leaves and deviations use the opponent row that reached them. An own-turn stopping board reached through different opponent moves needs edge-specific rating contributions before merging.

For chapter comparisons, use the same first-entry weights and chapter-local own-move policy as the existing score. Individual lines in globally ranked tables use their reach under the overall selected policy; alternative chapters retain their separate comparison contexts. A canonical board can have a different incoming rating mixture in different chapter or line contexts.

Chapter entry-baseline rating context uses the existing first-entry mixture applied to local position ratings, rather than continuation stopping ratings. Do not combine ratings across chapters or repertoire colors. Missing chapter or line data is never treated as rating zero.

### Coverage, missing data, and provenance

At an opponent-turn board, coverage is the number of games represented by usable rated move rows divided by the cached board's total game count. This discloses games without a recorded continuation and moves with missing ratings. At an own-turn board, coverage is the share of incoming policy flow whose opponent move row has a usable rating.

Missing ratings, absent incoming parents, zero games, terminal positions with no responses, and unnamed residual outcome buckets remain unavailable or partial. A residual outcome bucket has no identifiable move-maker rating; do not fill it with the mean of named replies. Incomplete coverage does not change score probabilities or renormalize opponent move frequencies.

Serialize a structured rating value with mean, known/missing coverage, basis, rated observations where meaningful, incoming origins and weights, policy identity, and cache provenance. Ratings supplied by Lichess are rounded means; this data alone does not support a sampling confidence interval for the rating mean.

Preserve all current filters: rated blitz, rapid, and classical, all rating groups, and the saved date range. These are pooled rating descriptions across time controls. Do not turn them into a rating-adjusted win rate, a difficulty penalty, or an absolute personal-performance rating. Existing repertoire scores, rankings, deltas, Elo equivalents, and correlation coefficients remain unchanged.

### Report integration

Keep exactly the two existing readable files, reports/report.md and reports/summary.md. Add Avg opponent rating alongside the relevant scores or position contexts:

- **Overview, combined section, color score sections, and whole-study starting baselines:** no opponent-rating averages or aggregate rating-coverage fields.
- **Chapter tables in both full report and summary:** chapter score-evidence opponent rating. Show entry-baseline rating and first-entry coverage in the expanded chapter analysis.
- **Most common prepared and unprepared positions:** local position opponent rating, preserving the current separation and omission of positions immediately before a guaranteed own reply.
- **Prepared and unprepared opponent vulnerabilities:** the specific opponent reply's averageRating. This is exact move context even when the after-reply score includes a longer prepared continuation.
- **Our selected moves:** the resulting opponent-turn board's cached response mean. The parent's row for our move contains our own rating and must not be used. For an observed alternative, use its resulting board only if already cached; otherwise disclose unavailable.
- **Stopping-position contributions, uncertainty priorities, and exact entry routes:** the rating from the stopping or entry context. Preserve arrival-specific ratings until probability-weighted aggregation.
- **Preparation/reply tables:** own-decision rows use the preceding opponent move; opponent-decision rows use the current reply distribution. Pawn-structure groups within a chapter can summarize their stopping-outcome rating mixture; example routes use their own position context. Whole-repertoire feature groups do not receive rating averages.

Use one compact rating column for line tables. Disclose partial coverage inline or in the associated scope note and retain full details in JSON. Distinguish local position ratings from score-evidence averages in definitions. Baseline comparisons must not imply identical rating populations when the evidence does not establish that.

### Implementation sequence

1. Add a shared repertoire_score/ratings.py module for pure parent-row extraction, current-reply weighted means, arrival aggregation, stopping-evidence moments, and coverage. It must accept evidence and model flows rather than perform HTTP requests.
2. Reconstruct arrivals under each existing policy profile. Extend the first-entry calculations in evaluate.py/transitions.py as needed to expose incoming edge flow and known rating mass. Reuse the canonical graph and existing own-move precedence rules.
3. Add a cache-only repertoire-ratings command. Write reports/data/white.ratings.json and black.ratings.json with score/source hashes, filters, supporting-analysis hashes where used, and cache provenance. The ledger holds chapter ratings and position/move/entry identities, with no color-level or combined rating averages, so current score JSONs and their correlation provenance need not be rewritten.
4. Reuse preparation.py's Evaluator for continuation rating contexts. Share the same source rule with character.py and vulnerabilities.py; never create independent versions of the arithmetic in each renderer.
5. Add the rating ledger to consolidated.py's companion validation and rendering. Update render.py, pyproject.toml, README.md, and run-repertoires.ps1 so ratings are generated before final assembly and stale rating snapshots cannot be mixed into the report.
6. Run the tests and cache-only validation below, generate fresh rating ledgers, and regenerate report.md and summary.md. Preserve the source PGNs and all Explorer cache files.

The extension does not request candidate child endpoints, expand unprepared lines, or collect new filtered rating buckets. Existing cached ratings suffice; missing information is disclosed.

### Acceptance checks

- Verify both repertoire colors and both sides to move with deliberately different own/opponent averages; using the wrong player's field must fail.
- Verify count-weighted reply means, empty/zero/missing ratings, named-reply coverage, and unavailable own-turn roots.
- Verify multiple parents and immediate transpositions using policy arrival weights distinct from database-count weights. Duplicate PGN providers must not duplicate flow or alter results.
- Verify multiple chapter entries, chapter-local alternatives, and first-entry edge mixtures. Use each active policy's flow rather than the overall or representative route by default.
- Verify chapter and line weighted rating sums and known/missing mass against an independent enumeration of stopping events. Known and missing mass must sum to one in each normalized chapter or line context; deeper lines must not receive extra weight.
- Verify that White, Black, combined, and whole-study baseline summaries contain no rating averages in Markdown or JSON. Globally ranked individual-line rows must retain their permitted rating context.
- Verify partial and missing chapter/line ratings and continuation contexts.
- Forbid network access in fixtures and in real-data regeneration. Specifically prove that an unprepared reply gets its rating from a cached parent even when its child board has no cached entry.
- Validate rating-ledger hashes, report links, coverage labels, and source preservation. Existing score JSON bytes, scores, reach, depth, vulnerability rankings, and correlation estimates must reproduce the current saved results.


## Revised report comparisons and layout

Own-move vulnerability drag is parent database score minus the full repertoire continuation score after the selected move, in percentage points. Rank by this direct difference and display modeled reach separately. Strengths rank the opposite positive difference. Preserve historical move-row scores only as diagnostic JSON data. Opponent reply drag keeps its reach-weighted comparison against the parent repertoire value.

Replace strongest/weakest endpoint rankings with one strengths section containing strongest own moves and highest raw stopping-position contributions. Merge exact boards of the same stopping type across arrivals, use reach-weighted scores and opponent ratings, and flag overlapping pooled sample counts. The sum of all known stopping contributions must reproduce the resolved score; missing evidence must stay unresolved. Scores at prepared endpoints use endpoint evidence and unprepared replies use cached parent rows. Never request candidate children for this report.

The summary leads with snapshot status and one score overview, then per-color strengths, deficits, common preparation gaps and highest reply drag, plus stopping reasons. Chapter tables are expandable. The full report places each color's common positions and chapter overview before detailed analyses. State reach denominators, distinguish database and continuation scores, omit empty categories, and link repeated definitions. A saved-comparison refresh may use matching saved character continuation values without parsing a newer PGN; preserve score, preparation, character, correlations, cache and PGNs, validate all hashes, and mark the saved snapshot clearly.
