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

At every prepared opponent-turn position, including after the last move recorded in the PGN, expand its cached opponent response table. An unlisted move that immediately reaches a known position counts as a transposition into theory. Otherwise, it is a deviation, and evaluation stops there. Do not search through unknown positions for later re-entry or fetch their response tables.

A position with no prepared continuation at our turn is a theory leaf. An opponent-turn position with no recorded PGN continuation still uses its cached replies and can transpose back into preparation. Score, position reach, prepared depth, chapter entries, and opening sources all follow these same transitions. The reach of an unanswered own-turn board equals its first-gap probability, summed over all transposed arrivals.

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
