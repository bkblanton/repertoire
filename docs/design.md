# Model

How Repertoire turns a set of study chapters and Opening Explorer tables into an expected score, and how it handles uncertainty and missing evidence. For how the results appear in the reports, see [Reports and metrics](metrics.md).

## What is estimated

For each color, Repertoire estimates:

- The expected score of the whole repertoire.
- Each chapter's expected score among games that enter it, and how often it is entered.
- How often games reach the end of prepared lines, and where they leave preparation before that.
- The uncertainty caused by the limited number of games behind each table.
- Which positions and replies matter most to the score.

Your own moves are treated as certain and the opponent's as probabilistic, using Lichess Opening Explorer statistics for a chosen population of players (rating bands, time controls and dates). The result is an estimate for that population. It is not calibrated to you, and it does not measure whether an opening is objectively good.

White and Black are analyzed separately. The combined row in the reports is a fixed 50/50 average of the two.

## The repertoire graph

Every chapter and variation is parsed with [python-chess](https://python-chess.readthedocs.io/). Malformed games, illegal moves and chess variants are rejected. Chapter IDs and names are kept for every position and move.

All chapters are merged into one graph of positions. A position is identified by its piece placement, side to move, castling rights and a legal en passant square, ignoring move counters, so transpositions within and across chapters meet at the same node. A line that ends in one chapter is not the end of preparation if another chapter continues from that position.

Every recorded variation is part of the repertoire. Comments are never read as instructions; lines can be removed explicitly with `exclude` in the configuration.

## Repetitions

A line can return to an earlier position, as in 1.e4 e5 2.Nc3 Qf6 3.Nd5 Qd8 4.Nc3. The game is drawn when a position occurs for the third time, so a loop is followed until then: the opponent picks again from the position's table every time it comes up, and the move into a third occurrence ends the game as a draw, scored 0.5 without database evidence.

Inside a loop the future depends on the game's history, which the position key leaves out. Positions in a loop are therefore expanded into one node per history: the position plus how often each position of the loop has occurred so far. Only the loop's own positions count, because positions before the loop cannot come up again and a game that leaves a loop can never return to it. Everywhere else a node is the position itself, and transpositions merge as usual. A loop can be entered at any of its positions; each entry gets its own nodes, so the draw always comes at exactly the third occurrence.

A chapter entry inside a loop is evaluated in the nodes games first arrive at from the starting position, in proportion to their probabilities. A position evaluated on its own counts as its first occurrence. Reports show one row per position: its reach adds up its occurrences, and its score and outcomes are averaged over them by reach. Move tables keep one row per occurrence, labeled by the line that goes round the loop.

## Choosing your moves

At each of your positions, one move is played:

1. **An explicit policy** in the configuration, a move or a set of weights summing to one, always wins.
2. **Competing chapters are decided by score.** Where chapters record different first moves, each move is scored with the best choices after it, deciding later positions first (backward induction), and the highest-scoring move is played. Together these choices maximize the score from every starting position. The comparison uses posterior mean scores, so a position with no evidence counts at its prior mean. An exact tie keeps the earlier chapter.
3. **Otherwise the first recorded move is played:** the main line before side variations, earlier chapters before later ones. Side variations within one chapter do not compete.

The resolved choices, every competing move's score and the winners are saved with the score, and every later stage replays them. Two moves are never both given probability one. Choosing the best of several estimated scores is biased upward, so a small winning margin can be chance.

## Opponent replies and where preparation ends

At every opponent position the model uses the position's full reply table, including after the last move recorded in a chapter:

- A reply that reaches a prepared position, directly or by transposition, continues the repertoire.
- Any other reply is a **deviation**, and preparation ends there. The model does not search through unknown positions for a later return to preparation, and never fetches the deviation's own table.

A position where you are to move and have no prepared move is a **theory leaf**, and preparation ends there too. Checkmate, stalemate, insufficient material and the third occurrence of a position are scored directly without database evidence.

The score, position reach, prepared depth, chapter entries and opening sources all follow these same transitions.

## The score

All scores are from your side. A table with $W$ wins, $D$ draws and $L$ losses for you scores

$$
S = \frac{W + \tfrac{1}{2}D}{W + D + L}.
$$

The value of a position $s$ is

$$
V(s) =
\begin{cases}
S(s) & \text{at a theory leaf,} \\
V(s_a) & \text{at your turn, playing your move } a, \\
\displaystyle\sum_{a \in K(s)} p(a \mid s)\, V(s_a) + \sum_{a \in D(s)} p(a \mid s)\, S(s, a) & \text{at the opponent's turn,}
\end{cases}
$$

where $K(s)$ are the replies that continue the repertoire, $D(s)$ are the deviations, $p(a \mid s)$ is the share of games at $s$ in which $a$ was played, and $S(s, a)$ is the score of the games with that move in the table at $s$. Reply probabilities are never renormalized over prepared replies: deviations keep their full share.

A theory leaf uses its own position's results, and a deviation uses its move's row in the parent position's table. These are different groups of games and are never substituted for each other.

Values are computed backward through the graph and probabilities forward. Probability arriving at a position by several routes is added up before it is passed on, and a shared position's value is computed once. Each point where preparation ends records its probability (the product of the opponent's reply probabilities along the way) and its score; together these reproduce the root value exactly. These probabilities are modeled frequencies under your repertoire, not observed frequencies of complete move sequences.

## Explorer evidence

Tables come from the authenticated [Lichess Opening Explorer](https://lichess.org/api#tag/Opening-Explorer) endpoint, `https://explorer.lichess.org/lichess`, with enough move rows to cover every legal move, no example games, and the same filters for the whole run.

- Tables are fetched for every opponent position, every theory leaf, every chapter entry and the starting position. A position where you have a prepared move needs no table, because your move is forced.
- Requests are sequential. Rate limits are waited out indefinitely; server errors and dropped connections are retried for up to 30 minutes per request.
- Every response is cached on disk, keyed by endpoint, position and filters, with its retrieval time. Runs can be stopped and resumed.
- Each score saves a manifest of its configuration, filters, input hashes and the cache entries it used.
- The token is never logged or written to any output.

Each table is checked: the win, draw and loss totals of the position must match the sum of its move rows. When every move is listed and some games are left over, those games form a separate **no recorded continuation** outcome, which is not treated as an opponent deviation. A table that cannot be reconciled is rejected.

A failed request is never treated as a position with no games, and neither becomes a zero score or a zero probability.

## Uncertainty

A rare database position may be common in your repertoire, because your moves are forced. Probability is therefore kept regardless of sample size, and uncertainty is modeled instead.

Each table gets a Dirichlet posterior. For an end score,

$$
(\theta_W, \theta_D, \theta_L) \sim \operatorname{Dirichlet}(W + \alpha_W,\; D + \alpha_D,\; L + \alpha_L), \qquad S = \theta_W + \tfrac{1}{2}\theta_D,
$$

with a default prior of $(0.5, 0.5, 0.5)$, set with `--prior`. Scores never fall back to the parent position's score. At opponent positions the posterior is over the joint table of moves and results, so reply probabilities and deviation scores share their evidence. The prior's total strength is spread across all legal moves, so positions with more legal moves are not smoothed more heavily. Legal moves that were never played keep their prior probability as an explicit unresolved share, rather than being treated as impossible.

Uncertainty is propagated analytically, not by simulation. A score is a sum over paths of products of probabilities from distinct positions, so:

1. **Its posterior mean is exact:** evaluate it once with each table at its posterior mean. The exception is a line that goes round a repetition loop, which uses a loop position's table twice; there the mean is off by that table's own variance, which shrinks with its number of games.
2. **Its variance is first-order:** each table's posterior variance, weighted by the square of that table's influence on the score, adding up its influence wherever it is used. Interactions between tables are left out.
3. **Chapter scores** divide by uncertain first-entry weights, so a table's influence includes its effect on those weights.
4. **Intervals** fit a Beta distribution to the mean and variance.

The same evidence is one quantity wherever it is used, including at transpositions. The target is uncertainty in the expected score, not in the result of a single game.

The Explorer does not show how far the games behind different positions overlap. Intervals are therefore approximate model-based credible intervals. They do not cover differences between the database population and you, dependence between games by the same players, or bias from how the repertoire was chosen.

## Missing evidence and sensitivity

An end position with no games is unresolved; any estimate based only on the prior is labeled as such. An opponent position with no usable reply table makes everything after it unresolved, rather than falling back to the position's overall score or to uniform play.

Unresolved probability is kept. With known contribution $K$ and unresolved probability $U$, the score lies in

$$
E \in [K,\, K + U].
$$

These bounds hold the resolved inputs fixed; they are not confidence intervals.

A second range assigns every sparse end position (fewer than 30 games by default) any score from 0 to 1, and a third rescores the repertoire with priors of 0.1 and 2. Three things are kept apart throughout:

- Posterior uncertainty from sampling variation.
- Sensitivity to the prior and to sparse evidence.
- Probability that is entirely unresolved.

## Chapters

A chapter's score is the expected score among games that first enter it, playing the chapter's own first moves where it records them and the overall repertoire elsewhere.

A chapter's PGN start is usually not where it begins in practice, because most chapters repeat the opening moves. A chapter is entered at its **entries**: by default the first positions on its lines, following its own first moves, that no other chapter continues from. A chapter that ends a line at a position does not share it. If other chapters continue from every position of a chapter, its first own-turn main-line position, or its start, is used. Entries can be set explicitly in the configuration; give competing systems the same entry to compare them directly.

Entry is the first arrival at any entry, by any route, under the chapter's moves. Later shared positions do not count, and a later return does not count as a second entry. The score, entry weights, baseline, depth, transitions and vulnerabilities all use the same entries and moves. Entry probability is reported separately from the conditional score. Every chapter is kept, including chapters whose moves lost to an alternative; their numbers are labeled as alternatives, with their reach under the moves actually played shown separately. Compatible continuations recorded in other chapters stay merged.

Chapters can overlap, so their contributions need not sum to the overall score, which is computed independently from the starting position. A chapter that starts from a custom position with no route to it gets a conditional score, but not an absolute reach, unless root weights are configured.

## Engine evaluations

Engine evaluations are a separate view, never an input to the model. When the Lichess evaluation export has been imported, each position's deepest Stockfish evaluation is converted to an expected score for you with the Lichess win-chance curve,

$$
E = \frac{1}{1 + e^{-0.00368208\,c}},
$$

where $c$ is the evaluation in centipawns from your side, and mate counts as 1 or 0. The reports compare $E$ with database scores where preparation ends and use its changes to judge single moves. A position without an evaluation stays missing. After an unprepared reply without one, the evaluation of the position before the reply, which assumes the opponent's best reply, is a lower bound, and is reported apart from evaluations.

## Outputs

Each score is saved as JSON containing:

- The raw empirical score, when fully defined, and the posterior mean with an approximate 95% interval.
- The probability of reaching a theory leaf, a deviation, another stop or an unresolved outcome.
- Sparse-evidence sensitivity, overall and per chapter.
- Sample counts, prior sensitivity and evidence provenance.
- Own-move conflicts, chapter entries and how they were chosen.
- Every end event, with its position, a representative move path, its type, chapters, probability, score, uncertainty and weighted contribution.

A deviation is never called good or bad without a stated baseline. Its contribution to the score is never negative; its effect relative to a baseline can go either way.

## Validation

Tests run on synthetic Explorer tables and cover:

- Forced own moves that are rare in the database.
- Deviations that raise or lower the expected score.
- Transpositions within and across chapters.
- Duplicate chapters and shared endings.
- Conflicting own moves.
- Sparse, empty, incomplete and inconsistent tables.
- Overlapping chapters and multiple entry routes.
- Repetition loops, entered at any of their positions, and scoring from Black's side as well as White's.

These invariants are checked:

- Every evaluated position conserves probability.
- The probabilities of all end and unresolved outcomes sum to one.
- The weighted end contributions reproduce the root value.
- Duplicating a chapter does not change the overall result.
- PGNs that differ only by transposition give the same results.
- The exact posterior means and first-order variances agree with a brute-force Dirichlet simulation.

Parsing, graph construction, Explorer access, estimation, evaluation and reporting are kept separate, and the evaluator runs entirely on cached or synthetic data.
