# Reports and metrics

How to read the reports, how the score is calculated, and what each metric means. The full report ends with a glossary that defines every column briefly; this page gives the detail.

A few rules hold throughout:

- **Scores favor you**, the repertoire owner, for both colors: a win is 1, a draw 0.5 and a loss 0.
- **Positions are exact boards.** Every move order that reaches the same position (same pieces, side to move, castling rights and en passant square) is combined into one row.
- **Overlapping numbers do not add up.** Positions along the same game, chapters that share lines, opening categories and highlighted gains all overlap, so their reach and contributions must not be summed.
- **Missing evidence stays missing.** A position with no games, or a table that could not be fetched, is reported as unresolved, never as a zero score.
- **The numbers describe, they do not explain.** A higher score means a higher modeled score from database games, not proof that a move causes better results.

## Contents

- [Reading the reports](#reading-the-reports)
- [How the score is calculated](#how-the-score-is-calculated)
  - [Uncertainty](#uncertainty)
  - [Missing and sparse evidence](#missing-and-sparse-evidence)
- [Metric reference](#metric-reference)
  - [Scores and baselines](#scores-and-baselines)
  - [Chapter reach and entries](#chapter-reach-and-entries)
  - [Prepared depth](#prepared-depth)
  - [Where preparation ends](#where-preparation-ends)
  - [Free transpositions](#free-transpositions)
  - [Gaps](#gaps)
  - [Score spread and outcome volatility](#score-spread-and-outcome-volatility)
  - [Reuse, reply variety and position profiles](#reuse-reply-variety-and-position-profiles)
  - [Effort and value](#effort-and-value)
  - [Engine view](#engine-view)
  - [Opening names](#opening-names)
  - [Opponent ratings](#opponent-ratings)
  - [Correlations](#correlations)
- [Strengths and vulnerabilities](#strengths-and-vulnerabilities)
  - [Where your edge comes from](#where-your-edge-comes-from)

## Reading the reports

### The summary

Start with `reports/summary.md`. After the headline scores, each color leads with what to work on:

- **Where preparation ends** groups every unprepared reply by the last prepared position before it, so one row is one study task. *Games leaving prep here* is the share of all games with that color whose preparation ends at that position. *Of games at this position* separates a chapter that simply stops (100%) from rare sidelines at a busy position. Unlike other rankings, these rows do not overlap.
- **Own moves to review** ranks your moves by how often they are played times how far they fall below the database score of their position.

Collapsed sections follow: the most common positions as a tree, chapter comparisons, costly unprepared replies, where your edge comes from, and preparation and variability. Where chapters compete in the same position, each alternative is listed with its score (see [Move selection](usage.md#move-selection)). Saved [comparisons](usage.md#comparing-alternative-preparation) are listed at the end. Notices about changed studies or missing analyses appear above the scores.

### The full report, chapter and opening pages

`reports/report.md` holds the detail for both colors: chapter tables, exit points, free transpositions, positions, openings, vulnerabilities, strengths, where the edge comes from, effort and value, the engine view, gap priorities, depth distributions, correlations, and the glossary (*Definitions and evidence*). Tables state their main caveat in one sentence and link to the glossary instead of repeating it.

Each chapter has its own page in `reports/chapters/` (`W1.md`, `W2.md`, ... and `B1.md`, `B2.md`, ...). It opens with a table of the chapter's headline figures and a short list of its opponents, preparation, gaps and evidence, then shows where the chapter starts, its exit points, free transpositions, positions, vulnerabilities, strengths, where its edge comes from, gaps, depth, and every entry position and route. It links to the neighboring chapters and to the chapter on Lichess. `reports/openings/white.md` and `black.md` hold the evidence for each opening.

### Columns

| Column | Meaning |
| --- | --- |
| Repertoire score | Your expected score when you play the repertoire's moves. |
| Baseline / delta | The ordinary database score at the start or at the chapter's entries, and the repertoire score minus it. |
| Position reach | Probability of reaching a position, by any move order, before preparation ends. On chapter pages, among games that enter the chapter. |
| Move reach | Probability of reaching the parent position and then playing that move. It can be lower than the reach of the resulting position when other move orders also lead there. |
| Games leaving prep here | Probability that preparation ends right after this prepared position, over all of its unprepared replies. |
| Gap reach | Probability of first reaching a position where you have no prepared move. |
| Score spread | How much the scores of the continuations vary, including later replies. Omitted for unprepared replies, where preparation has ended. |
| 1 in N games | How often a position or move comes up, shown under its reach: 1 / reach, assuming independent games. |
| Per 1,000 games | A reach-weighted gain, drag or contribution, in score points per 1,000 games with that color (or entering the chapter). |
| Games | Games in the database at that position or in the parent table's row for that move. A position where you have a prepared move counts the opponent move rows that lead to it. Not the sample size of the whole continuation. |
| Opponent rating | The average rating of the opponents at that point, with its difference from the parent position on reply rows. It describes the database games and never adjusts a score. |

### Lines, links and units

A line shown in a table is one legal route to the position, not the only way games get there. Under it are the most common opening name on the way to the position and the chapters that contain it, with consecutive chapters as ranges such as W1-W3. Unprepared replies name the chapters they leave (*unprepared in W1*), and replies that transpose name the chapters they reach (*transposes to W5*). Each position uses the same representative route as its label in every table, and a move row adds its move to its parent's label. Line links open the Lichess analysis board where you are to move: after the opponent's reply, or before your move. A position followed by a move you always play is merged into the position after that move.

Tables show percentages with one decimal, or two below 1% so that rare lines stay distinct. A difference such as `+5.0%` means five percentage points, not a relative change. Large game counts are abbreviated (`27k`, `3.1M`). Centipawn equivalents appear only beside headline deltas. Columns that are the same in every row are left out. A dagger (†) marks pooled game counts that may count the same historical games more than once.

## How the score is calculated

Your selected moves are played every time (or with the weights in your configuration). Each opponent reply gets its share of all games at that position in the Explorer, including replies you have not prepared. The model then follows every reply:

- **A prepared reply** continues the repertoire. This includes replies after the last move of a line that transpose straight into another prepared position.
- **An unprepared reply** ends preparation. It is scored with that move's results in the parent position's table, so its own table is never fetched.
- **A position where you have no move** ends preparation and is scored with that position's results.
- **Checkmate, stalemate and insufficient material** are scored exactly.

The repertoire score is the probability-weighted average of these end scores. The same rules drive position reach, prepared depth, chapter entries and opening sources. Games in a table that match no listed move form a separate *no recorded continuation* outcome. The [model description](design.md) gives the formulas.

### Uncertainty

The repertoire score is computed exactly from the cached tables. Its uncertainty asks how much the score could move because each table is a finite sample of games. It is calculated, not simulated, so it needs no random seed.

Each position's table gets a Dirichlet posterior over its moves and results: the observed counts plus a prior. The default prior is 0.5 wins, 0.5 draws and 0.5 losses; at opponent positions that strength is spread across every legal move and any games with no listed move. A score is a sum over paths of products of probabilities from different positions, so:

- **Posterior means are exact.** They come from one pass with each table replaced by its posterior mean.
- **Variances are first-order.** Each table contributes its variance, weighted by the square of its influence on the score. Interactions between pairs of tables are left out; they matter only for very sparse positions. A position reached by several move orders contributes once.
- **95% intervals** fit a Beta distribution to each score's mean and variance. Gains, drags and paired differences use a normal interval.

Change the prior with `score run --prior W D L`. Reports also show the score under priors of 0.1 and 2 per result. The main tables show the raw score; the posterior mean is kept in the JSON. Intervals do not account for the same game counting in several positions' tables, for selection bias, or for differences between the database population and you.

### Missing and sparse evidence

- **Unresolved evidence.** A position with no games, or with no usable reply table, is unresolved. Reports give bounds, from the resolved contribution alone to the resolved contribution plus all unresolved probability. These are bounds given the rest of the model, not confidence intervals.
- **Sparse evidence.** A position with fewer than 30 games (set with `--sparse-threshold`) is flagged sparse. Sparse positions still count in the score, and a sensitivity range shows how far the score could move if each sparse outcome scored anywhere from 0 to 1.

## Metric reference

Whole-repertoire metrics start from the starting position. Chapter metrics start from the chapter's entries, weighted by how often each is reached first, and use the chapter's own moves where it records them.

### Scores and baselines

The **baseline** is the ordinary database score before forcing any of your moves, under the same Explorer filters: at the starting position for a whole color, or at the chapter's entries, weighted by how often each is reached first. The **delta** is the repertoire score minus the baseline, in percentage points. A missing entry baseline stays unresolved.

The overview and summary show both colors and a **combined** row that weights White and Black equally. It averages scores, baselines and depths and adds up chapter counts. It appears only when both colors use the same filters and the standard starting position.

Each overview row also shows an **Elo equivalent**, `400 * log10(score / (1 - score)) - 400 * log10(baseline / (1 - baseline))`. It translates the score edge to the Elo scale; it is not a measured rating gain.

Headline deltas also show a **centipawn equivalent (CP)**, using `C(p) = ln(p / (1 - p)) / 0.00368208`, the inverse of the [Lichess score curve](https://lichess.org/page/accuracy). For example, a 52.50% score for Black is about +27 cp from Black's side. CP delta is `C(score) - C(baseline)`. These convert human results, not engine evaluations. Mixtures are converted after averaging their scores, and CP is unavailable at 0% or 100%.

### Chapter reach and entries

A chapter is reached at its [entries](usage.md#chapter-entries): by default, the first positions on its lines that no other chapter continues from.

**Entry probability** is the chance of reaching one of a chapter's entries, by any move order, before preparation ends, counting each game once. A transposition onto an entry counts. A transposition onto a later position that the chapter shares with others does not: that game belongs to the chapter whose entry it passed. For example, a chapter that reaches the Vienna through 1...Nf6 2.Nc3 e5 is entered only at 1...Nf6, and Vienna games do not count toward it.

A **chapter score** is the expected score among games that enter the chapter. After entry the whole repertoire applies, including other chapters' continuations, but the chapter's own moves are used wherever it records them. Its baseline, depth, transitions and vulnerabilities use the same entries and moves. For a chapter whose moves lost to an alternative, these numbers describe what would happen if you played it; a separate reach shows how often the repertoire you actually play reaches its entries.

Chapters still overlap where one chapter's line passes through another's entry, so chapter reach and scores do not add up to the whole.

**Chapter transitions** (full report) answer: among games that first enter one chapter, how often do they reach another at or after that point? Entering both at once counts. Transitions are directed: A to B differs from B to A. The model does not follow games through unknown positions to a later return.

**Entry routes** (chapter pages) show the most likely single route to each entry. An entry's weight includes every route that reaches it first; the route shown may carry only part of that weight.

### Prepared depth

**Expected prepared depth** is the expected number of your own prepared moves still to come before preparation ends. Each of your moves counts one; opponent moves count nothing themselves but weight what follows by how often they are played. An available move at the start counts; earlier moves and the entry itself do not. There is no cutoff or discount. Chapter depth starts from the chapter's entries and follows the whole repertoire from there.

Depth is computed backward: a position where you move has depth `1 + depth after your move`, an opponent position has `sum(reply probability * depth after reply)`, and an unprepared reply or end of a line has depth zero. Positions reached by several move orders are counted once. Missing results do not affect depth; a missing reply table gives lower and upper bounds.

The **prepared-depth distribution** in the full report shows the probability of playing at least each number of prepared moves, and where games end at each depth: at a prepared endpoint, an unprepared reply or another stop. Paths that transpose into the same position keep their own move counts. The "at least" probabilities from depth 1 sum to the expected depth. The summary shows the median.

### Where preparation ends

**Exit points** group every place a game leaves preparation by the last prepared position before it: the position where the opponent chose an unprepared reply, or a position where you have no move. **Games leaving prep here** is the total probability of those exits, and **share of games at this position** divides it by the position's reach. Each game leaves preparation once, so exit rows do not overlap. Finished games and missing opponent data are left out. The database score is the reach-weighted average of the unprepared replies' results.

A position where many games leave through many rare replies, such as a chapter that ends one move early, appears as one row instead of dozens of 1% reply rows.

### Free transpositions

Preparation ends at an unprepared reply, and the reports score that game with the reply's row in the parent table: the database result of every game after it, whatever you played next. Sometimes one of your legal moves from there reaches a position you have prepared, so you could continue your preparation. The full report lists these replies, and each chapter page lists the ones that leave that chapter.

**Change** is your repertoire score at the prepared position minus the database score after the reply. It splits into the **move**, the prepared position's database score minus the score after the reply, and your **preparation**, your score minus the prepared position's database score. For example, after 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qd8 4.d4 e6, 5.Ne4 reaches a French Rubinstein position: its database score is lower than after 4...e6, but preparation there can make up the difference.

The comparison is with average play after the reply, because the table at that position is not fetched. A negative change is therefore a good reason not to transpose, while a positive one only says that transposing beats average play, not that it is your best move. The prepared position's database score also counts every move order into it, which can differ from games that arrive this way. The 95% interval treats the reply's row and the prepared position's evidence as independent. Thinly sampled replies are left out, and the score itself never follows a game back into preparation.

### Gaps

A **gap** is the first position a game reaches where you have no prepared move.

**Equivalent gap reach** summarizes how concentrated your gaps are: `R = sqrt(sum(p_i ** 2))`, where `p_i` is the probability of first reaching gap `i`. `R ** 2` is the probability that two independent games hit the same first gap, and `R` is the reach of a single gap with that same repeat probability. Lower values mean gaps that are rarer or more spread out. Chapter tables show it among games entering the chapter, and a **weighted gap reach contribution**, the entry probability times that value. These do not add up across chapters.

**Gap priorities** rank each gap by its share of the repeat probability, `p_i ** 2 / sum(p_j ** 2)`, showing which gaps dominate. Unknown gap probability, from missing reply tables or table rows with no listed move, is shown separately.

### Score spread and outcome volatility

**Score spread** measures how much the scores of the continuations from a position differ. It is computed backward, `B(s) = sum(p * (B(child) + (score(child) - score(s)) ** 2))`, and shown as `100 * sqrt(B(s))` in percentage points. Known end scores have no spread, and your forced moves inherit the spread after them. **Reply** spread, shown beneath, covers only the opponent's next reply. **Prep ends** marks where the model stops, not a game result.

**Outcome volatility** also includes the spread of results within each end position: `400 * (W + D/4 - (W + D/2) ** 2)` on a 0% to 100% scale, using the expected win and draw rates. It is 100% for equal wins and losses, 10% for 5% wins, 90% draws and 5% losses, and zero for a certain result. Because most database games are decisive, it sits near 90% to 95% in every chapter, so it appears only in detailed breakdowns. It is stored as `outcomes.sharpness` in the JSON.

### Reuse, reply variety and position profiles

**Expected reuse.** A move of yours that is played with probability `p` per game is expected `N * p` times in `N` games, and seen at least once with probability `1 - (1 - p) ** N`. Summing over your moves gives total encounters and the number of distinct moves seen; the difference is repeat encounters. The curve uses 10, 50, 100 and 500 games by default (`character --games`). Chapter curves count games entering the chapter. This measures exposure, not memory.

**Reply predictability** is the entropy of the opponent's replies at each of their positions, shown as effective replies, `2 ** H`. The summary averages the entropy by reach. Games with no listed move count as missing coverage, never as another move.

**Position profiles** describe the positions where preparation ends, including after an unprepared reply: queens on the board, king wings, bishop pairs, isolated, doubled and passed pawns, isolated d-pawns and pawn structures, with an effective number of pawn structures. King wings are current files, not castling history. Profiles describe where preparation ends, not the middlegames that follow.

### Effort and value

The full report weighs what your preparation earns against the moves it takes to know. Values are in score points per 1,000 games with that color, as in the [edge ledger](#where-your-edge-comes-from).

- **Chapters by edge per move.** A chapter's edge is its entry probability times its delta, and its moves are the own moves it records that games reach. A move recorded in several chapters counts in each, since you study it in each. Chapters near the bottom earn little for what they ask you to remember.
- **Lines to consider pruning.** Dropping one of your moves leaves preparation at that position, so it loses `move reach * gain`, where the gain is the repertoire score after the move minus the database score of the position. It also removes every move of yours that can only be reached through it: the positions it dominates, with transpositions accounted for. Each row is such a move with everything it dominates, ranked by value per move, showing groups of at least three moves. The rows never overlap, because a smaller group inside a listed one is not listed again. Moves that lose score on their own belong to the vulnerability tables instead.
- **Valuable moves you rarely play.** A move reached with probability `p` per game is missing from your last 100 games with probability `(1 - p) ** 100`. Ranking `reach * gain` by that chance favors moves worth a lot that come up about once per hundred games: common moves are practised in play, and very rare ones are worth little.

The summary's preparation table adds one line: how many of your moves come up less than once in 1,000 games, and their share of the edge. Thinly sampled moves are left out of both lists.

### Engine view

When you have [imported engine evaluations](usage.md#engine-evaluations), the reports add an engine's view beside the database's. Each position uses the deepest Stockfish evaluation in the Lichess export, first line, from your side, converted to an **expected score** with the Lichess win-chance curve `1 / (1 + exp(-0.00368208 * centipawns))`, the curve whose inverse gives the reports' centipawn equivalents. Mate counts as 100% or 0%. Engine and database scores are then on the same scale; pawn figures beside an average convert it back on the same curve.

- **Where preparation ends.** The summary, the full report and each chapter page give the engine's average expected score at the positions where preparation ends, beside the database score over the same games. The full report's chapter table adds it as a column, with its coverage when some positions have no evaluation. A database score above the engine's means opponents there go wrong in practice; one below means positions better than they play.
- **Your moves the engine questions.** A move's engine loss is the expected score before it minus after it. 5, 10 and 15 points match Lichess's inaccuracy, mistake and blunder thresholds (0.1, 0.2 and 0.3 in winning chances). Moves of at least an inaccuracy are listed with their database move gain, so you can see which still score well in practice.
- **Opponent mistakes.** Replies that raise your expected score by at least 5 points, with your score after them.
- **Where the database and the engine disagree.** The places preparation ends whose database and engine scores differ most, weighted by how often games end there.
- **Free transpositions** also show the engine loss of the transposing move: how much worse it is than the engine's best move after the reply.

Each chapter page has its own engine view with these tables, for games entering the chapter.

Evaluations exist only for positions someone analyzed on Lichess, so rare positions deep in a line often have none. They stay missing. Where preparation ends after such a reply, the position before the reply bounds it from below, because the engine evaluates that position with the opponent's best reply; the reports state how much of each average rests on evaluations and how much only on these floors, and averages use evaluated positions only. Engine figures never change a score, a ranking or a move choice.

### Opening names

Names and ECO codes come from the bundled [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings) dataset, the same list the Explorer uses, so naming a position needs no extra request. At a named position every arrival takes its name. At an unnamed position, each route keeps the last name it passed and its own probability. For example, an unnamed position reached with 1% probability through the Alekhine and 20% through the Vienna counts 1% for each opening respectively, not 21% for both.

The full report has an opening table for each color with every opening reached. Openings with the same name under several ECO codes share one row. An opening's **reach** is first arrival at that name or a more specific variation of it, where a variation's name extends the opening's at a colon or comma (Sicilian Defense: Accelerated Dragon is part of Sicilian Defense). Openings overlap, so their reach does not add up. Opening scores, baselines, deltas, depth and gap reach use the same first-arrival weights. Opening tables use the repertoire you actually play, not chapter alternatives.

The **opening source** under each line shows the last name carried by the largest share of arrivals, followed by that share when other names also contribute. Position rows combine every route; move rows count only arrivals through that move; chapter tables follow the chapter's own moves.

### Opponent ratings

The Explorer's average rating describes the player who made the move. At your turn, the opponent rating comes from their previous move's row in the parent table. At their turn, it is the game-weighted average over their replies. White's starting position has no previous opponent move and shows n/a.

Chapter ratings are averaged over the positions where preparation ends in that chapter. **Rating Δ vs parent** on reply rows is the reply's average rating minus the average over all replies at that position.

Ratings describe the database games at that point. They never change a score or a ranking, and there is no repertoire-wide average.

### Correlations

The full report includes two reach-weighted correlations. Both are point estimates without intervals, exclude sparse and unresolved data, and describe association, not cause.

**Prepared depth and gain.** Each of your moves is one observation, weighted by how often it is played. Its depth is the expected number of prepared moves after it, and its gain is the repertoire score after it minus that move's database score. The report shows weighted linear and rank correlations and the weighted slope of gain on depth, with unweighted results and a check that excludes zero depth in collapsed details.

**Opponent rating and score.** The full report compares replies within the same parent position, overall and separately for prepared replies, unprepared replies and replies with at least 1,000 games. Running `repertoire rating-correlations` on its own also writes `reports/comparisons/opponent-rating-score.md`, which adds chapter scores and deltas against their chapters' opponent ratings, grouping chapters that share positions. Both describe the database cohorts, not what would happen against a particular rating.

## Strengths and vulnerabilities

### Gain and drag

- **Opponent reply drag** is the score before the reply minus the score after it. **Weighted drag** multiplies it by the reply's reach (parent reach times reply probability) and ranks the opponent replies that cost you most. Prepared replies use the score of your continuation; unprepared replies use the reply's database score.
- **Own move drag** is the database score of the parent position minus your score after your move. It measures how your move and the preparation after it compare with what players do there in general. The strengths section ranks the reverse, your gain.
- **Gain** splits into the **move gain**, your move's database score minus the position's, and the **preparation gain**, your score after the move minus the move's database score. The two add up to the total.

Tables for positions where you have a prepared move are never fetched, because your move is forced and the score does not need them. The position's database score is therefore pooled from the opponent move rows that lead to it, or its own table at the starting position, and your move's database score is the whole table after it. Where a position is reached by one move order, these are the same games its own table would hold. At a transposition, the position's score counts only the routes in your studies, and the table after your move counts every move order.

Weighted values are shown in score points per 1,000 games, so a weighted drag of `0.0317%` reads as `0.32`. The summary ranks your moves by reach times gain or drag; the full report ranks them by the gain or drag itself. Intervals on gains and drags combine every table the comparison depends on. **1 in N games** is `1 / reach`: how many games with that color (or entering the chapter) pass before a position comes up once on average.

Chapter rankings are among games that enter the chapter. For a chapter whose moves lost to an alternative, they describe the alternative, not the repertoire you play.

### Where your edge comes from

Gains overlap: the gain of `1.e4` includes everything after it. The edge ledger splits the delta into parts that add up instead. Each of your moves contributes its **edge**, `move reach * move gain`, where the move gain is the database score after the move minus the database score of the position it is played from. A move counts only its own step, so the rows add up, and the whole table answers which of your decisions earn the delta. The full report also groups the edge by your move number and lists the moves that cost the most.

The moves alone add up to slightly more or less than the delta, for two reasons, each shown as its own row:

- **Move orders at transpositions.** A prepared position's database score pools the opponent moves leading to it by game count, but your repertoire arrives through each move order in its own proportion. Rare move orders often score differently: after 1.e4 d6 2.d4 Nf6 3.Nc3 g6 4.g3 Bg7, games arriving by the Modern order (1...g6) may score better than the Pirc order, and your repertoire may send more games through it than the database does. Each arrival adds `its reach * (pooled score - score of its own move row)`. In a chapter, the row also covers move orders the chapter never uses and an entry where you are to move, whose baseline is its own table.
- **Theory leaves**, which are scored with their own table, while the move into them is compared with its row in the parent table. Finished games (checkmate, stalemate, insufficient material) are scored exactly and form a third row when they occur.

Together these reproduce the delta exactly; the build checks this for every color and chapter. Values are in score points per 1,000 games, so the total equals the delta times ten. Thinly sampled moves count in the totals but are left out of the ranked tables, as elsewhere.

### Position contributions

The strengths section also ranks every reached prepared position and unprepared reply by **contribution**, reach times score. Prepared positions use your score after them; unprepared replies use their database score. Positions on the way to others count too, so after `1.e4` the whole White score passes through one row. Contributions overlap along a game and show where the score flows, not where it improves.

The preparation analysis also keeps a ledger of where each game ends. Each game ends once, so those contributions add up to the score.

### Sparse evidence and limits

Strength and vulnerability tables, and their summary highlights, leave out rows flagged sparse in the position, its parent or the position right after. A merged unprepared position is left out if any of its arrivals is sparse. Rows are filtered before display limits are applied, and the JSON keeps every row, including negative drags. Sparse evidence still counts fully in the score. Game counts in the parent table's row describe how often a reply is played; a prepared continuation's score can rest on different, deeper samples.
