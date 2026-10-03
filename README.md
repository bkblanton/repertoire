# Repertoire score

Python 3.12+ application implementing `design.md`, managed with uv. It estimates a population-based expected chess score under a fixed repertoire policy. It is not an individual performance prediction or an engine evaluation.

## Run

```powershell
uv sync
$env:LICHESS_TOKEN = (Get-Content -LiteralPath 'C:/path/to/lichess_token.txt' -Raw).Trim()
uv run repertoire-score inspect 'C:/path/to/repertoire.pgn' --color white --output reports/data/inspection
uv run repertoire-score run 'C:/path/to/repertoire.pgn' --color white --config configs/white.json --output reports/data/white
Remove-Item Env:LICHESS_TOKEN
```

The token is read only from `LICHESS_TOKEN`; it is never written to reports or cache files. Network errors abort the run and do not become zero-game evidence. Cache files permit resuming a run. Cached responses do not expire automatically; use `--refresh` for a fresh database snapshot. To recompute without network or a token:

```powershell
uv run repertoire-score run 'C:/path/to/repertoire.pgn' --color white --config configs/white.json --output reports/data/white --offline
uv run pytest -q
```

If Windows prevents uv from accessing its default cache or managed Python folder, set `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` to writable directories before running uv. A compatible installed interpreter can be specified with `uv sync --python C:/path/to/python.exe`.

## Configuration

All ratings are selected explicitly: 0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500. Speeds are blitz, rapid and classical, with the full supported date range. The API covers indexed rated games, not every game ever played on Lichess. These defaults follow the [official Explorer endpoint specification](https://github.com/lichess-org/api/blob/master/doc/specs/tags/openingexplorer/lichess.yaml).

Configuration is a JSON object:

```json
{
  "policy": {"canonical four-field FEN": "e2e4"},
  "chapter_regions": {
    "chapter-id": {
      "anchors": [{"path": ["e4", "e5", "Nc3", "Nf6", "g3", "Nc6"]}],
      "description": "Quiet System after both black knights develop"
    }
  },
  "filters": {"ratings": "0,1000,1200,1400,1600,1800,2000,2200,2500", "speeds": "blitz,rapid,classical", "since": "1952-01", "until": "3000-12"},
  "exclude": []
}
```

Policy keys are canonical positions: piece placement, turn, castling and legal en passant, without counters. Values are one UCI move or a move-to-weight map summing to one. Without an explicit override, own-move conflicts choose the first recorded move: PGN main variation before side variations, and earlier chapters before later chapters. Conflicting alternatives are never averaged or assigned simultaneous probability one. All PGN variations remain available as repertoire content; annotations are not instructions and do not remove lines.

Every chapter remains in the chapter report, including alternatives excluded from the overall policy. For each chapter comparison, its first recorded own moves take precedence and the overall policy applies elsewhere. This retains compatible preparation split across multiple chapters. Score, baseline, expected prepared depth, entry probability, transitions from that chapter, and vulnerabilities all use that same comparison policy. Alternative rows are labeled; a separate overall-policy region reach shows how often the selected overall repertoire enters that region. Shared region reach does not imply that the alternative own move was selected. Reordering chapters changes overall priority without discarding the alternatives' comparisons. Saved JSON records the exact policy overrides.

Default runs keep just `reports/report.md` and `reports/summary.md` as readable reports. Supporting files live under `reports/data/`. Custom score output prefixes remain supported; a prefix inside a `data` directory renders the two Markdown files in its parent.

The supplied configurations use **chapter regions**. Each chapter has explicit subject anchors, accepting canonical FEN strings or path objects containing SAN/UCI moves and an optional `root_fen`. Its region contains those anchors and all descendants through moves recorded in that chapter, including positions shared with other chapters. Repeated introductory moves before the anchors are excluded. Comments are not executed or automatically interpreted as membership rules. The exact subject anchors and region membership are recorded in the report.

Entry probability means **chapter reach probability**: first arrival anywhere in the region before the model stops, counting each modeled game once per chapter. A path may bypass an early anchor and enter at a shared descendant through a later transposition. The program finds possible first arrivals by walking the resolved repertoire from its roots and stopping on region entry, then propagates first-arrival probability mass. It does not simply sum unrestricted position frequencies or discard a late entry because it descends from an earlier one. After entry, scoring follows the complete merged repertoire, including other chapters' continuations. Those continuations do not automatically become members of the source chapter's region.

The Quiet System starts at `1.e4 e5 2.Nc3 Nf6 3.g3 Nc6`, Mieses at `1.e4 e5 2.Nc3 Nf6 3.g3`, and Paulsen at `1.e4 e5 2.Nc3 Nc6 3.g3`. Several other shared opening subjects have explicit earlier anchors; other chapters retain their previous opening anchors and now include shared descendants. These are documented definitions of where preparation becomes relevant, not exclusive opening classifications. Adjust `chapter_regions` to change a subject boundary. Updated PGN descendants are included automatically on rerun; missing anchors fail validation.

For compatibility, `entries` still accepts a map from chapter IDs to exact entry positions or paths. A chapter cannot have both `entries` and `chapter_regions`. Without either, all first chapter-unique positions become anchors, with chapter-owned descendants defining the region. If none can be entered under that chapter's policy, the first opponent reply on its mainline (or its PGN root) is used. Explicit subject anchors are preferable for comparing alternatives from a common position, such as `1.e4 e6` for both Advance and Tarrasch French chapters; automatic entries can describe different conditional subtrees.

Full reports also include directed **chapter transition probabilities**: conditional on first entering a source chapter, how often does the model reach the destination at or after that point? Shared or simultaneous entry counts. The JSON retains all ordered chapter pairs, including zeros and undefined results. This is different from an unordered intersection, because the destination may have been visited only before the source. Overlapping chapter frequencies and transition rows are not additive. The model does not follow deviations through unknown positions to possible later re-entry.

`exclude` can contain chapter IDs or strings of the form `chapter-id:canonical-position:uci` to omit a branch and its descendants from that chapter. `root_weights` may map canonical root positions to weights summing to one. Disconnected custom-FEN chapters do not receive absolute reach probabilities without a connecting route or explicit root weights. Reachable cycles fail with a position sequence for diagnosis.

## Statistics and missing evidence

Our selected moves have probability one, or configured mixture weights. Opponent moves use their share of all games at that parent, including deviations. Leaves use position results; deviations use parent move-row results. Legal unobserved moves remain in the posterior model. Valid residual results form a separate no-recorded-continuation bucket.

The default stopping-score prior is Dirichlet(0.5, 0.5, 0.5), in owner win/draw/loss order. At opponent nodes its total strength is divided across all legal moves and any observed residual bucket. Joint move-by-result sampling preserves the dependency between move probability and deviation score. The same samples are reused across all paths into a position.

Default Monte Carlo settings are 2,000 simulations and seed 20260928. Change these with `--simulations`, `--seed`, and `--prior W D L`. Reports include sensitivity to symmetric priors of 0.1 and 2 per result category. `--sparse-threshold` defaults to 30 observations. `--tolerance` defaults to a 1 percentage point interval-width target; it is a reporting flag, not a pruning threshold.

Zero-data stopping scores are unresolved in empirical results. A zero-data opponent distribution stops unresolved at that position, without uniform play or parent-score fallback. Reports give conditional bounds [resolved contribution, resolved contribution + unresolved mass]. Sparse sensitivity assigns all flagged stopping events any score from zero to one. These are conditional sensitivity bounds, not credible intervals.

The approximate 95% interval and posterior statistics retained in JSON use explicit prior completion for unresolved scores. They must be read alongside unresolved mass and conservative bounds. They do not account for all dependence from games appearing in multiple position aggregates, selection bias, or population mismatch. Posterior unresolved mass includes prior probability assigned to legal but unobserved moves and may exceed empirical unresolved mass. Main tables label the raw empirical estimate as **Repertoire score** and omit the posterior mean.

## Outputs and implementation

Each color's section includes only that color's standard starting-position baseline. The overview and summary show both colors and their respective deltas, in percentage points. Both also include a combined row with 50% weight for White and 50% for Black, provided both use matching Explorer filters and standard starting positions. The row averages scores, starting baselines and prepared depths; chapter counts are summed across colors. Each overview row includes an **Elo equivalent**: `400 * log10(score / (1 - score)) - 400 * log10(baseline / (1 - baseline))`. This translates the modeled score edge to an Elo scale; it is not a measured rating gain. One-color reports omit the combined row. The reference uses ordinary database play under the same Explorer filters before forcing repertoire moves, with counts and retrieval provenance recorded in JSON. It does not alter the repertoire calculation. JSON retains both color references for compatibility.

Every table containing expected scores also shows their direct **centipawn equivalents (CP)**, including baseline, before/after, stopping-score, and score-interval columns. Use `C(p) = ln(p / (1 - p)) / 0.00368208`, the inverse [Lichess score curve](https://lichess.org/page/accuracy). For example, Black's 52.50% score is about +27 cp from Black's perspective. **CP delta** is `C(after) - C(before)`, or score CP minus entry/starting-baseline CP. Negative changes indicate a worse score; positive drag has the opposite sign. Scores include half a point for draws. These are human-results conversions rather than engine evaluations. Convert weighted mixtures after averaging their scores, and score intervals by converting both endpoints. Reach, frequency, and contribution percentages are not expected scores. Missing scores remain unresolved and conversion at 0% or 100% is unavailable.

Each chapter also includes an empirical entry baseline and the repertoire score minus that baseline in percentage points. Reports display score deltas, gains, drag, and weighted contributions with `%` in the value rather than `pp` in the heading. For example, 55% minus 50% displays as `+5.00%`, a five percentage point difference rather than a relative change. JSON retains the existing percentage-point units and field names. Multiple entries use their normalized first-entry probabilities, matching the chapter's empirical repertoire calculation. Database sample counts are not used as mixture weights. JSON retains the component weights, scores, counts and provenance; the table shows one weighted baseline per chapter. Missing entry evidence remains unresolved. A positive difference means higher modeled score, not demonstrated causal improvement or statistical significance.

**Expected prepared depth** is the expected number of remaining repertoire-owner moves before leaving theory or reaching a theory leaf or terminal outcome. Each selected own move contributes one; opponent moves contribute no unit themselves and weight subsequent prepared moves by their empirical frequencies. An available own move at the starting position is included, but previous moves and entry itself earn no bonus. There is no cutoff, discount, or tunable coverage parameter. Overall depth starts at the repertoire root; chapter depth uses the same first-entry mixture as its score and baseline and follows the complete merged repertoire thereafter.

The evaluator computes depth backward through the DAG: an own-move node has depth `1 + weighted child depth`, an opponent node has depth `sum(reply probability * child depth)`, and a deviation or leaf has depth zero. Shared prefixes, duplicate lines, and transpositions do not create additional per-game moves. Deeper or broader preparation earns credit according to its probability of being used. Missing leaf outcome counts do not affect depth; missing opponent distributions preserve lower/upper bounds using the finite remaining repertoire, and missing first-entry weights leave chapter depth unresolved. Bounds are conditional on the resolved frequencies, not confidence intervals. JSON stores `prepared_depth` under `overall` and each chapter's `score`; all current tables and summaries display depth in own moves. Historical JSON without the metric displays `not calculated` until reanalyzed.

**Equivalent gap reach** summarizes recurring first unprepared positions: `R = sqrt(sum(p_i ** 2))`, where `p_i` is the probability of first reaching canonical board `i` with no prepared own reply. Combine all exact transpositions into one gap before squaring. `R ** 2` is the probability that two independent modeled games first encounter the same gap; `R` is the reach of one gap with the same repeat probability. Lower values indicate less concentrated or less likely gaps. Probabilities use the full scope, without conditioning on reaching a gap, omitting sparse rows, or imposing a depth cutoff.

The cache-only character analysis follows selected own moves and cached opponent reply frequencies. After a prepared endpoint, it uses that endpoint's cached reply table to identify the first unanswered positions. A reply that transposes into a prepared board continues through the merged repertoire. It never fetches unprepared child tables. Terminal games produce no gap. Missing or zero response tables, unnamed residuals, and closed canonical cycles retain unresolved probability, with conservative bounds rather than a falsely exact value. Cyclic components with exits are solved as absorbing Markov chains. Scores and prepared depth keep their existing stopping rules.

Both consolidated files show each color's equivalent gap reach. Chapter tables include **Equivalent gap reach after entry** and **Weighted gap reach contribution**, calculated as chapter entry probability times the conditional value. The conditional distribution uses the same normalized first-entry mixture and comparison policy as the chapter score. These weighted values are on the full-repertoire probability scale, but are not additive: chapters and gap boards can overlap, and equivalent gap reach is nonlinear. Character JSON retains every canonical gap probability, resolved/terminal/unresolved mass, bounds, and conservation checks under `gap_coverage`.

- `.json`: complete event ledger, overall and chapter scores, sensitivity, sample counts, prior influence, policy diagnostics, and manifest with input hash, filters, cache keys, and retrieval timestamps.
- `report.md`: the main consolidated report. Scores, the most common reached positions within each color section, vulnerabilities, strongest moves and position contributions, reuse, reply predictability, position profiles, and depth correlations appear together. Each chapter has one expandable analysis with exact entry routes. Compact chapter links preserve every source without repeating long lists of names.
- `summary.md`: a readable overview with both scores and baselines, preparation/exposure highlights, the largest unprepared pressure point per color, and all chapter entry probabilities, baselines, scores, deltas, and depths. It links into the full report and is no longer framed as a Lichess study description.
- `data/`: supporting JSON results, companion metrics, correlations, inspection diagnostics, and the output registry. These are inputs to the consolidated renderer, not additional readable reports. Color-specific and analysis-family Markdown are no longer generated. Hypothetical comparisons are kept separately in `comparisons/`.
- `.inspection.json`: parsed chapters, conflicts and entry candidates.

All Markdown generation is automated. To change presentation without fetching evidence or recalculating scores, render the saved results:

```powershell
uv run repertoire-report reports/data/white.json reports/data/black.json --require-complete
```

This command writes `report.md` and `summary.md` without changing JSON results, reading the Explorer cache, or making network requests. `--output` and `--summary` choose destinations; `--top` (default 10) and `--chapter-top` (default 5) control displayed ranking lengths. `--position-top` (default 20) controls each prepared-position and unprepared-reply ranking for every color and chapter. Categories are filtered before the limit so frequent unprepared replies remain visible. Color rankings use reach under the overall policy. Each chapter's expandable analysis begins with common-position tables using reach conditional on first entry under its comparison policy, including alternative chapters. All rankings combine transposed routes and include prepared endpoints and the first unprepared opponent replies. Unprepared reply reach is computed from cached parent response tables, without querying the reached child positions; transposed replies share one row with combined reach and chapter context. Boards immediately before a prepared own move are omitted from the display in favor of the board after that reply. Own-turn endpoints with no reply are labeled. The standard starting board is also omitted. Prepared rows show the repertoire continuation score and cached position game count; unprepared rows show the cached parent-move score and count (or cached position outcomes for recorded endpoints). Every score has its CP equivalent; rows also show games and local opponent ratings. Transposed unprepared scores use modeled incoming reach weights; parent counts are pooled and marked with † because samples may overlap. Games are not the sample size of the full continuation score. Rows can overlap along a game and must not be summed. Exact FENs and all reached positions are stored in each character JSON; this section does not make API requests. Complete event ledgers and rankings remain in JSON. The former most-frequently-used-own-decisions table is no longer displayed; reuse metrics and decision data remain available.

Companion report hashes, input hashes, colors, and filters must match. Stale analyses cause the explicit rendering command to fail rather than silently mix snapshots. `--require-complete` also requires all four companion analysis families and matching depth correlations. Without it, missing analyses are clearly marked pending. A changed or missing source PGN produces a prominent saved-snapshot notice; rendering an archived result does not pretend it describes the current study. Historical improvements and hypothetical comparisons remain separate because they use different policies or snapshots.

Scoring and each analysis command refresh the consolidated files automatically using the latest registered result per color. During a refresh, stale companions are omitted and marked pending until their analysis is regenerated. The data folder's `.report-index.json` records the latest score filenames; standard `white.json` and `black.json` are discovered on the first run. `run-repertoires.ps1` runs scores, vulnerabilities, preparation, character, ratings, and correlations in sequence, then requires complete consolidation. It preserves the persistent cache and fetches only missing tables unless explicitly refreshed. Source PGNs and live Lichess studies are never edited.

`graph.py` parses and resolves the shared graph and constructs chapter regions and their entry frontiers. `explorer.py` handles evidence and caching. `model.py` prepares evidence and samples posteriors. `evaluate.py` performs backward value and forward probability passes. `depth.py` computes expected prepared depth. `transitions.py` computes directed chapter transitions. `report.py` renders outputs. The evaluator has no network dependency.

## Correlations and confidence intervals

Analyze the saved chapter estimates, with approximate 95% cluster-bootstrap confidence intervals:

```powershell
uv run repertoire-correlations reports/data/white.json reports/data/black.json --output reports/data/depth-delta-correlation
```

This writes correlation JSON beside the score data and refreshes the correlation section of the consolidated report, including ordinary, rank, reach-weighted, baseline-adjusted and alternative correlations. Defaults are 20,000 bootstrap replicates and seed 20260929; `--bootstrap-samples` and `--seed` control reproducibility and numerical precision. To also regenerate the standalone interactive scatterplot, use `uv run --with plotly repertoire-correlations reports/data/white.json reports/data/black.json --output comparisons/depth-delta-correlation --plot`. Plotly is optional and is not added to the program's core dependencies.

Clusters are connected components of chapters whose reachable post-entry theory shares canonical positions, using the selected policy and complete merged graph. Whole clusters are sampled with replacement, keeping each chapter's depth, score, baseline and reach together. Pooled resampling is stratified by color; regression adjustment and ranks are recalculated per replicate. The original number of clusters is retained per color, while the number of sampled chapter rows may vary. Each interval uses the 2.5th and 97.5th percentiles. Undefined replicates are counted, and an interval is withheld if more than 1% are undefined.

These are exploratory confidence intervals across chapter groups, conditional on saved model estimates. They do not include finite Lichess evidence uncertainty, repertoire-selection bias or all cross-group dependence. With only 13 White and 15 Black groups in the current studies, nominal coverage is approximate. For a fixed set of saved chapter values, the correlation itself is an exact descriptive calculation; these intervals describe variation under sampling similar groups, not uncertainty in that arithmetic. Baseline adjustment is an optional sensitivity comparison, not a correction that invalidates ordinary delta. The generated report documents the assumptions and cluster membership and links the statistical references.

No Lichess requests or token are needed. The PGN files are read only to reconstruct dependencies and must still match the saved input hashes. Correlations run automatically in `run-repertoires.ps1` or through this separate command. Presentation-only `repertoire-report` combines the matching saved correlations without recalculating them.

## Vulnerability reports

Generate overall and chapter rankings from saved scores and cached parent-position tables:

```powershell
uv run repertoire-vulnerabilities reports/data/white.json reports/data/black.json
```

This defaults to **cache-only** and requires no token. If own decision positions were not needed by an earlier score run, add `--fetch-missing` with `LICHESS_TOKEN` set. Only missing own-parent tables are fetched, once per canonical position and filter set. Every candidate reply or alternative is read from its parent's cached move rows. Candidate child endpoints are never requested for screening. Existing score evidence must match the saved report's cache keys and retrieval timestamps; a changed PGN or refreshed evaluation cache requires regenerating scores first.

Outputs are `data/white.vulnerabilities.json` and `data/black.vulnerabilities.json`. The consolidated report has separate tables for unprepared opponent replies, prepared opponent replies, and selected own moves. Opponent replies rank by weighted drag; own moves rank by the direct deficit against the parent database score. Each category is filtered before its display limit. Set displayed ranking lengths with `repertoire-report --top` and `--chapter-top`; JSON always retains all rankings and signed comparisons. `run-repertoires.ps1` generates this analysis after scoring both colors, fetching missing parent tables unless `-Offline` is set. Rendering alone does not recalculate vulnerabilities.

**Opponent reply drag** is `repertoire value before reply - value after reply`, in percentage points. **Weighted drag** multiplies that local drop by `parent reach * reply probability` and determines opponent rankings. Prepared replies use the full merged continuation; deviations use the parent move row's empirical score. **Our move drag** is `ordinary parent database score - repertoire continuation score after our move`, expressed in percentage points without multiplying by reach. The strengths section ranks the opposite difference, `repertoire continuation score - parent database score`. All before/after comparisons also show the signed CP delta after converting both scores. Our move's historical popularity is never applied. The benchmarks differ, so the two rankings remain separate. Historical same-table alternatives remain in JSON as separate screening information with their sample sizes; they compare database outcomes and are not substituted for prepared continuation scores. They are not evaluated replacement policies or recommendations, and the maximum observed score can exaggerate sampling noise.

Reach sums incoming mass across exact-position transpositions. Each position/move is counted once per ranking. Chapter rankings use the same normalized first-entry mixture and comparison policy as chapter scoring. Opponent weighted drag and all reported chapter reach are conditional on chapter entry. Own drag is a direct score subtraction at its parent. JSON also retains reach-weighted own comparisons and comparisons weighted again by chapter entry probability. For alternative chapters it is counterfactual, not an impact on the selected overall repertoire. A move that enters the chapter belongs to the overall or upstream ranking. Representative lines are legal route labels, not exclusive historical sequence probabilities. Reports retain the owner's overall baseline and delta, and each chapter's weighted entry baseline, score, delta and expected prepared depth for context.

Positive drag highlights below-reference branches. Negative signed changes are preserved in JSON. Nested lines and overlapping chapters must not be summed, and these screening measures do not decompose the overall baseline delta or estimate causal improvement. No-data comparisons remain unresolved rather than becoming zero scores. Strengths and vulnerability tables, and their summary highlights, omit rows flagged sparse in their local, parent, or immediate endpoint evidence. Filtering happens before each display limit. Position contributions exclude missing or sparse local counts; merged unprepared positions are omitted if any parent-row arrival is sparse, even when pooled counts exceed the threshold. JSON retains all rows, and the score and probability models still include all evidence under the existing sparse threshold (normally 30 games). Opponent move counts describe reply frequency; prepared values can depend on different downstream samples. Residual non-move stopping buckets are validated but not ranked as chess moves. Validation reproduces saved overall and chapter scores, checks probability conservation, and checks that signed opponent deviations, including residuals, balance around their parent means.

## Validation

Sanity checks enforce conservation at every evaluated node and reproduce the root value from weighted stopping contributions. Tests cover forced moves, beneficial and harmful deviations relative to an explicit leaf baseline, duplicate chapters, shared leaves, transpositions, own-move conflicts, sparse and missing evidence, residual buckets, inconsistent responses, first-entry weighting, cycles, color reversal and fixed-seed reproducibility.

The supplied PGN source files are never modified. `prefetch.py` can warm the cache before policy selection, but normal runs fetch all required data themselves.

## Repertoire character: reuse, reply predictability and position profiles

All report tables containing individual lines now include linked chapter attribution. A recorded move lists its exact position/move providers, including every shared source. A position lists the chapters containing that canonical board. An unprepared reply is labeled unprepared and lists its parent chapters as context; an unrecorded move that transposes into preparation lists the destination chapters. Representative routes may combine chapters. These relationships are stored separately from the older parent-membership `chapters` fields in JSON.

To refresh attribution in saved score, vulnerability, preparation and character reports without recomputing estimates or querying Lichess:

```powershell
uv run repertoire-attribution reports/data/white.json reports/data/black.json
```

The command validates source hashes and companion-report hashes before writing, preserves all numerical results and updates companion source-report hashes. Normal analysis commands also populate attribution automatically.

```powershell
uv run repertoire-character reports/data/white.json reports/data/black.json
```

This cache-only command writes `data/white.character.json` and `data/black.character.json`, then refreshes the consolidated report and summary. It also runs automatically in `run-repertoires.ps1`. No token or API requests are needed. Source PGN hashes and cached scoring evidence must match the saved scores. Overall and chapter scores and expected prepared depths are independently reproduced before writing these metrics. Chapter scopes retain weighted first-entry mixtures and chapter-local policy, including unselected alternatives.

**Repertoire sharpness:** the report and summary show sharpness on a 0%-100% scale alongside every repertoire score, including overall and chapter results, common positions, strengths, vulnerabilities and position contributions. Reply comparisons show before and after sharpness. Preserve owner-relative win/draw/loss probabilities through the same recursive policy and stopping rules as repertoire score, then calculate `400 * (W + D/4 - (W + D/2)**2)`. This is normalized variance of the eventual 1 / 0.5 / 0 game score, with no tuning parameter. Own choices inherit the selected continuation; opponent replies average child WDL by observed frequency. Unprepared replies use cached parent move rows, endpoints use cached position results, and terminal results are exact. Exact transpositions reuse their continuation. Average WDL before calculating sharpness, including weighted chapter first entries and the combined 50/50 color mixture; do not average child sharpness. WDL must sum to one and W + D/2 must reproduce the saved score. Each character scope and reached position stores its WDL, unresolved probability and sharpness in `outcomes`. Prepared positions use recursive WDL; merged unprepared positions mix their cached parent-row WDL by modeled incoming reach before calculating sharpness. Individual unprepared-reply comparisons retain their specific parent-row WDL rather than the merged position mixture. Missing results or entry weights remain unresolved or unavailable. All evidence remains included, including sparse samples; no new API request or prior is used. Sharpness is 100% for 50% wins and 50% losses, 10% for 5% wins / 90% draws / 5% losses, and 0% for a certain result. It describes outcome volatility while following preparation, not tactical difficulty or vulnerability to forgetting a move.

**Expected reuse:** for a distinct own position/move decision with modeled encounter probability `p`, `N*p` is its expected encounters in N independent games and `1-(1-p)^N` is its probability of being seen at least once. Summing these yields total encounters and distinct decisions encountered; their difference gives repeat encounters. Exact transpositions and duplicate chapter providers share one decision. The curve defaults to 10, 50, 100 and 500 games; customize it with `--games`. Chapter curves count games entering that chapter, not all games. The denominator includes only selected decisions with positive empirical reach. Exposure is not memory retention.

**Reply predictability:** entropy over observed named opponent replies at each active decision, with effective replies `2^H`. The scope summary exponentiates the mean entropy weighted by position reach and the recorded continuation fraction. Unrecorded continuation mass is reported as missing coverage, never invented as another chess move. The accumulated information in bits per game is shown separately from the average per decision. Zero observations produce unavailable predictability. Sparse samples are flagged using the existing scoring threshold without excluding them. Leaves are not queried for further replies.

**Position profiles:** aggregate board features at the boundary of preparation, using prepared leaf boards and the board after each unprepared opponent reply. Show queens, current king wings, bishop pairs, isolated/doubled/passed pawns, isolated d-pawns and exact pawn skeleton frequencies. Effective skeleton count is `2^H` over the weighted skeleton distribution. JSON also includes material, pawn and rook counts and profiles conditional on stopping type. Current king files are not treated as proof of castling history. Unresolved opponent distributions stop at their known board and are flagged; downstream reuse is unknown. These describe preparation boundaries rather than eventual middlegames or personal outcomes.

The consolidated report includes chapter comparison tables and per-chapter details. Its `--top` and `--chapter-top` options control displayed ranking lengths; JSON preserves every row. Descriptive empirical estimates have no sampling confidence intervals. The three measures do not assign a combined quality score or an arbitrary depth discount.

## Stopping outcomes and score contributions

After scoring and vulnerability generation, run:

```powershell
uv run repertoire-preparation reports/data/white.json reports/data/black.json
```

This command is cache-only. It writes `.preparation.json` beside each score result and refreshes the consolidated report and summary. It also runs automatically in `run-repertoires.ps1`. Missing candidate evidence is reported, never fetched silently or treated as a zero score.

The strengths section appears overall and per chapter. It shows our largest positive continuation-score differences against the parent database score, followed by **all reached prepared positions and unprepared opponent replies** ranked by contribution: reach times score. Prepared positions use their full repertoire continuation score; unprepared replies use cached parent-row outcomes or recorded endpoint outcomes. Exact boards combine all transposed arrivals. Boards immediately before guaranteed own replies and the standard starting board are omitted, as in the common-positions section. Intermediate positions count, so White's `1.e4` carries the entire repertoire score. These contributions overlap along a game and must not be summed; they show score carried through a position, rather than incremental improvement. Chapter reach and contributions are conditional on entry. Sparse and unresolved positions are filtered before display limits. Pooled parent counts may overlap and are marked with a dagger. No scoring or new Explorer queries are needed to render this ranking from the saved character data.

The preparation JSON retains a separate stopping-outcome ledger and baseline-relative contributions for analysis. Each modeled game stops once, so its complete stopping contributions still reproduce the resolved score without repeatedly counting intermediate positions.

The summary's own-move highlights rank by **reach times local continuation gain or drag**, after the existing sparse filter. Full-report own-move rankings retain their local comparisons. Every highlighted gain includes the later prepared continuation, so these weighted comparisons overlap and must not be added. Line tables also show **Avg games per encounter**, `1 / reach`, for independent modeled games. In chapter tables this means games that enter the chapter; in overall tables it means games with that color. Zero reach is never encountered under the policy, rather than a finite waiting interval.

`repertoire-preparation` automatically saves a **prepared-depth distribution** for each color and chapter. It propagates probability over both canonical board and elapsed own-move count, so paths that transpose into a shared board retain their different remaining-depth histories. The full report shows cumulative probabilities of preparing at least each depth and exact-depth endings at prepared endpoints, unprepared replies, and other stops. The survival probabilities from depth 1 sum to expected prepared depth. The summary links the full distribution and reports its median. Zero-data leaf outcomes do not obscure depth; missing opponent distributions retain finite structural bounds and are labeled unresolved. There is no depth cutoff or discount parameter.

The same analysis saves actual **first-entry route examples** under each chapter's comparison policy. It stops every root-to-entry path when it first reaches any chapter-region position, merges all arriving probability at the exact board, and keeps the most likely single route as an example. Entry-position weights include every first-arrival route; the separately displayed example weight covers just that route. Both are conditional on reaching any position in the chapter. Examples are validated as legal and cannot pass an earlier chapter position. These explanations preserve the existing transposition-inclusive chapter scores and reach; ordinary position-table lines remain representative board labels. Older results without these fields explicitly request a cache-only preparation refresh.

## Opponent rating contexts

```powershell
uv run repertoire-ratings reports/data/white.json reports/data/black.json
uv run repertoire-report reports/data/white.json reports/data/black.json --require-complete
```

Run ratings after the vulnerability, preparation and character commands. It reads only existing Explorer cache entries, writes `data/white.ratings.json` and `data/black.ratings.json`, and refreshes the same consolidated report and summary. The runner includes this step automatically. Source, score, supporting-analysis and cache hashes identify the evidence. Refresh ratings whenever a supporting analysis changes; strict assembly rejects stale ledgers.

Explorer `averageRating` is the move maker's rating. At our turn, use the previous opponent move's parent-table row. At the opponent's turn, weight the current response rows by game counts, including the Black repertoire's starting-board row where White is to move. The White starting-board row has no preceding opponent move and is labeled n/a. These are local position contexts; neither creates a whole-repertoire rating average. Transposed arrivals use modeled reach rather than database counts. Specific opponent vulnerabilities use the reply row; our selected moves and cached alternatives use the resulting board's opponent response rows. Unprepared child tables are never fetched.

Chapter score-evidence averages weight ratings once at the stopping outcomes. Entry-baseline context instead uses the chapter's first-entry mixture, including the incoming edges under its comparison policy. Local line ratings describe the exact position or move context. Chapter pawn groups can combine stopping evidence; whole-repertoire groups show only the individual example's rating. Missing ratings and residual outcome buckets remain unavailable, and partial coverage is disclosed. These descriptive ratings do not modify scores or rankings. There is no White, Black, combined-study or whole-study baseline rating average.


Opponent reply rows also show **Rating Δ vs parent**: the reply's move-maker average minus the parent's game-weighted opponent response average. Transposed replies combine these paired differences by modeled arrival reach. Missing comparisons remain unavailable; a current-position response average without a specific opponent reply to compare is marked n/a. Partial parent-rating and paired-arrival coverage are disclosed. This describes the reply cohort and does not change any score or ranking.

To add or refresh only these differences using the already matching rating ledgers and identical cached evidence:

```powershell
uv run repertoire-ratings reports/data/white.json reports/data/black.json --differences-only
```

Normal rating generation also includes these differences automatically. The differences-only refresh can use a saved score snapshot after its PGN changes, because it does not parse or rescore that PGN. The consolidated report flags the snapshot, and matching score, analysis and cache hashes remain required.

## Report readability and saved comparison refresh

`report.md` contains one score overview, per-color common positions and early chapter overviews, vulnerabilities, strengths, and expandable chapter details. `summary.md` shows up to five rows in each ranking: weighted own-move gains and drag, common unprepared replies, unprepared replies with the largest weighted drag, and position contributions. It also includes stopping reasons and expandable chapter tables. Chapter source links use compact W/B identifiers. Reach labels state whether the denominator is the repertoire or games after chapter entry; opponent reply frequency is conditional on the parent. Snapshot notices appear before scores, with scoring and evidence retrieval dates separately. Empty categories are omitted.

To update own-move comparison definitions from matching saved continuation analysis without parsing a changed PGN or requesting any data:

```powershell
uv run repertoire-vulnerabilities reports/data/white.json reports/data/black.json --refresh-saved
uv run repertoire-report reports/data/white.json reports/data/black.json --require-complete
```

This refresh checks score and character provenance, updates only vulnerability comparisons, and preserves all existing rating contexts while updating their supporting comparison hash. It does not rescore newer source edits; the report explicitly flags changed or missing PGNs. Normal analysis still requires the current PGN to match its saved score snapshot.
