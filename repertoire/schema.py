"""The shape of the saved score JSON (`white.json`, `black.json`) and the manifest every companion shares.

These TypedDicts document the format and give editors and type checkers something to check against; nothing
validates files against them at run time. Scores are owner-relative expected points between 0 and 1, and
`*_pp` fields are percentage points. Fields that cannot be resolved from the evidence are None.
"""

from typing import Any, NotRequired, TypedDict

from .status import Status

Position = str  # canonical four-field FEN: pieces, side to move, castling, legal en passant
# A JSON object read from or written to a saved analysis. Companion analyses and the rows the report
# renders are typed this way; the score file itself is described by the TypedDicts below.
JsonObject = dict[str, Any]


class Masses(TypedDict):
    theory_leaf: float
    deviation: float
    other_stop: float
    unresolved: float


class Posterior(TypedDict):
    """Calculated uncertainty from finite database samples (see uncertainty.py)."""

    mean: float
    standard_deviation: float
    credible_interval_95: list[float]
    label: str
    unresolved_mass_mean: float
    resolved_contribution_mean: float
    conditional_bounds_mean: list[float]
    masses_mean: Masses
    method: str


class PreparedDepth(TypedDict):
    expected_moves: float | None
    unit: str
    status: NotRequired[Status]
    conditional_bounds: NotRequired[list[float]]


class ScoreSummary(TypedDict):
    """A repertoire score: the exact empirical score plus its bounds, masses and posterior."""

    raw_empirical_score: float | None
    resolved_contribution: float
    unresolved_mass: float
    conditional_bounds: list[float]
    sparse_mass: float
    sparse_sensitivity: list[float]
    masses: Masses
    posterior: Posterior
    prepared_depth: NotRequired[PreparedDepth]


class ChapterScore(TypedDict, total=False):
    """A chapter's score conditional on first entry: the ScoreSummary fields plus entry probabilities.

    When the score cannot be resolved, `status` is set and the ScoreSummary fields are absent.
    """

    raw_empirical_score: float | None
    resolved_contribution: float
    unresolved_mass: float
    conditional_bounds: list[float]
    sparse_mass: float
    sparse_sensitivity: list[float]
    masses: Masses
    posterior: Posterior
    prepared_depth: PreparedDepth
    status: Status
    entry_probability: float | None
    posterior_entry_probability_mean: float | None
    first_entry_weights: dict[Position, float | None]
    entry_probability_bounds: list[float]
    overall_policy_entry_probability: float | None
    overall_policy_entry_probability_bounds: list[float] | None
    entry_probability_basis: str
    conditional_basis: str
    conditional_score_bounds: list[float]  # with UNRESOLVED_ENTRY_WEIGHTS
    probability_conditional_on_custom_root: float | None


class EntryBaseline(TypedDict, total=False):
    status: Status
    basis: str
    raw_score: float | None
    unresolved_mass: float
    conditional_bounds: list[float]
    difference_pp: float | None
    difference_definition: str
    components: list[dict]


class Chapter(TypedDict):
    id: str
    name: str
    url: str | None
    entry_status: str
    policy_overrides: dict[Position, str]
    policy_basis: str
    prepared_positions: list[Position]  # the entries and their chapter-owned descendants
    entries: list[dict]
    score: ChapterScore
    entry_baseline: EntryBaseline


class MoveOrderRow(TypedDict):
    """Where games go after an unplayed move compared with the played one, until their lines meet again."""

    kind: str  # 'meet': first position both moves reach; 'only': one move's line that never meets; 'stop'
    position: Position  # the position reached; for a stop, the position it stops from
    move: str | None  # a stop's unprepared reply (UCI), if any
    stop: str | None  # a stop's kind, as in the stopping-event ledger
    path: list[str]  # SAN from the root, through this move, along the most likely route
    played_path: list[str] | None  # a position both moves reach: the played move's most likely route there
    share: float  # probability after this move
    played_share: float  # probability after the played move
    score: float  # prior-completed score from there (for a stop, the stop's score)
    effect: float  # share-weighted score relative to the played move; the rows add up to the difference
    chapters: list[str]  # chapters holding the position


class AlternativeOption(TypedDict):
    move: str
    san: str
    chapters: list[str]  # chapters recording this move here
    sideline: bool  # only side variations record it: no chapter's first move here
    score: float | None  # repertoire score after this move, with the best choices below it
    resolved_contribution: float
    unresolved_mass: float
    prior_completed_score: float  # the value compared when choosing
    lines: list[MoveOrderRow]  # against the played move; empty for the played move itself


class Alternative(TypedDict):
    """An own-turn board where chapters record different first moves; the highest-scoring move is played."""

    position: Position
    path: list[str]
    reach: float | None  # under the overall policy; None when roots are custom
    selected: str
    options: list[AlternativeOption]


class StoppingEvent(TypedDict):
    """One way a modeled game leaves preparation under the overall policy."""

    parent_position: Position
    position: str  # full FEN after the move
    move: str | None
    representative_path_san: list[str]
    chapters: list[str]
    type: str
    unresolved: bool
    score_status: str
    sample_count: int
    counts_white_draw_black: list[int]
    probability: float
    posterior_probability_mean: float
    raw_score: float | None
    posterior_score_mean: float
    posterior_score_interval_95: list[float]
    contribution: float | None
    posterior_contribution_mean: float
    uncertainty_priority: float
    prior_fraction: float
    evidence: str
    chapter_attribution: NotRequired[dict]


class ScoreManifest(TypedDict):
    """How a score was made: source, configuration, evidence provenance and settings."""

    created_at: str
    input_path: str
    input_sha256: str
    configuration: dict
    filters: dict[str, str]
    endpoint: str
    evidence: dict[Position, dict]
    prior: list[float]
    uncertainty_method: str
    sparse_threshold: int
    positions: int
    evaluated_positions: int
    overall_policy_evaluated_positions: int
    root_weights: dict[Position, float]
    schema_version: int
    policy_profile_count: int
    overall_basis: str
    tolerance_score_points: float
    tolerance_met: bool
    selected_alternatives: dict[Position, str]  # the winner at each board in `alternatives`


class ScoreResult(TypedDict):
    """The saved score file for one color."""

    color: str
    overall: ScoreSummary
    chapters: list[Chapter]
    chapter_transitions: list[dict]
    alternatives: list[Alternative]
    starting_position_reference: dict
    events: list[StoppingEvent]
    prior_sensitivity: list[dict]
    manifest: ScoreManifest
    diagnostics: dict
    chapter_catalog: list[dict]


class CompanionManifest(TypedDict, total=False):
    """Provenance every companion analysis shares; its hashes tie it to one exact score result."""

    created_at: str
    schema_version: int
    report_path: str
    report_sha256: str
    input_path: str
    input_sha256: str
    filters: dict[str, str]
    cache_only: bool
    network_requests: int
    evidence: dict[Position, dict]
    uncached_positions: list[Position]
    supporting_sha256: dict[str, str]
