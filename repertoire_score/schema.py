"""The shape of the saved score JSON (`white.json`, `black.json`) and the manifest every companion shares.

These TypedDicts document the format and give editors and type checkers something to check against; nothing
validates files against them at run time. Scores are owner-relative expected points between 0 and 1, and
`*_pp` fields are percentage points. Fields that cannot be resolved from the evidence are None.
"""

from typing import NotRequired, TypedDict

from .status import Status

Position = str  # canonical four-field FEN: pieces, side to move, castling, legal en passant


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


class ChapterScore(ScoreSummary, total=False):
    """A chapter's score conditional on first entry; `status` replaces the score when it cannot be resolved."""

    status: Status
    entry_probability: float | None
    posterior_entry_probability_mean: float | None
    first_entry_weights: dict[Position, float | None]
    entry_probability_bounds: list[float]
    overall_policy_entry_probability: float | None
    overall_policy_entry_probability_bounds: list[float] | None
    entry_probability_basis: str
    conditional_basis: str


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
    region: dict | None
    entries: list[dict]
    score: ChapterScore
    entry_baseline: EntryBaseline


class AlternativeOption(TypedDict):
    move: str
    san: str
    chapters: list[str]  # chapters whose first recorded move here is this one
    score: float | None  # repertoire score after this move, with the best choices below it
    resolved_contribution: float
    unresolved_mass: float
    prior_completed_score: float  # the value compared when choosing


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
