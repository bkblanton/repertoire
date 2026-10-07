"""Status values saved in score and analysis JSON. They serialize as their plain string values."""

from enum import StrEnum


class Status(StrEnum):
    # Calculated completely from known evidence.
    RESOLVED = 'resolved'
    AVAILABLE = 'available'
    EVALUATED = 'evaluated'
    CALCULATED = 'calculated'
    # A chapter cannot be scored as configured.
    ENTRY_CONFIGURATION_REQUIRED = 'entry_configuration_required'
    UNRESOLVED_ENTRY_WEIGHTS = 'unresolved_first_entry_weights'
    UNREACHABLE_MULTIPLE_ENTRIES = 'unreachable_multiple_entries_require_conditional_weights'
    NO_REACHABLE_ENTRY = 'no_reachable_entry'
    # Some evidence is missing; values are bounded or partial rather than invented.
    UNRESOLVED_EVIDENCE = 'unresolved_evidence'
    UNRESOLVED_STOPPING_EVIDENCE = 'unresolved_stopping_evidence'
    UNRESOLVED_OPPONENT_DISTRIBUTION = 'unresolved_opponent_distribution'
    UNRESOLVED_MOVE_DISTRIBUTION = 'unresolved_move_distribution'
    UNRESOLVED_GAP_REACH = 'unresolved_gap_reach'
    UNRESOLVED_ENTRY_SCORE = 'unresolved_entry_score'
    UNRESOLVED_DESTINATION_REACH = 'unresolved_destination_reach'
    UNRESOLVED_SOURCE_REACH = 'unresolved_or_zero_source_reach'
    TOO_FEW_GROUPS = 'too_few_groups'
