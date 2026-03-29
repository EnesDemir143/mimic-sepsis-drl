"""
Safety diagnostics for retrospective policy review.

Pairs OPE outputs with clinician sanity checks, action heatmaps, subgroup
breakdowns, and support-aware warnings before any policy is described as
clinically plausible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, Sequence, TypeAlias

from mimic_sepsis_rl.evaluation.ope import ActionSelectionPolicy, HeldOutEpisode, HeldOutStep
from mimic_sepsis_rl.mdp.actions.bins import ActionBinArtifacts, ActionBinner, N_BINS

EpisodeId: TypeAlias = str | int

VASO_BIN_LABELS = ("no_vaso", "vaso_Q1", "vaso_Q2", "vaso_Q3", "vaso_Q4")
FLUID_BIN_LABELS = ("no_fluid", "fluid_Q1", "fluid_Q2", "fluid_Q3", "fluid_Q4")


@dataclass(frozen=True)
class ActionSupport:
    """Behavior-support summary for a candidate policy action."""

    behavior_prob: float
    count: int

    def is_low_support(
        self,
        *,
        min_behavior_prob: float,
        min_count: int,
    ) -> bool:
        return self.behavior_prob < min_behavior_prob or self.count < min_count


@dataclass(frozen=True)
class SafetyReviewRow:
    """One policy-vs-clinician comparison point for safety review."""

    episode_id: EpisodeId
    step_index: int
    clinician_action: int
    policy_action: int
    policy_action_support_prob: float
    policy_action_support_count: int
    subgroup: str = "all"

    @property
    def agreement(self) -> bool:
        return self.clinician_action == self.policy_action

    def is_low_support(
        self,
        *,
        min_behavior_prob: float,
        min_count: int,
    ) -> bool:
        return (
            self.policy_action_support_prob < min_behavior_prob
            or self.policy_action_support_count < min_count
        )


@dataclass(frozen=True)
class ClinicianAgreementSummary:
    """Top-line agreement and support summary."""

    n_rows: int
    agreement_rate: float
    low_support_fraction: float
    low_support_override_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionHeatmap:
    """5×5 action heatmap for vasopressor × fluid bins."""

    title: str
    counts: tuple[tuple[int, ...], ...]
    normalized: tuple[tuple[float, ...], ...]
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "counts": [list(row) for row in self.counts],
            "normalized": [list(row) for row in self.normalized],
            "row_labels": list(self.row_labels),
            "column_labels": list(self.column_labels),
            "total": self.total,
        }


@dataclass(frozen=True)
class ClinicianSanityCase:
    """Ranked mismatch or low-support case for manual clinician review."""

    episode_id: EpisodeId
    step_index: int
    subgroup: str
    clinician_action_label: str
    policy_action_label: str
    policy_action_support_prob: float
    policy_action_support_count: int
    low_support: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubgroupSafetySummary:
    """Safety summary for one user-defined subgroup."""

    subgroup: str
    n_rows: int
    agreement_rate: float
    low_support_fraction: float
    mean_policy_action_support_prob: float
    mean_policy_action_support_count: float
    dominant_policy_action: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SupportWarning:
    """Warning emitted for poorly supported policy behavior."""

    severity: str
    episode_id: EpisodeId
    step_index: int
    subgroup: str
    policy_action_label: str
    policy_action_support_prob: float
    policy_action_support_count: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyReviewReport:
    """Combined safety review output for one evaluated policy."""

    agreement_summary: ClinicianAgreementSummary
    clinician_heatmap: ActionHeatmap
    policy_heatmap: ActionHeatmap
    delta_heatmap: ActionHeatmap
    subgroup_summaries: tuple[SubgroupSafetySummary, ...]
    sanity_cases: tuple[ClinicianSanityCase, ...]
    support_warnings: tuple[SupportWarning, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreement_summary": self.agreement_summary.to_dict(),
            "clinician_heatmap": self.clinician_heatmap.to_dict(),
            "policy_heatmap": self.policy_heatmap.to_dict(),
            "delta_heatmap": self.delta_heatmap.to_dict(),
            "subgroup_summaries": [summary.to_dict() for summary in self.subgroup_summaries],
            "sanity_cases": [case.to_dict() for case in self.sanity_cases],
            "support_warnings": [warning.to_dict() for warning in self.support_warnings],
        }


class ActionSupportLookup(Protocol):
    """Lookup behavior support for a policy action at a state."""

    def __call__(self, state: Sequence[float], action: int) -> ActionSupport:
        """Return support probability and count."""


SubgroupLookup = Callable[[HeldOutStep], str]


def _coerce_action_binner(
    action_bins: ActionBinner | ActionBinArtifacts,
) -> ActionBinner:
    if isinstance(action_bins, ActionBinner):
        return action_bins
    return ActionBinner().load(action_bins)


def build_safety_review_rows(
    policy: ActionSelectionPolicy,
    held_out_episodes: Sequence[HeldOutEpisode],
    support_lookup: ActionSupportLookup,
    *,
    subgroup_lookup: SubgroupLookup | None = None,
) -> tuple[SafetyReviewRow, ...]:
    """Build row-wise safety review data directly from held-out episodes."""
    rows: list[SafetyReviewRow] = []

    for episode in held_out_episodes:
        for step in episode.steps:
            policy_action = int(policy.select_action(step.state))
            support = support_lookup(step.state, policy_action)
            if not 0.0 <= support.behavior_prob <= 1.0:
                raise ValueError(
                    f"Support probability must be in [0, 1], got {support.behavior_prob!r}."
                )
            subgroup = subgroup_lookup(step) if subgroup_lookup else "all"
            rows.append(
                SafetyReviewRow(
                    episode_id=step.episode_id,
                    step_index=step.step_index,
                    clinician_action=step.action,
                    policy_action=policy_action,
                    policy_action_support_prob=support.behavior_prob,
                    policy_action_support_count=support.count,
                    subgroup=subgroup,
                )
            )

    return tuple(rows)


def _normalize_matrix(matrix: Sequence[Sequence[int]]) -> tuple[tuple[float, ...], ...]:
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return tuple(tuple(0.0 for _ in range(N_BINS)) for _ in range(N_BINS))
    return tuple(
        tuple(value / total for value in row)
        for row in matrix
    )


def _build_matrix(
    action_ids: Sequence[int],
    action_binner: ActionBinner,
) -> list[list[int]]:
    matrix = [[0 for _ in range(N_BINS)] for _ in range(N_BINS)]
    for action_id in action_ids:
        vaso_bin, fluid_bin = action_binner.decode_action(action_id)
        matrix[vaso_bin][fluid_bin] += 1
    return matrix


def build_action_heatmap(
    action_ids: Sequence[int],
    action_bins: ActionBinner | ActionBinArtifacts,
    *,
    title: str,
) -> ActionHeatmap:
    """Build a 5×5 action heatmap from discrete action IDs."""
    action_binner = _coerce_action_binner(action_bins)
    matrix = _build_matrix(action_ids, action_binner)
    return ActionHeatmap(
        title=title,
        counts=tuple(tuple(row) for row in matrix),
        normalized=_normalize_matrix(matrix),
        row_labels=VASO_BIN_LABELS,
        column_labels=FLUID_BIN_LABELS,
        total=len(action_ids),
    )


def build_delta_heatmap(
    clinician_action_ids: Sequence[int],
    policy_action_ids: Sequence[int],
    action_bins: ActionBinner | ActionBinArtifacts,
    *,
    title: str = "policy_minus_clinician",
) -> ActionHeatmap:
    """Return a heatmap showing policy count deltas over clinician actions."""
    action_binner = _coerce_action_binner(action_bins)
    clinician_matrix = _build_matrix(clinician_action_ids, action_binner)
    policy_matrix = _build_matrix(policy_action_ids, action_binner)
    delta_matrix = [
        [policy_value - clinician_value for policy_value, clinician_value in zip(policy_row, clinician_row)]
        for policy_row, clinician_row in zip(policy_matrix, clinician_matrix)
    ]

    return ActionHeatmap(
        title=title,
        counts=tuple(tuple(row) for row in delta_matrix),
        normalized=_normalize_matrix([[abs(value) for value in row] for row in delta_matrix]),
        row_labels=VASO_BIN_LABELS,
        column_labels=FLUID_BIN_LABELS,
        total=len(policy_action_ids),
    )


def summarize_clinician_agreement(
    rows: Sequence[SafetyReviewRow],
    *,
    min_behavior_prob: float = 0.05,
    min_count: int = 10,
) -> ClinicianAgreementSummary:
    """Summarize policy agreement with clinicians and support exposure."""
    if not rows:
        raise ValueError("At least one safety review row is required.")

    agreements = sum(1 for row in rows if row.agreement)
    low_support = sum(
        1
        for row in rows
        if row.is_low_support(
            min_behavior_prob=min_behavior_prob,
            min_count=min_count,
        )
    )
    low_support_overrides = sum(
        1
        for row in rows
        if (not row.agreement)
        and row.is_low_support(
            min_behavior_prob=min_behavior_prob,
            min_count=min_count,
        )
    )

    total_rows = len(rows)
    return ClinicianAgreementSummary(
        n_rows=total_rows,
        agreement_rate=agreements / total_rows,
        low_support_fraction=low_support / total_rows,
        low_support_override_fraction=low_support_overrides / total_rows,
    )


def rank_clinician_sanity_cases(
    rows: Sequence[SafetyReviewRow],
    action_bins: ActionBinner | ActionBinArtifacts,
    *,
    limit: int = 10,
    min_behavior_prob: float = 0.05,
    min_count: int = 10,
) -> tuple[ClinicianSanityCase, ...]:
    """Return the most review-worthy cases for manual clinician inspection."""
    action_binner = _coerce_action_binner(action_bins)

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            0 if row.agreement else 1,
            1 if row.is_low_support(
                min_behavior_prob=min_behavior_prob,
                min_count=min_count,
            ) else 0,
            -row.policy_action_support_prob,
            -row.policy_action_support_count,
        ),
        reverse=True,
    )

    cases: list[ClinicianSanityCase] = []
    for row in ranked_rows[:limit]:
        cases.append(
            ClinicianSanityCase(
                episode_id=row.episode_id,
                step_index=row.step_index,
                subgroup=row.subgroup,
                clinician_action_label=action_binner.action_label(row.clinician_action),
                policy_action_label=action_binner.action_label(row.policy_action),
                policy_action_support_prob=row.policy_action_support_prob,
                policy_action_support_count=row.policy_action_support_count,
                low_support=row.is_low_support(
                    min_behavior_prob=min_behavior_prob,
                    min_count=min_count,
                ),
            )
        )
    return tuple(cases)


def summarize_subgroups(
    rows: Sequence[SafetyReviewRow],
    *,
    min_behavior_prob: float = 0.05,
    min_count: int = 10,
) -> tuple[SubgroupSafetySummary, ...]:
    """Group safety diagnostics by subgroup label."""
    grouped: dict[str, list[SafetyReviewRow]] = defaultdict(list)
    for row in rows:
        grouped[row.subgroup].append(row)

    summaries: list[SubgroupSafetySummary] = []
    for subgroup in sorted(grouped):
        subgroup_rows = grouped[subgroup]
        action_counter = Counter(row.policy_action for row in subgroup_rows)
        total_rows = len(subgroup_rows)
        summaries.append(
            SubgroupSafetySummary(
                subgroup=subgroup,
                n_rows=total_rows,
                agreement_rate=sum(1 for row in subgroup_rows if row.agreement) / total_rows,
                low_support_fraction=sum(
                    1
                    for row in subgroup_rows
                    if row.is_low_support(
                        min_behavior_prob=min_behavior_prob,
                        min_count=min_count,
                    )
                )
                / total_rows,
                mean_policy_action_support_prob=sum(
                    row.policy_action_support_prob for row in subgroup_rows
                )
                / total_rows,
                mean_policy_action_support_count=sum(
                    row.policy_action_support_count for row in subgroup_rows
                )
                / total_rows,
                dominant_policy_action=action_counter.most_common(1)[0][0],
            )
        )
    return tuple(summaries)


def generate_support_warnings(
    rows: Sequence[SafetyReviewRow],
    action_bins: ActionBinner | ActionBinArtifacts,
    *,
    min_behavior_prob: float = 0.05,
    min_count: int = 10,
) -> tuple[SupportWarning, ...]:
    """Generate row-level warnings for poorly supported policy actions."""
    action_binner = _coerce_action_binner(action_bins)
    warnings: list[SupportWarning] = []

    for row in rows:
        if not row.is_low_support(
            min_behavior_prob=min_behavior_prob,
            min_count=min_count,
        ):
            continue

        severe = (
            row.policy_action_support_prob < (min_behavior_prob / 2.0)
            or row.policy_action_support_count < max(1, min_count // 2)
        )
        severity = "high" if severe else "medium"
        policy_action_label = action_binner.action_label(row.policy_action)
        warnings.append(
            SupportWarning(
                severity=severity,
                episode_id=row.episode_id,
                step_index=row.step_index,
                subgroup=row.subgroup,
                policy_action_label=policy_action_label,
                policy_action_support_prob=row.policy_action_support_prob,
                policy_action_support_count=row.policy_action_support_count,
                message=(
                    f"Policy action {policy_action_label} has weak behavior support "
                    f"(prob={row.policy_action_support_prob:.3f}, "
                    f"count={row.policy_action_support_count})."
                ),
            )
        )

    return tuple(warnings)


def build_safety_review(
    rows: Sequence[SafetyReviewRow],
    action_bins: ActionBinner | ActionBinArtifacts,
    *,
    min_behavior_prob: float = 0.05,
    min_count: int = 10,
    sanity_case_limit: int = 10,
) -> SafetyReviewReport:
    """Build the combined safety review bundle."""
    if not rows:
        raise ValueError("At least one safety review row is required.")

    clinician_actions = [row.clinician_action for row in rows]
    policy_actions = [row.policy_action for row in rows]

    return SafetyReviewReport(
        agreement_summary=summarize_clinician_agreement(
            rows,
            min_behavior_prob=min_behavior_prob,
            min_count=min_count,
        ),
        clinician_heatmap=build_action_heatmap(
            clinician_actions,
            action_bins,
            title="clinician_actions",
        ),
        policy_heatmap=build_action_heatmap(
            policy_actions,
            action_bins,
            title="policy_actions",
        ),
        delta_heatmap=build_delta_heatmap(
            clinician_actions,
            policy_actions,
            action_bins,
        ),
        subgroup_summaries=summarize_subgroups(
            rows,
            min_behavior_prob=min_behavior_prob,
            min_count=min_count,
        ),
        sanity_cases=rank_clinician_sanity_cases(
            rows,
            action_bins,
            limit=sanity_case_limit,
            min_behavior_prob=min_behavior_prob,
            min_count=min_count,
        ),
        support_warnings=generate_support_warnings(
            rows,
            action_bins,
            min_behavior_prob=min_behavior_prob,
            min_count=min_count,
        ),
    )


__all__ = [
    "ActionHeatmap",
    "ActionSupport",
    "ClinicianAgreementSummary",
    "ClinicianSanityCase",
    "SafetyReviewRow",
    "SafetyReviewReport",
    "SubgroupSafetySummary",
    "SupportWarning",
    "build_action_heatmap",
    "build_delta_heatmap",
    "build_safety_review",
    "build_safety_review_rows",
    "generate_support_warnings",
    "rank_clinician_sanity_cases",
    "summarize_clinician_agreement",
    "summarize_subgroups",
]
