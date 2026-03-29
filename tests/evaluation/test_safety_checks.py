"""Regression tests for evaluation safety diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mimic_sepsis_rl.evaluation.ope import HeldOutEpisode, HeldOutStep
from mimic_sepsis_rl.evaluation.safety import (
    ActionSupport,
    build_safety_review,
    build_safety_review_rows,
    SafetyReviewRow,
)
from mimic_sepsis_rl.mdp.actions.bins import ActionBinArtifacts


@dataclass
class MappingPolicy:
    """Deterministic policy keyed by exact state tuple."""

    actions: dict[tuple[float, ...], int]

    def select_action(self, state: tuple[float, ...]) -> int:
        return self.actions[tuple(float(value) for value in state)]


def _action_bins() -> ActionBinArtifacts:
    return ActionBinArtifacts(
        spec_version="1.0.0",
        manifest_seed=42,
        vaso_edges=(0.1, 0.2, 0.3),
        fluid_edges=(100.0, 200.0, 300.0),
        n_train_vaso_nonzero=100,
        n_train_fluid_nonzero=100,
    )


def test_build_safety_review_combines_agreement_heatmaps_and_warnings() -> None:
    rows = (
        SafetyReviewRow(
            episode_id="ep-1",
            step_index=0,
            clinician_action=0,
            policy_action=0,
            policy_action_support_prob=0.40,
            policy_action_support_count=30,
            subgroup="shock",
        ),
        SafetyReviewRow(
            episode_id="ep-1",
            step_index=1,
            clinician_action=6,
            policy_action=24,
            policy_action_support_prob=0.02,
            policy_action_support_count=3,
            subgroup="shock",
        ),
        SafetyReviewRow(
            episode_id="ep-2",
            step_index=0,
            clinician_action=12,
            policy_action=12,
            policy_action_support_prob=0.20,
            policy_action_support_count=18,
            subgroup="renal",
        ),
        SafetyReviewRow(
            episode_id="ep-2",
            step_index=1,
            clinician_action=19,
            policy_action=19,
            policy_action_support_prob=0.03,
            policy_action_support_count=7,
            subgroup="renal",
        ),
    )

    report = build_safety_review(
        rows,
        _action_bins(),
        min_behavior_prob=0.05,
        min_count=10,
        sanity_case_limit=2,
    )

    assert report.agreement_summary.agreement_rate == pytest.approx(0.75)
    assert report.agreement_summary.low_support_fraction == pytest.approx(0.5)
    assert report.agreement_summary.low_support_override_fraction == pytest.approx(0.25)
    assert report.clinician_heatmap.total == 4
    assert report.policy_heatmap.counts[4][4] == 1
    assert report.delta_heatmap.counts[4][4] == 1
    assert len(report.subgroup_summaries) == 2
    assert report.subgroup_summaries[0].subgroup == "renal"
    assert report.subgroup_summaries[1].subgroup == "shock"
    assert report.subgroup_summaries[1].agreement_rate == pytest.approx(0.5)
    assert report.sanity_cases[0].policy_action_label == "vaso_Q4×fluid_Q4"
    assert report.sanity_cases[0].low_support is True
    assert [warning.severity for warning in report.support_warnings] == ["high", "medium"]


def test_build_safety_review_rows_uses_policy_and_support_lookup() -> None:
    episodes = (
        HeldOutEpisode(
            episode_id="ep-1",
            steps=(
                HeldOutStep(
                    episode_id="ep-1",
                    step_index=0,
                    state=(0.0, 1.0),
                    action=0,
                    reward=0.0,
                    done=False,
                    behavior_action_prob=0.4,
                ),
                HeldOutStep(
                    episode_id="ep-1",
                    step_index=1,
                    state=(1.0, 1.0),
                    action=6,
                    reward=0.0,
                    done=True,
                    behavior_action_prob=0.3,
                ),
            ),
        ),
    )
    policy = MappingPolicy(actions={(0.0, 1.0): 0, (1.0, 1.0): 24})
    support_map = {
        0: ActionSupport(behavior_prob=0.30, count=25),
        24: ActionSupport(behavior_prob=0.01, count=2),
    }

    rows = build_safety_review_rows(
        policy,
        episodes,
        lambda state, action: support_map[action],
        subgroup_lookup=lambda step: "early" if step.step_index == 0 else "late",
    )

    assert [row.policy_action for row in rows] == [0, 24]
    assert [row.subgroup for row in rows] == ["early", "late"]
    assert rows[1].policy_action_support_prob == pytest.approx(0.01)
    assert rows[1].policy_action_support_count == 2
