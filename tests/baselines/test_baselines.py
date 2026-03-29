"""
Regression tests for the baseline comparison pipeline.

Covers:
- Clinician baseline: action distribution, episode returns, mortality rate
- No-treatment baseline: correct fixed action, metric computation
- Behavior cloning: training, accuracy, prediction, serialisation
- Empty input handling for all baselines
- Dataset compatibility: baselines consume the shared transition contract
- Result serialisation round-trips
"""

from __future__ import annotations

import math
import random

import pytest

from mimic_sepsis_rl.datasets.transitions import TransitionRow
from mimic_sepsis_rl.baselines.clinician import (
    ClinicianBaselineResult,
    evaluate_clinician_baseline,
)
from mimic_sepsis_rl.baselines.no_treatment import (
    NoTreatmentBaselineResult,
    evaluate_no_treatment_baseline,
    NO_TREATMENT_ACTION,
)
from mimic_sepsis_rl.baselines.behavior_cloning import (
    BehaviorCloningResult,
    SoftmaxClassifier,
    train_behavior_cloning,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_transitions(
    n_episodes: int = 5,
    steps_per_episode: int = 4,
    state_dim: int = 4,
    seed: int = 42,
) -> list[TransitionRow]:
    """Create synthetic transitions for testing."""
    rng = random.Random(seed)
    transitions: list[TransitionRow] = []

    for ep in range(n_episodes):
        mortality = ep % 2  # alternating survival
        for step in range(steps_per_episode):
            state = tuple(rng.gauss(0, 1) for _ in range(state_dim))
            is_last = step == steps_per_episode - 1
            reward = (15.0 if mortality == 0 else -15.0) if is_last else rng.uniform(-0.5, 0.5)
            next_state = state if is_last else tuple(rng.gauss(0, 1) for _ in range(state_dim))
            transitions.append(
                TransitionRow(
                    stay_id=ep * 100,
                    step_index=step,
                    state=state,
                    action=rng.randint(0, 24),
                    reward=reward,
                    next_state=next_state,
                    done=is_last,
                )
            )

    return transitions


# ===================================================================
# Clinician Baseline Tests
# ===================================================================


class TestClinicianBaseline:
    """Verify clinician-action replay baseline."""

    def test_episode_count(self) -> None:
        """Correct number of episodes reported."""
        trans = _make_transitions(n_episodes=5)
        result = evaluate_clinician_baseline(trans)
        assert result.n_episodes == 5

    def test_transition_count(self) -> None:
        """Correct total transition count."""
        trans = _make_transitions(n_episodes=3, steps_per_episode=6)
        result = evaluate_clinician_baseline(trans)
        assert result.n_transitions == 18

    def test_action_distribution_sums(self) -> None:
        """Action distribution sums to total transitions."""
        trans = _make_transitions(n_episodes=5, steps_per_episode=4)
        result = evaluate_clinician_baseline(trans)
        total_actions = sum(result.action_distribution.values())
        assert total_actions == 20

    def test_mortality_rate(self) -> None:
        """Mortality rate is computed correctly."""
        trans = _make_transitions(n_episodes=4)  # ep 0,2 survive; ep 1,3 die
        result = evaluate_clinician_baseline(trans)
        assert result.mortality_rate == pytest.approx(0.5)

    def test_mean_reward_is_finite(self) -> None:
        """Mean reward is a finite number."""
        trans = _make_transitions()
        result = evaluate_clinician_baseline(trans)
        assert math.isfinite(result.mean_reward)

    def test_empty_input(self) -> None:
        """Empty input returns zero-filled result."""
        result = evaluate_clinician_baseline([])
        assert result.n_episodes == 0
        assert result.n_transitions == 0
        assert result.mean_episode_return == 0.0

    def test_serialisation(self) -> None:
        """Result serialises to dict with correct keys."""
        trans = _make_transitions()
        result = evaluate_clinician_baseline(trans)
        d = result.to_dict()
        assert d["baseline"] == "clinician"
        assert "mean_episode_return" in d
        assert "action_distribution" in d


# ===================================================================
# No-Treatment Baseline Tests
# ===================================================================


class TestNoTreatmentBaseline:
    """Verify no-treatment baseline."""

    def test_policy_action_is_zero(self) -> None:
        """No-treatment always selects action 0."""
        trans = _make_transitions()
        result = evaluate_no_treatment_baseline(trans)
        assert result.policy_action == NO_TREATMENT_ACTION

    def test_episode_count(self) -> None:
        """Correct episode count."""
        trans = _make_transitions(n_episodes=7)
        result = evaluate_no_treatment_baseline(trans)
        assert result.n_episodes == 7

    def test_mortality_rate(self) -> None:
        """Mortality rate matches observed outcomes."""
        trans = _make_transitions(n_episodes=4)
        result = evaluate_no_treatment_baseline(trans)
        assert result.mortality_rate == pytest.approx(0.5)

    def test_empty_input(self) -> None:
        """Empty input returns zero-filled result."""
        result = evaluate_no_treatment_baseline([])
        assert result.n_episodes == 0
        assert result.policy_action == 0

    def test_serialisation(self) -> None:
        """Result serialises to dict."""
        trans = _make_transitions()
        result = evaluate_no_treatment_baseline(trans)
        d = result.to_dict()
        assert d["baseline"] == "no_treatment"
        assert d["policy_action"] == 0


# ===================================================================
# Behavior Cloning Tests
# ===================================================================


class TestBehaviorCloning:
    """Verify behavior cloning baseline."""

    def test_training_produces_result(self) -> None:
        """Training returns a valid BehaviorCloningResult."""
        trans = _make_transitions(n_episodes=5, state_dim=4)
        result, model = train_behavior_cloning(trans, n_epochs=5)
        assert result.n_train == 20
        assert result.n_epochs == 5
        assert 0.0 <= result.train_accuracy <= 1.0

    def test_prediction_in_valid_range(self) -> None:
        """Predictions are valid action IDs."""
        trans = _make_transitions(n_episodes=3, state_dim=4)
        _, model = train_behavior_cloning(trans, n_epochs=3)
        for t in trans[:5]:
            action = model.predict(t.state)
            assert 0 <= action < 25

    def test_softmax_probabilities_sum_to_one(self) -> None:
        """Softmax probabilities sum to 1.0."""
        model = SoftmaxClassifier(state_dim=4, n_actions=25, seed=42)
        state = (1.0, -0.5, 0.3, 0.8)
        probs = model.predict_probs(state)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0 for p in probs)

    def test_accuracy_improves_with_training(self) -> None:
        """Accuracy should not decrease after training (imperfect guarantee)."""
        # Use a dataset with a learnable pattern
        rng = random.Random(42)
        transitions = []
        for i in range(100):
            # Simple pattern: action depends on sign of first feature
            feat = rng.gauss(0, 1)
            action = 0 if feat >= 0 else 1
            transitions.append(
                TransitionRow(
                    stay_id=(i // 10) * 100,
                    step_index=i % 10,
                    state=(feat, rng.gauss(0, 1)),
                    action=action,
                    reward=0.0 if i % 10 != 9 else 15.0,
                    next_state=(feat, rng.gauss(0, 1)),
                    done=i % 10 == 9,
                )
            )

        result_short, _ = train_behavior_cloning(transitions, n_epochs=1, seed=42)
        result_long, _ = train_behavior_cloning(transitions, n_epochs=50, seed=42)
        # After more training, accuracy should be at least as good
        assert result_long.train_accuracy >= result_short.train_accuracy - 0.1

    def test_empty_input(self) -> None:
        """Empty input returns zero-filled result."""
        result, _ = train_behavior_cloning([], n_epochs=5)
        assert result.n_train == 0
        assert result.train_accuracy == 0.0

    def test_eval_with_separate_set(self) -> None:
        """Eval accuracy uses eval transitions when provided."""
        train = _make_transitions(n_episodes=3, state_dim=4, seed=42)
        eval_data = _make_transitions(n_episodes=2, state_dim=4, seed=99)
        result, _ = train_behavior_cloning(
            train,
            eval_transitions=eval_data,
            n_epochs=5,
        )
        assert result.n_train == 12  # 3 episodes × 4 steps
        assert result.n_eval == 8   # 2 episodes × 4 steps

    def test_serialisation(self) -> None:
        """Result serialises to dict."""
        trans = _make_transitions()
        result, _ = train_behavior_cloning(trans, n_epochs=3)
        d = result.to_dict()
        assert d["baseline"] == "behavior_cloning"
        assert "train_accuracy" in d
        assert "eval_accuracy" in d


# ===================================================================
# Dataset Compatibility Tests
# ===================================================================


class TestDatasetCompatibility:
    """Verify all baselines consume the shared transition contract."""

    def test_all_baselines_accept_same_transitions(self) -> None:
        """All baselines can process the same transition list."""
        trans = _make_transitions(n_episodes=5)

        clin = evaluate_clinician_baseline(trans)
        assert clin.n_episodes == 5

        no_treat = evaluate_no_treatment_baseline(trans)
        assert no_treat.n_episodes == 5

        bc, _ = train_behavior_cloning(trans, n_epochs=2)
        assert bc.n_train == 20

    def test_results_are_comparable(self) -> None:
        """All baselines report the same episode count and transition count."""
        trans = _make_transitions(n_episodes=4, steps_per_episode=5)

        clin = evaluate_clinician_baseline(trans)
        no_treat = evaluate_no_treatment_baseline(trans)
        bc, _ = train_behavior_cloning(trans, n_epochs=2)

        assert clin.n_episodes == no_treat.n_episodes
        assert clin.n_transitions == no_treat.n_transitions
        assert bc.n_train == clin.n_transitions
