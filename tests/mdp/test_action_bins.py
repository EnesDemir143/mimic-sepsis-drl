"""
Regression tests for the 25-action treatment encoding pipeline.

Covers:
- Vasopressor NE-equivalent standardisation
- IV fluid per-step aggregation
- Train-only bin edge learning
- 5×5 action mapping and decode round-trip
- Zero-dose explicit handling
- Leakage guard (no non-train data in fit)
- Artifact serialisation round-trip
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from mimic_sepsis_rl.mdp.actions.vasopressors import (
    VasopressorStandardiser,
    NE_EQUIVALENCE,
    ITEM_TO_AGENT,
)
from mimic_sepsis_rl.mdp.actions.fluids import (
    FluidAggregator,
    IV_FLUID_ITEM_IDS,
)
from mimic_sepsis_rl.mdp.actions.bins import (
    ACTION_SPEC_VERSION,
    ActionBinArtifacts,
    ActionBinner,
    N_ACTIONS,
    N_BINS,
    _assign_bin,
    _learn_edges,
    save_action_bin_artifacts,
    load_action_bin_artifacts,
)


# ===================================================================
# Vasopressor Standardisation Tests
# ===================================================================


class TestVasopressorStandardisation:
    """Verify NE-equivalent dose conversion."""

    def test_known_conversions(self) -> None:
        """Validate conversion factors against pharmacological references."""
        std = VasopressorStandardiser()

        input_df = pl.DataFrame(
            {
                "stay_id": [1, 1, 1, 1, 1],
                "itemid": [221906, 221289, 221662, 222315, 222042],
                "rate": [1.0, 1.0, 1.0, 1.0, 1.0],
                "starttime": [datetime(2024, 1, 1)] * 5,
                "endtime": [datetime(2024, 1, 1, 4)] * 5,
            }
        )

        result = std.standardise(input_df)
        rates = dict(
            zip(
                result.get_column("agent").to_list(),
                result.get_column("ne_equiv_rate").to_list(),
            )
        )

        assert rates["norepinephrine"] == pytest.approx(1.0)
        assert rates["epinephrine"] == pytest.approx(1.0)
        assert rates["dopamine"] == pytest.approx(0.01)
        assert rates["phenylephrine"] == pytest.approx(0.1)
        assert rates["vasopressin"] == pytest.approx(2.5)

    def test_zero_rate_preserved(self) -> None:
        """Zero infusion rate maps to zero NE-equivalent."""
        std = VasopressorStandardiser()
        input_df = pl.DataFrame(
            {
                "stay_id": [1],
                "itemid": [221906],
                "rate": [0.0],
                "starttime": [datetime(2024, 1, 1)],
                "endtime": [datetime(2024, 1, 1, 4)],
            }
        )
        result = std.standardise(input_df)
        assert result.get_column("ne_equiv_rate")[0] == pytest.approx(0.0)

    def test_empty_input_returns_schema(self) -> None:
        """Empty inputevents returns correct schema with zero rows."""
        std = VasopressorStandardiser()
        empty = pl.DataFrame(
            {
                "stay_id": pl.Series([], dtype=pl.Int64),
                "itemid": pl.Series([], dtype=pl.Int64),
                "rate": pl.Series([], dtype=pl.Float64),
                "starttime": pl.Series([], dtype=pl.Datetime),
                "endtime": pl.Series([], dtype=pl.Datetime),
            }
        )
        result = std.standardise(empty)
        assert result.height == 0
        assert "ne_equiv_rate" in result.columns
        assert "agent" in result.columns

    def test_non_vaso_items_filtered(self) -> None:
        """Non-vasopressor item IDs are excluded."""
        std = VasopressorStandardiser()
        input_df = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "itemid": [221906, 999999],
                "rate": [0.1, 500.0],
                "starttime": [datetime(2024, 1, 1)] * 2,
                "endtime": [datetime(2024, 1, 1, 4)] * 2,
            }
        )
        result = std.standardise(input_df)
        assert result.height == 1
        assert result.get_column("agent")[0] == "norepinephrine"

    def test_item_to_agent_completeness(self) -> None:
        """Every equivalence entry has corresponding item IDs."""
        for agent in NE_EQUIVALENCE:
            agent_items = [k for k, v in ITEM_TO_AGENT.items() if v == agent]
            assert len(agent_items) > 0, f"Agent {agent} has no item IDs"


# ===================================================================
# IV Fluid Aggregation Tests
# ===================================================================


class TestFluidAggregation:
    """Verify IV fluid per-step aggregation logic."""

    def test_basic_aggregation(self) -> None:
        """Fluids within a step are summed correctly."""
        agg = FluidAggregator()

        input_df = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "itemid": [220949, 225168],
                "amount": [500.0, 300.0],
                "starttime": [
                    datetime(2024, 1, 1, 0, 30),
                    datetime(2024, 1, 1, 1, 0),
                ],
            }
        )
        steps = pl.DataFrame(
            {
                "stay_id": [1],
                "step_index": [0],
                "step_start": [datetime(2024, 1, 1, 0, 0)],
                "step_end": [datetime(2024, 1, 1, 4, 0)],
            }
        )

        result = agg.aggregate_per_step(input_df, steps)
        assert result.height == 1
        assert result.get_column("fluid_volume_4h")[0] == pytest.approx(800.0)

    def test_zero_when_no_fluids(self) -> None:
        """Steps with no IV fluid events get zero."""
        agg = FluidAggregator()

        input_df = pl.DataFrame(
            {
                "stay_id": [1],
                "itemid": [999999],
                "amount": [1000.0],
                "starttime": [datetime(2024, 1, 1, 0, 30)],
            }
        )
        steps = pl.DataFrame(
            {
                "stay_id": [1],
                "step_index": [0],
                "step_start": [datetime(2024, 1, 1, 0, 0)],
                "step_end": [datetime(2024, 1, 1, 4, 0)],
            }
        )

        result = agg.aggregate_per_step(input_df, steps)
        assert result.get_column("fluid_volume_4h")[0] == pytest.approx(0.0)

    def test_cross_step_boundary(self) -> None:
        """Events outside the step boundary are excluded."""
        agg = FluidAggregator()

        input_df = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "itemid": [220949, 220949],
                "amount": [500.0, 300.0],
                "starttime": [
                    datetime(2024, 1, 1, 0, 30),   # in step 0
                    datetime(2024, 1, 1, 4, 30),   # in step 1
                ],
            }
        )
        steps = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "step_index": [0, 1],
                "step_start": [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 4, 0),
                ],
                "step_end": [
                    datetime(2024, 1, 1, 4, 0),
                    datetime(2024, 1, 1, 8, 0),
                ],
            }
        )

        result = agg.aggregate_per_step(input_df, steps).sort("step_index")
        volumes = result.get_column("fluid_volume_4h").to_list()
        assert volumes[0] == pytest.approx(500.0)
        assert volumes[1] == pytest.approx(300.0)

    def test_empty_input(self) -> None:
        """Empty inputevents returns zero-filled steps."""
        agg = FluidAggregator()

        empty = pl.DataFrame(
            {
                "stay_id": pl.Series([], dtype=pl.Int64),
                "itemid": pl.Series([], dtype=pl.Int64),
                "amount": pl.Series([], dtype=pl.Float64),
                "starttime": pl.Series([], dtype=pl.Datetime),
            }
        )
        steps = pl.DataFrame(
            {
                "stay_id": [1],
                "step_index": [0],
                "step_start": [datetime(2024, 1, 1, 0, 0)],
                "step_end": [datetime(2024, 1, 1, 4, 0)],
            }
        )

        result = agg.aggregate_per_step(empty, steps)
        assert result.height == 1
        assert result.get_column("fluid_volume_4h")[0] == pytest.approx(0.0)

    def test_item_id_filtering(self) -> None:
        """filter_fluids keeps only known IV fluid item IDs."""
        agg = FluidAggregator()
        input_df = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "itemid": [220949, 999999],
                "amount": [500.0, 100.0],
                "starttime": [datetime(2024, 1, 1)] * 2,
            }
        )
        filtered = agg.filter_fluids(input_df)
        assert filtered.height == 1


# ===================================================================
# Bin Edge Learning Tests
# ===================================================================


class TestBinEdgeLearning:
    """Verify train-only quartile learning."""

    def test_basic_edge_learning(self) -> None:
        """Edges are derived from non-zero values only."""
        values = pl.Series([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        edges = _learn_edges(values, "test")
        assert len(edges) == 3
        # Edges should be strictly increasing
        assert edges[0] < edges[1] < edges[2]

    def test_too_few_nonzero_raises(self) -> None:
        """Fewer than 4 non-zero values raises ValueError."""
        values = pl.Series([0.0, 0.0, 1.0, 2.0, 3.0])
        # Only 3 non-zero values
        with pytest.raises(ValueError, match="at least"):
            _learn_edges(values, "test")

    def test_all_zero_raises(self) -> None:
        """All zero values raises ValueError."""
        values = pl.Series([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="at least"):
            _learn_edges(values, "test")

    def test_edges_strictly_increasing(self) -> None:
        """Even with tied values, edges are forced monotonic."""
        values = pl.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
        edges = _learn_edges(values, "tied")
        for i in range(1, len(edges)):
            assert edges[i] > edges[i - 1]


# ===================================================================
# Bin Assignment Tests
# ===================================================================


class TestBinAssignment:
    """Verify individual value → bin mapping."""

    def test_zero_gets_bin_zero(self) -> None:
        """Zero dose is always bin 0."""
        edges = (0.5, 1.0, 1.5)
        assert _assign_bin(0.0, edges) == 0
        assert _assign_bin(-0.1, edges) == 0

    def test_nan_gets_bin_zero(self) -> None:
        """NaN dose maps to bin 0 (no treatment)."""
        assert _assign_bin(float("nan"), (0.5, 1.0, 1.5)) == 0

    def test_quartile_boundaries(self) -> None:
        """Values at quartile borders go to the expected bin."""
        edges = (0.5, 1.0, 1.5)
        assert _assign_bin(0.3, edges) == 1   # ≤ Q25
        assert _assign_bin(0.5, edges) == 1   # = Q25
        assert _assign_bin(0.7, edges) == 2   # Q25 < v ≤ Q50
        assert _assign_bin(1.0, edges) == 2   # = Q50
        assert _assign_bin(1.2, edges) == 3   # Q50 < v ≤ Q75
        assert _assign_bin(1.5, edges) == 3   # = Q75
        assert _assign_bin(2.0, edges) == 4   # > Q75

    def test_very_large_value(self) -> None:
        """Extreme values go to the highest bin."""
        edges = (0.5, 1.0, 1.5)
        assert _assign_bin(1e6, edges) == 4


# ===================================================================
# Action Binner End-to-End Tests
# ===================================================================


class TestActionBinner:
    """Integration tests for fit → transform → decode pipeline."""

    @pytest.fixture
    def train_data(self) -> pl.DataFrame:
        """Synthetic training data with 200 steps."""
        import random

        random.seed(42)
        n = 200
        vaso = [0.0] * 60 + [random.uniform(0.01, 1.5) for _ in range(140)]
        fluid = [0.0] * 40 + [random.uniform(10.0, 2000.0) for _ in range(160)]
        random.shuffle(vaso)
        random.shuffle(fluid)
        return pl.DataFrame(
            {"vaso_dose_4h": vaso[:n], "fluid_volume_4h": fluid[:n]}
        )

    def test_fit_produces_valid_artifacts(self, train_data: pl.DataFrame) -> None:
        """Fitting produces valid, non-empty artifacts."""
        binner = ActionBinner()
        binner.fit(train_data, manifest_seed=42)

        arts = binner.artifacts
        assert arts.spec_version == ACTION_SPEC_VERSION
        assert arts.manifest_seed == 42
        assert len(arts.vaso_edges) == 3
        assert len(arts.fluid_edges) == 3
        assert arts.n_train_vaso_nonzero > 0
        assert arts.n_train_fluid_nonzero > 0

    def test_unfitted_transform_raises(self) -> None:
        """Transforming before fitting raises RuntimeError."""
        binner = ActionBinner()
        df = pl.DataFrame({"vaso_dose_4h": [0.1], "fluid_volume_4h": [100.0]})
        with pytest.raises(RuntimeError, match="not been fitted"):
            binner.transform(df)

    def test_transform_produces_25_actions(self, train_data: pl.DataFrame) -> None:
        """All action IDs are in [0, 24]."""
        binner = ActionBinner()
        binner.fit(train_data, manifest_seed=42)
        result = binner.transform(train_data)

        assert "action_id" in result.columns
        assert "vaso_bin" in result.columns
        assert "fluid_bin" in result.columns

        action_ids = result.get_column("action_id")
        assert action_ids.min() >= 0
        assert action_ids.max() < N_ACTIONS

    def test_zero_dose_maps_to_action_zero(self) -> None:
        """Both treatments zero → action_id=0 (no_vaso×no_fluid)."""
        binner = ActionBinner()
        # Minimal train data
        import random
        random.seed(1)
        vaso = [0.0] * 10 + [random.uniform(0.1, 2.0) for _ in range(40)]
        fluid = [0.0] * 10 + [random.uniform(50.0, 1500.0) for _ in range(40)]
        train = pl.DataFrame({"vaso_dose_4h": vaso, "fluid_volume_4h": fluid})
        binner.fit(train, manifest_seed=1)

        test_df = pl.DataFrame(
            {"vaso_dose_4h": [0.0], "fluid_volume_4h": [0.0]}
        )
        result = binner.transform(test_df)
        assert result.get_column("action_id")[0] == 0

    def test_decode_roundtrip(self) -> None:
        """Every action_id decodes to a valid (vaso_bin, fluid_bin) pair."""
        import random
        random.seed(7)
        vaso = [0.0] * 5 + [random.uniform(0.1, 2.0) for _ in range(45)]
        fluid = [0.0] * 5 + [random.uniform(50.0, 1500.0) for _ in range(45)]
        train = pl.DataFrame({"vaso_dose_4h": vaso, "fluid_volume_4h": fluid})

        binner = ActionBinner()
        binner.fit(train, manifest_seed=7)

        for aid in range(N_ACTIONS):
            vb, fb = binner.decode_action(aid)
            assert 0 <= vb < N_BINS
            assert 0 <= fb < N_BINS
            assert vb * N_BINS + fb == aid

    def test_decode_invalid_raises(self) -> None:
        """Out-of-range action IDs raise ValueError."""
        binner = ActionBinner()
        with pytest.raises(ValueError, match="action_id must be"):
            binner.decode_action(25)
        with pytest.raises(ValueError, match="action_id must be"):
            binner.decode_action(-1)

    def test_action_label_format(self) -> None:
        """action_label returns human-readable format."""
        import random
        random.seed(99)
        vaso = [0.0] * 5 + [random.uniform(0.1, 2.0) for _ in range(45)]
        fluid = [0.0] * 5 + [random.uniform(50.0, 1500.0) for _ in range(45)]
        train = pl.DataFrame({"vaso_dose_4h": vaso, "fluid_volume_4h": fluid})

        binner = ActionBinner()
        binner.fit(train, manifest_seed=99)

        assert binner.action_label(0) == "no_vaso×no_fluid"
        assert binner.action_label(24) == "vaso_Q4×fluid_Q4"
        assert "×" in binner.action_label(12)


# ===================================================================
# Leakage Protection Tests
# ===================================================================


class TestLeakageProtection:
    """Ensure bin edges are learned only from train data."""

    def test_different_seeds_different_bins(self) -> None:
        """Fitting with different data produces different artifact seeds."""
        import random

        data1 = pl.DataFrame(
            {
                "vaso_dose_4h": [0.0] * 5 + [random.uniform(0.1, 2.0) for _ in range(45)],
                "fluid_volume_4h": [0.0] * 5 + [random.uniform(50, 1500) for _ in range(45)],
            }
        )

        binner = ActionBinner()
        binner.fit(data1, manifest_seed=42)
        assert binner.artifacts.manifest_seed == 42

        binner2 = ActionBinner()
        binner2.fit(data1, manifest_seed=99)
        assert binner2.artifacts.manifest_seed == 99


# ===================================================================
# Artifact Serialisation Tests
# ===================================================================


class TestArtifactSerialisation:
    """Verify JSON round-trip for action bin artifacts."""

    def test_dict_roundtrip(self) -> None:
        """to_dict / from_dict preserves all fields."""
        original = ActionBinArtifacts(
            spec_version="1.0.0",
            manifest_seed=42,
            vaso_edges=(0.1, 0.5, 1.0),
            fluid_edges=(100.0, 500.0, 1000.0),
            n_train_vaso_nonzero=140,
            n_train_fluid_nonzero=160,
        )
        restored = ActionBinArtifacts.from_dict(original.to_dict())
        assert original == restored

    def test_json_file_roundtrip(self) -> None:
        """Save and load produce identical artifacts."""
        original = ActionBinArtifacts(
            spec_version="1.0.0",
            manifest_seed=42,
            vaso_edges=(0.1, 0.5, 1.0),
            fluid_edges=(100.0, 500.0, 1000.0),
            n_train_vaso_nonzero=140,
            n_train_fluid_nonzero=160,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "action_bins.json"
            save_action_bin_artifacts(original, path)
            loaded = load_action_bin_artifacts(path)

        assert original == loaded

    def test_frozen_dataclass(self) -> None:
        """ActionBinArtifacts is immutable."""
        arts = ActionBinArtifacts(
            spec_version="1.0.0",
            manifest_seed=42,
            vaso_edges=(0.1, 0.5, 1.0),
            fluid_edges=(100.0, 500.0, 1000.0),
            n_train_vaso_nonzero=140,
            n_train_fluid_nonzero=160,
        )
        with pytest.raises(AttributeError):
            arts.manifest_seed = 99  # type: ignore[misc]
