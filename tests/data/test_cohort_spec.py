"""
Guardrail tests for cohort spec completeness.

These tests verify that the CohortSpec exposes all required rule
categories (adult-only gating, ICU stay scope, Sepsis-3 flag, explicit
inclusion / exclusion) and that config bindings remain intact across
future edits.

They are deliberately narrow — they test the *contract*, not the
implementation details of any extraction query.
"""

from __future__ import annotations

import pytest

from mimic_sepsis_rl.data.cohort.spec import (
    COHORT_SPEC_VERSION,
    CohortSpec,
    ExclusionCriteria,
    InclusionCriteria,
)


# ---------------------------------------------------------------------------
# Spec version
# ---------------------------------------------------------------------------


class TestSpecVersion:
    def test_version_string_present(self):
        assert COHORT_SPEC_VERSION, "COHORT_SPEC_VERSION must be a non-empty string"

    def test_version_propagated_to_spec(self):
        spec = CohortSpec()
        assert spec.version == COHORT_SPEC_VERSION


# ---------------------------------------------------------------------------
# Adult-only gating
# ---------------------------------------------------------------------------


class TestAdultOnlyGating:
    def test_adult_only_flag_exists_on_spec(self):
        spec = CohortSpec()
        assert hasattr(spec, "adult_only"), "CohortSpec must expose an 'adult_only' attribute"

    def test_adult_only_is_true_by_default(self):
        spec = CohortSpec()
        assert spec.adult_only is True, "adult_only must default to True"

    def test_min_age_is_18_by_default(self):
        spec = CohortSpec()
        assert spec.inclusion.min_age_years == 18, (
            "Minimum age must be 18 to enforce adult restriction"
        )

    def test_min_age_must_be_positive_integer(self):
        assert isinstance(InclusionCriteria().min_age_years, int)
        assert InclusionCriteria().min_age_years > 0


# ---------------------------------------------------------------------------
# ICU stay scope
# ---------------------------------------------------------------------------


class TestIcuStayScope:
    def test_require_icu_stay_flag_exists(self):
        spec = CohortSpec()
        assert hasattr(spec.inclusion, "require_icu_stay")

    def test_require_icu_stay_is_true_by_default(self):
        spec = CohortSpec()
        assert spec.inclusion.require_icu_stay is True

    def test_min_los_hours_is_positive(self):
        spec = CohortSpec()
        assert spec.inclusion.min_los_hours > 0, "min_los_hours must be positive"

    def test_min_los_hours_at_least_one_step(self):
        """Minimum LOS must cover at least one 4-hour episode step."""
        spec = CohortSpec()
        assert spec.inclusion.min_los_hours >= 4.0, (
            "min_los_hours must be >= 4.0 to guarantee at least one MDP step"
        )


# ---------------------------------------------------------------------------
# Sepsis-3 flag (inclusion category)
# ---------------------------------------------------------------------------


class TestSepsis3InclusionCategory:
    def test_require_sepsis3_flag_exists(self):
        spec = CohortSpec()
        assert hasattr(spec.inclusion, "require_sepsis3")

    def test_require_sepsis3_is_true_by_default(self):
        spec = CohortSpec()
        assert spec.inclusion.require_sepsis3 is True


# ---------------------------------------------------------------------------
# Explicit exclusion categories
# ---------------------------------------------------------------------------


class TestExclusionRuleCompleteness:
    """Verify that all four planned exclusion categories are present and active."""

    def test_missing_sepsis_anchor_excluded_by_default(self):
        spec = CohortSpec()
        assert spec.exclusion.exclude_missing_sepsis_anchor is True

    def test_readmissions_excluded_by_default(self):
        spec = CohortSpec()
        assert spec.exclusion.exclude_readmissions is True

    def test_missing_demographics_excluded_by_default(self):
        spec = CohortSpec()
        assert spec.exclusion.exclude_missing_demographics is True

    def test_max_age_years_field_exists(self):
        """max_age_years must exist even when set to None (no upper bound)."""
        exc = ExclusionCriteria()
        assert hasattr(exc, "max_age_years")

    def test_max_age_years_none_by_default(self):
        exc = ExclusionCriteria()
        assert exc.max_age_years is None, "Default must be None (no upper age cap)"


# ---------------------------------------------------------------------------
# Config binding
# ---------------------------------------------------------------------------


class TestConfigBinding:
    def test_source_tables_dict_not_empty(self):
        spec = CohortSpec()
        assert spec.source_tables, "source_tables must not be empty"

    def test_required_source_tables_present(self):
        spec = CohortSpec()
        required_aliases = {"icustays", "patients", "admissions", "sepsis3"}
        missing = required_aliases - set(spec.source_tables.keys())
        assert not missing, f"source_tables missing required aliases: {missing}"

    def test_rule_summary_covers_all_keys(self):
        """rule_summary() must expose all inclusion and exclusion fields."""
        spec = CohortSpec()
        summary = spec.rule_summary()
        required_keys = {
            "spec_version",
            "adult_only",
            "min_age_years",
            "require_icu_stay",
            "require_sepsis3",
            "min_los_hours",
            "exclude_missing_sepsis_anchor",
            "exclude_readmissions",
            "exclude_missing_demographics",
            "max_age_years",
        }
        missing = required_keys - set(summary.keys())
        assert not missing, f"rule_summary() missing keys: {missing}"

    def test_description_is_non_empty_string(self):
        spec = CohortSpec()
        assert isinstance(spec.description, str) and spec.description.strip()


# ---------------------------------------------------------------------------
# Immutability (frozen dataclasses)
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_cohort_spec_is_frozen(self):
        spec = CohortSpec()
        with pytest.raises((AttributeError, TypeError)):
            spec.adult_only = False  # type: ignore[misc]

    def test_inclusion_criteria_is_frozen(self):
        inc = InclusionCriteria()
        with pytest.raises((AttributeError, TypeError)):
            inc.min_age_years = 0  # type: ignore[misc]

    def test_exclusion_criteria_is_frozen(self):
        exc = ExclusionCriteria()
        with pytest.raises((AttributeError, TypeError)):
            exc.exclude_readmissions = False  # type: ignore[misc]
