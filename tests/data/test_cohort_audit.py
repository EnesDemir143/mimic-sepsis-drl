"""
Regression tests for cohort extraction, audit, and output models.

Covers:
- CohortResult model structure
- ExclusionReason enum values
- CohortExtractor rule application (age, LOS, demographics, readmissions)
- Audit report formatting
- Completeness validation
- Exclusion summary table
- Edge cases (empty input, all excluded, all included)
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from mimic_sepsis_rl.data.cohort.models import (
    COHORT_MODELS_VERSION,
    CohortResult,
    ExclusionReason,
)
from mimic_sepsis_rl.data.cohort.audit import (
    exclusion_summary_table,
    format_audit_report,
    validate_completeness,
)
from mimic_sepsis_rl.data.cohort.spec import CohortSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_icu_row(
    subject_id: int = 1,
    hadm_id: int = 100,
    stay_id: int = 1000,
    anchor_age: int = 55,
    gender: str = "M",
    los_hours: float = 48.0,
    intime: str = "2150-01-15 12:00:00",
    outtime: str = "2150-01-17 12:00:00",
) -> dict:
    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "stay_id": stay_id,
        "anchor_age": anchor_age,
        "gender": gender,
        "los_hours": los_hours,
        "intime": datetime.strptime(intime, "%Y-%m-%d %H:%M:%S"),
        "outtime": datetime.strptime(outtime, "%Y-%m-%d %H:%M:%S"),
        "first_careunit": "MICU",
        "last_careunit": "MICU",
        "los": los_hours / 24.0,
        "deathtime": None,
        "hospital_expire_flag": 0,
    }


def _make_test_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# ExclusionReason enum
# ---------------------------------------------------------------------------


class TestExclusionReason:
    def test_all_reasons_have_string_values(self):
        for reason in ExclusionReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0

    def test_expected_reasons_exist(self):
        expected = {
            "AGE_BELOW_MIN", "AGE_ABOVE_MAX", "LOS_TOO_SHORT",
            "MISSING_DEMOGRAPHICS", "READMISSION", "NO_SEPSIS_MARKERS",
        }
        actual = {r.name for r in ExclusionReason}
        assert expected.issubset(actual)


# ---------------------------------------------------------------------------
# CohortResult model
# ---------------------------------------------------------------------------


class TestCohortResult:
    def test_result_holds_dataframes(self):
        included = pl.DataFrame({"stay_id": [1, 2]})
        excluded = pl.DataFrame({"stay_id": [3], "exclusion_reason": ["test"]})
        result = CohortResult(included=included, excluded=excluded)
        assert result.included.height == 2
        assert result.excluded.height == 1

    def test_default_audit_is_empty(self):
        result = CohortResult(
            included=pl.DataFrame(),
            excluded=pl.DataFrame(),
        )
        assert result.audit_summary == {}

    def test_version_is_defined(self):
        assert COHORT_MODELS_VERSION
        assert isinstance(COHORT_MODELS_VERSION, str)


# ---------------------------------------------------------------------------
# Rule application (unit tests with synthetic data)
# ---------------------------------------------------------------------------


class TestRuleApplication:
    """Test inclusion/exclusion rules using synthetic data (no file I/O)."""

    def _apply_rules_on_df(self, df: pl.DataFrame, spec: CohortSpec | None = None):
        """Simulate CohortExtractor._apply_rules on a pre-merged DataFrame."""
        from mimic_sepsis_rl.data.cohort.extract import CohortExtractor

        if spec is None:
            spec = CohortSpec()
        extractor = CohortExtractor(spec=spec)
        return extractor._apply_rules(df)

    def test_adult_filter_excludes_minors(self):
        rows = [
            _make_icu_row(subject_id=1, stay_id=1, anchor_age=55),
            _make_icu_row(subject_id=2, stay_id=2, anchor_age=17),
            _make_icu_row(subject_id=3, stay_id=3, anchor_age=18),
        ]
        df = _make_test_df(rows)
        included, excluded = self._apply_rules_on_df(df)
        assert included.height == 2
        assert 2 not in included["stay_id"].to_list()
        assert excluded.height == 1

    def test_los_filter_excludes_short_stays(self):
        rows = [
            _make_icu_row(subject_id=1, stay_id=1, los_hours=48.0),
            _make_icu_row(subject_id=2, stay_id=2, los_hours=2.0),  # below 4h
            _make_icu_row(subject_id=3, stay_id=3, los_hours=4.0),   # exactly 4h
        ]
        df = _make_test_df(rows)
        included, excluded = self._apply_rules_on_df(df)
        included_ids = included["stay_id"].to_list()
        assert 1 in included_ids
        assert 3 in included_ids
        assert 2 not in included_ids

    def test_missing_demographics_excluded(self):
        rows = [
            _make_icu_row(subject_id=1, stay_id=1, anchor_age=30, gender="F"),
            _make_icu_row(subject_id=2, stay_id=2, anchor_age=40, gender=None),
        ]
        df = _make_test_df(rows)
        included, excluded = self._apply_rules_on_df(df)
        assert 1 in included["stay_id"].to_list()

    def test_readmission_keeps_first_stay_only(self):
        rows = [
            _make_icu_row(
                subject_id=1, stay_id=1,
                intime="2150-01-10 12:00:00",
                outtime="2150-01-12 12:00:00",
            ),
            _make_icu_row(
                subject_id=1, stay_id=2,
                intime="2150-02-15 12:00:00",
                outtime="2150-02-17 12:00:00",
            ),
            _make_icu_row(
                subject_id=2, stay_id=3,
                intime="2150-03-01 12:00:00",
                outtime="2150-03-03 12:00:00",
            ),
        ]
        df = _make_test_df(rows)
        included, excluded = self._apply_rules_on_df(df)
        included_ids = included["stay_id"].to_list()
        # Patient 1: only first stay (id=1) kept
        assert 1 in included_ids
        assert 2 not in included_ids
        # Patient 2: only stay (id=3) kept
        assert 3 in included_ids

    def test_all_rules_combined(self):
        rows = [
            _make_icu_row(subject_id=1, stay_id=1, anchor_age=55, los_hours=48.0),  # pass all
            _make_icu_row(subject_id=2, stay_id=2, anchor_age=15, los_hours=48.0),  # fail age
            _make_icu_row(subject_id=3, stay_id=3, anchor_age=30, los_hours=1.0),   # fail LOS
        ]
        df = _make_test_df(rows)
        included, excluded = self._apply_rules_on_df(df)
        assert included.height == 1
        assert included["stay_id"][0] == 1

    def test_max_age_exclusion(self):
        from mimic_sepsis_rl.data.cohort.spec import ExclusionCriteria
        spec = CohortSpec(
            exclusion=ExclusionCriteria(
                max_age_years=80,
                exclude_readmissions=False,
            )
        )
        rows = [
            _make_icu_row(subject_id=1, stay_id=1, anchor_age=55),
            _make_icu_row(subject_id=2, stay_id=2, anchor_age=90),  # above max
        ]
        df = _make_test_df(rows)
        included, excluded = self._apply_rules_on_df(df, spec=spec)
        assert 1 in included["stay_id"].to_list()
        assert 2 not in included["stay_id"].to_list()

    def test_excluded_has_reason_column(self):
        rows = [
            _make_icu_row(stay_id=1, anchor_age=10),  # fail age
        ]
        df = _make_test_df(rows)
        _, excluded = self._apply_rules_on_df(df)
        assert "exclusion_reason" in excluded.columns

    def test_no_duplicate_stay_ids_in_included(self):
        rows = [
            _make_icu_row(subject_id=i, stay_id=i, anchor_age=30 + i)
            for i in range(1, 20)
        ]
        df = _make_test_df(rows)
        included, _ = self._apply_rules_on_df(df)
        assert included.select("stay_id").n_unique() == included.height


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


class TestAuditHelpers:
    def _make_result(self) -> CohortResult:
        included = pl.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "stay_id": [100, 200],
        })
        excluded = pl.DataFrame({
            "subject_id": [3, 4, 5],
            "hadm_id": [30, 40, 50],
            "stay_id": [300, 400, 500],
            "exclusion_reason": [
                ExclusionReason.AGE_BELOW_MIN.value,
                ExclusionReason.LOS_TOO_SHORT.value,
                ExclusionReason.AGE_BELOW_MIN.value,
            ],
        })
        return CohortResult(
            included=included,
            excluded=excluded,
            audit_summary={
                "spec_version": "1.0.0",
                "models_version": COHORT_MODELS_VERSION,
                "total_icu_stays": 5,
                "included": 2,
                "excluded": 3,
                "inclusion_rate_pct": 40.0,
                "exclusion_reasons": {
                    ExclusionReason.AGE_BELOW_MIN.value: 2,
                    ExclusionReason.LOS_TOO_SHORT.value: 1,
                },
                "rules_applied": [
                    {"rule": "age_below_min", "excluded_count": 2},
                    {"rule": "los_too_short", "excluded_count": 1},
                ],
                "unique_patients": 2,
            },
        )

    def test_format_audit_report_returns_string(self):
        result = self._make_result()
        report = format_audit_report(result)
        assert isinstance(report, str)
        assert "Cohort Extraction Audit Report" in report
        assert "40.00%" in report

    def test_validate_completeness_passes(self):
        result = self._make_result()
        errors = validate_completeness(result, total_before=5)
        assert errors == []

    def test_validate_completeness_fails_on_mismatch(self):
        result = self._make_result()
        errors = validate_completeness(result, total_before=999)
        assert len(errors) > 0
        assert "Count mismatch" in errors[0]

    def test_validate_completeness_detects_null_reasons(self):
        included = pl.DataFrame({
            "subject_id": [1], "hadm_id": [10], "stay_id": [100],
        })
        excluded = pl.DataFrame({
            "subject_id": [2], "hadm_id": [20], "stay_id": [200],
            "exclusion_reason": [None],
        })
        result = CohortResult(included=included, excluded=excluded)
        errors = validate_completeness(result, total_before=2)
        assert any("null exclusion_reason" in e for e in errors)

    def test_exclusion_summary_table(self):
        result = self._make_result()
        summary = exclusion_summary_table(result)
        assert summary.height == 2
        assert "exclusion_reason" in summary.columns
        assert "count" in summary.columns
        assert "pct" in summary.columns

    def test_exclusion_summary_empty(self):
        result = CohortResult(
            included=pl.DataFrame({"stay_id": [1]}),
            excluded=pl.DataFrame(schema={
                "stay_id": pl.Int64,
                "exclusion_reason": pl.Utf8,
            }),
        )
        summary = exclusion_summary_table(result)
        assert summary.height == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_stays_pass(self):
        from mimic_sepsis_rl.data.cohort.extract import CohortExtractor
        from mimic_sepsis_rl.data.cohort.spec import ExclusionCriteria, InclusionCriteria

        spec = CohortSpec(
            inclusion=InclusionCriteria(min_age_years=0, min_los_hours=0),
            exclusion=ExclusionCriteria(
                exclude_readmissions=False,
                exclude_missing_demographics=False,
            ),
        )
        rows = [_make_icu_row(stay_id=i, subject_id=i) for i in range(1, 5)]
        df = _make_test_df(rows)
        extractor = CohortExtractor(spec=spec)
        included, excluded = extractor._apply_rules(df)
        assert included.height == 4
        assert excluded.height == 0

    def test_all_stays_excluded(self):
        from mimic_sepsis_rl.data.cohort.extract import CohortExtractor

        rows = [
            _make_icu_row(stay_id=i, subject_id=i, anchor_age=10)
            for i in range(1, 5)
        ]
        df = _make_test_df(rows)
        extractor = CohortExtractor(spec=CohortSpec())
        included, excluded = extractor._apply_rules(df)
        assert included.height == 0
        assert excluded.height == 4
