"""
Cohort audit helpers.

This module provides utilities for summarising, formatting, and validating
cohort extraction audit records.  It works with the ``CohortResult`` output
from ``extract.py`` and the ``ExclusionReason`` enum from ``models.py``.

Usage
-----
    from mimic_sepsis_rl.data.cohort.audit import (
        format_audit_report,
        validate_completeness,
    )

Version history
---------------
v1.0.0  2026-03-28  Initial audit helpers.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from mimic_sepsis_rl.data.cohort.models import CohortResult, ExclusionReason


def format_audit_report(result: CohortResult) -> str:
    """Format a human-readable audit report from a CohortResult.

    Parameters
    ----------
    result:
        Completed extraction result with audit_summary populated.

    Returns
    -------
    Formatted multi-line string.
    """
    audit = result.audit_summary
    lines = [
        "=" * 60,
        "  Cohort Extraction Audit Report",
        "=" * 60,
        f"  Spec version    : {audit.get('spec_version', 'N/A')}",
        f"  Models version  : {audit.get('models_version', 'N/A')}",
        "",
        f"  Total ICU stays : {audit.get('total_icu_stays', 0):,}",
        f"  Included        : {audit.get('included', 0):,}",
        f"  Excluded        : {audit.get('excluded', 0):,}",
        f"  Inclusion rate  : {audit.get('inclusion_rate_pct', 0):.2f}%",
        f"  Unique patients : {audit.get('unique_patients', 0):,}",
        "",
        "  Exclusion Breakdown:",
    ]

    reasons = audit.get("exclusion_reasons", {})
    if reasons:
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"    {reason:<40} {count:>6,}")
    else:
        lines.append("    (none)")

    lines.extend([
        "",
        "  Rules Applied (in order):",
    ])

    rules = audit.get("rules_applied", [])
    if rules:
        for i, rule in enumerate(rules, 1):
            lines.append(
                f"    {i}. {rule['rule']:<35} −{rule['excluded_count']:>6,}"
            )
    else:
        lines.append("    (none)")

    lines.append("=" * 60)
    return "\n".join(lines)


def validate_completeness(
    result: CohortResult,
    total_before: int,
) -> list[str]:
    """Validate that included + excluded counts equal the original total.

    Parameters
    ----------
    result:
        Completed extraction result.
    total_before:
        Total stays before any filtering.

    Returns
    -------
    List of error messages (empty if valid).
    """
    errors: list[str] = []

    n_included = result.included.height
    n_excluded = result.excluded.height

    # Note: a stay can appear in excluded multiple times if it failed
    # multiple rules, but our sequential approach ensures each stay
    # is excluded exactly once (at the first failing rule).
    if n_included + n_excluded != total_before:
        errors.append(
            f"Count mismatch: included ({n_included}) + excluded ({n_excluded}) "
            f"= {n_included + n_excluded}, expected {total_before}"
        )

    # Check that all included stays have required columns
    required_cols = {"subject_id", "hadm_id", "stay_id"}
    missing = required_cols - set(result.included.columns)
    if missing:
        errors.append(f"Included DataFrame missing columns: {missing}")

    # Check that all excluded stays have a reason
    if result.excluded.height > 0:
        if "exclusion_reason" not in result.excluded.columns:
            errors.append("Excluded DataFrame missing 'exclusion_reason' column")
        else:
            null_reasons = result.excluded.filter(
                pl.col("exclusion_reason").is_null()
            ).height
            if null_reasons > 0:
                errors.append(
                    f"{null_reasons} excluded stays have null exclusion_reason"
                )

    # Check included stays have no duplicates
    if n_included > 0:
        n_unique = result.included.select("stay_id").n_unique()
        if n_unique != n_included:
            errors.append(
                f"Included has {n_included - n_unique} duplicate stay_ids"
            )

    return errors


def exclusion_summary_table(result: CohortResult) -> pl.DataFrame:
    """Return a summary table of exclusion reason counts.

    Parameters
    ----------
    result:
        Completed extraction result.

    Returns
    -------
    DataFrame with columns: exclusion_reason, count, pct
    """
    if result.excluded.height == 0 or "exclusion_reason" not in result.excluded.columns:
        return pl.DataFrame(schema={
            "exclusion_reason": pl.Utf8,
            "count": pl.Int64,
            "pct": pl.Float64,
        })

    total = result.excluded.height
    summary = (
        result.excluded
        .group_by("exclusion_reason")
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / total * 100).round(2).alias("pct")
        )
        .sort("count", descending=True)
    )
    return summary
