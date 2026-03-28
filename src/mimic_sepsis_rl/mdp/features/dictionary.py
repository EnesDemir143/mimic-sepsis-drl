"""
Clinical feature contract for the MIMIC Sepsis Offline RL pipeline.

This module encodes the *complete* v1 state-feature dictionary as typed,
machine-readable descriptors.  Every field that appears in a state vector
must have a corresponding ``FeatureSpec`` entry here so that downstream
extraction, imputation, normalisation, and reporting stages share a single
source of truth without hidden code edits.

MIMIC-IV source tables
----------------------
chartevents   – Bedside vitals, GCS components, ventilator settings
labevents     – Blood gas, chemistry, haematology panels
inputevents   – IV fluids and vasopressor infusions (cumulative state signal)
outputevents  – Urine output

Item-ID references are drawn from MIMIC-IV v2.2 ``d_items`` /
``d_labitems`` dictionaries.  Where multiple item IDs map to the same
clinical concept they are listed together; the extractor is responsible for
merging them before aggregation.

Usage (CLI validation)
----------------------
    python -m mimic_sepsis_rl.mdp.features.dictionary \\
        --config configs/features/default.yaml --validate

Version history
---------------
v1.0.0  2026-04-01  Initial feature contract for Phase 4.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version sentinel
# ---------------------------------------------------------------------------

FEATURE_SPEC_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FeatureFamily(str, Enum):
    """Broad clinical category grouping related features together."""

    VITALS = "vitals"
    LABS_BLOOD_GAS = "labs_blood_gas"
    LABS_CHEMISTRY = "labs_chemistry"
    LABS_HAEMATOLOGY = "labs_haematology"
    TREATMENTS = "treatments"
    DEMOGRAPHICS = "demographics"
    DERIVED = "derived"


class AggregationRule(str, Enum):
    """How multiple measurements within a 4-hour window are collapsed.

    LAST
        Most recent value before the window end (default for vitals /
        point-in-time labs — reflects the current clinical picture).
    MEAN
        Arithmetic mean of all values in the window.
    MAX
        Maximum value in the window (useful for peak markers such as
        temperature or lactate).
    MIN
        Minimum value in the window (useful for worst-case oxygenation).
    SUM
        Total quantity over the window (fluids, urine output).
    CUMULATIVE
        Running total from episode start to window end (cumulative IV
        fluid and vasopressor exposure).
    """

    LAST = "last"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    CUMULATIVE = "cumulative"


class MissingStrategy(str, Enum):
    """Fallback imputation when no measurement is available in the window.

    FORWARD_FILL
        Carry forward the last observed value from any earlier window
        in the same episode.  First-line strategy for time-series
        features with high measurement frequency.
    MEDIAN_TRAIN
        Replace with the feature-level median computed on the training
        split only.  Applied after forward-fill exhausts all prior
        values.
    ZERO
        Impute as zero.  Appropriate only for cumulative treatment
        quantities where absence genuinely means zero dose.
    NORMAL_VALUE
        Replace with a clinically normal reference value (defined per
        feature).  Used for stable physiological features where
        forward-fill is expected to rarely fail.
    """

    FORWARD_FILL = "forward_fill"
    MEDIAN_TRAIN = "median_train"
    ZERO = "zero"
    NORMAL_VALUE = "normal_value"


# ---------------------------------------------------------------------------
# Feature descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Complete contract for a single state-vector feature.

    Attributes
    ----------
    feature_id : str
        Stable snake_case identifier used as the DataFrame column name.
        Must be unique within the registry.  Never renamed after v1 ships.
    display_name : str
        Human-readable label for documentation, figures, and appendices.
    family : FeatureFamily
        Broad clinical category (vitals, labs, treatments, …).
    source_table : str
        Primary MIMIC-IV table name (``chartevents``, ``labevents``,
        ``inputevents``, ``outputevents``).
    item_ids : tuple[int, ...]
        One or more MIMIC-IV ``itemid`` values that map to this concept.
        The extractor merges duplicate concepts before aggregation.
    unit : str
        Physical unit of the raw measurement (e.g. ``"mmHg"``, ``"mg/dL"``).
    aggregation : AggregationRule
        Window-level aggregation strategy.
    missing_strategy : MissingStrategy
        Imputation fallback when no value is available in a window.
    valid_low : float | None
        Inclusive lower bound for physiologically plausible values.
        Values below this threshold are treated as erroneous and dropped
        before aggregation.
    valid_high : float | None
        Inclusive upper bound for physiologically plausible values.
    clip_low : float | None
        Hard lower clip applied *after* aggregation and *before*
        normalisation.  None means no lower clip.
    clip_high : float | None
        Hard upper clip applied *after* aggregation and *before*
        normalisation.  None means no upper clip.
    normal_value : float | None
        Clinically normal reference value.  Required when
        ``missing_strategy == MissingStrategy.NORMAL_VALUE``.
    include_missingness_flag : bool
        If True the extractor must emit a paired binary feature
        ``{feature_id}_missing`` alongside the imputed value.
        Recommended for labs with sparse measurement schedules.
    description : str
        One-sentence clinical rationale for including this feature.
    """

    feature_id: str
    display_name: str
    family: FeatureFamily
    source_table: str
    item_ids: tuple[int, ...]
    unit: str
    aggregation: AggregationRule
    missing_strategy: MissingStrategy
    valid_low: float | None
    valid_high: float | None
    clip_low: float | None
    clip_high: float | None
    normal_value: float | None
    include_missingness_flag: bool
    description: str


# ---------------------------------------------------------------------------
# Registry builder helpers
# ---------------------------------------------------------------------------


def _spec(
    feature_id: str,
    display_name: str,
    family: FeatureFamily,
    source_table: str,
    item_ids: tuple[int, ...],
    unit: str,
    aggregation: AggregationRule,
    missing_strategy: MissingStrategy,
    valid_low: float | None,
    valid_high: float | None,
    clip_low: float | None,
    clip_high: float | None,
    normal_value: float | None = None,
    include_missingness_flag: bool = False,
    description: str = "",
) -> FeatureSpec:
    """Thin convenience wrapper — keeps registry definition lines short."""
    return FeatureSpec(
        feature_id=feature_id,
        display_name=display_name,
        family=family,
        source_table=source_table,
        item_ids=item_ids,
        unit=unit,
        aggregation=aggregation,
        missing_strategy=missing_strategy,
        valid_low=valid_low,
        valid_high=valid_high,
        clip_low=clip_low,
        clip_high=clip_high,
        normal_value=normal_value,
        include_missingness_flag=include_missingness_flag,
        description=description,
    )


# ---------------------------------------------------------------------------
# v1 Feature registry
# ---------------------------------------------------------------------------
#
# Ordering follows clinical convention: vitals → blood gas → chemistry →
# haematology → treatments → demographics → derived.
#
# MIMIC-IV item IDs are listed as (primary_id, alias_id, …).
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict[str, FeatureSpec] = {}


def _register(*specs: FeatureSpec) -> None:
    for spec in specs:
        if spec.feature_id in FEATURE_REGISTRY:
            raise ValueError(f"Duplicate feature_id in registry: '{spec.feature_id}'")
        FEATURE_REGISTRY[spec.feature_id] = spec


# ── Vitals ──────────────────────────────────────────────────────────────────

_register(
    _spec(
        "heart_rate",
        "Heart Rate",
        FeatureFamily.VITALS,
        "chartevents",
        (220045,),
        "bpm",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=350.0,
        clip_low=0.0,
        clip_high=300.0,
        normal_value=75.0,
        description="Instantaneous heart rate; elevated/depressed values flag haemodynamic instability.",
    ),
    _spec(
        "map",
        "Mean Arterial Pressure",
        FeatureFamily.VITALS,
        "chartevents",
        (220052, 52, 220181, 224),
        "mmHg",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=300.0,
        clip_low=0.0,
        clip_high=200.0,
        normal_value=80.0,
        include_missingness_flag=True,
        description="Mean arterial pressure; primary haemodynamic target in septic shock management.",
    ),
    _spec(
        "sbp",
        "Systolic Blood Pressure",
        FeatureFamily.VITALS,
        "chartevents",
        (220179, 51),
        "mmHg",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=300.0,
        clip_low=0.0,
        clip_high=250.0,
        normal_value=120.0,
        description="Systolic BP; supplements MAP for identifying hypotension pattern.",
    ),
    _spec(
        "dbp",
        "Diastolic Blood Pressure",
        FeatureFamily.VITALS,
        "chartevents",
        (220180, 8368),
        "mmHg",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=250.0,
        clip_low=0.0,
        clip_high=200.0,
        normal_value=70.0,
        description="Diastolic BP; used alongside SBP for pulse-pressure estimation.",
    ),
    _spec(
        "resp_rate",
        "Respiratory Rate",
        FeatureFamily.VITALS,
        "chartevents",
        (220210, 618),
        "breaths/min",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=100.0,
        clip_low=0.0,
        clip_high=80.0,
        normal_value=16.0,
        description="Respiratory rate; tachypnoea is a sepsis criterion and hypoxaemia marker.",
    ),
    _spec(
        "temperature",
        "Body Temperature",
        FeatureFamily.VITALS,
        "chartevents",
        (223762, 676),
        "°C",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=25.0,
        valid_high=45.0,
        clip_low=25.0,
        clip_high=42.0,
        normal_value=37.0,
        description="Core body temperature; fever and hypothermia are both prognostically relevant in sepsis.",
    ),
    _spec(
        "spo2",
        "Peripheral Oxygen Saturation",
        FeatureFamily.VITALS,
        "chartevents",
        (220277, 646),
        "%",
        AggregationRule.MIN,
        MissingStrategy.FORWARD_FILL,
        valid_low=50.0,
        valid_high=100.0,
        clip_low=50.0,
        clip_high=100.0,
        normal_value=98.0,
        include_missingness_flag=True,
        description="SpO2; worst-case within window captures hypoxic episodes critical for ventilation decisions.",
    ),
    _spec(
        "gcs_total",
        "Glasgow Coma Scale — Total",
        FeatureFamily.VITALS,
        "chartevents",
        (220739, 223900, 223901),
        "score",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=3.0,
        valid_high=15.0,
        clip_low=3.0,
        clip_high=15.0,
        normal_value=15.0,
        include_missingness_flag=True,
        description="GCS total (eye + verbal + motor); captures neurological deterioration in septic encephalopathy.",
    ),
    _spec(
        "fio2_vent",
        "Fraction of Inspired Oxygen (ventilator)",
        FeatureFamily.VITALS,
        "chartevents",
        (223835, 3420),
        "fraction",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.21,
        valid_high=1.0,
        clip_low=0.21,
        clip_high=1.0,
        normal_value=0.21,
        include_missingness_flag=True,
        description="FiO2 set on mechanical ventilator; required for PaO2/FiO2 ratio computation.",
    ),
    _spec(
        "peep",
        "Positive End-Expiratory Pressure",
        FeatureFamily.VITALS,
        "chartevents",
        (220339, 224700),
        "cmH2O",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=40.0,
        clip_low=0.0,
        clip_high=35.0,
        normal_value=0.0,
        include_missingness_flag=True,
        description="PEEP; key ventilator setting influencing oxygenation and haemodynamics in ARDS-complicated sepsis.",
    ),
)

# ── Blood Gas ───────────────────────────────────────────────────────────────

_register(
    _spec(
        "pao2",
        "Partial Pressure of Arterial Oxygen",
        FeatureFamily.LABS_BLOOD_GAS,
        "labevents",
        (50821,),
        "mmHg",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=20.0,
        valid_high=600.0,
        clip_low=20.0,
        clip_high=550.0,
        normal_value=95.0,
        include_missingness_flag=True,
        description="PaO2 from arterial blood gas; combined with FiO2 to compute P/F ratio.",
    ),
    _spec(
        "paco2",
        "Partial Pressure of Arterial CO2",
        FeatureFamily.LABS_BLOOD_GAS,
        "labevents",
        (50818,),
        "mmHg",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=5.0,
        valid_high=200.0,
        clip_low=5.0,
        clip_high=150.0,
        normal_value=40.0,
        include_missingness_flag=True,
        description="PaCO2; reflects ventilatory status and metabolic compensation.",
    ),
    _spec(
        "arterial_ph",
        "Arterial pH",
        FeatureFamily.LABS_BLOOD_GAS,
        "labevents",
        (50820,),
        "pH units",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=6.5,
        valid_high=8.0,
        clip_low=6.8,
        clip_high=7.8,
        normal_value=7.4,
        include_missingness_flag=True,
        description="Arterial pH; acidaemia is associated with worse outcomes in septic shock.",
    ),
    _spec(
        "bicarbonate",
        "Serum Bicarbonate",
        FeatureFamily.LABS_BLOOD_GAS,
        "labevents",
        (50882, 50803),
        "mEq/L",
        AggregationRule.LAST,
        MissingStrategy.MEDIAN_TRAIN,
        valid_low=0.0,
        valid_high=60.0,
        clip_low=2.0,
        clip_high=50.0,
        normal_value=24.0,
        include_missingness_flag=True,
        description="HCO3; marker of metabolic acidosis and base deficit in organ dysfunction.",
    ),
    _spec(
        "base_excess",
        "Base Excess",
        FeatureFamily.LABS_BLOOD_GAS,
        "labevents",
        (50802,),
        "mEq/L",
        AggregationRule.LAST,
        MissingStrategy.MEDIAN_TRAIN,
        valid_low=-30.0,
        valid_high=30.0,
        clip_low=-25.0,
        clip_high=25.0,
        normal_value=0.0,
        include_missingness_flag=True,
        description="Base excess; sensitive early marker of tissue hypoperfusion.",
    ),
    _spec(
        "lactate",
        "Serum Lactate",
        FeatureFamily.LABS_BLOOD_GAS,
        "labevents",
        (50813,),
        "mmol/L",
        AggregationRule.MAX,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=30.0,
        clip_low=0.0,
        clip_high=20.0,
        normal_value=1.0,
        include_missingness_flag=True,
        description="Lactate; cornerstone Sepsis-3 organ dysfunction marker; worst-case in window preferred.",
    ),
)

# ── Chemistry ───────────────────────────────────────────────────────────────

_register(
    _spec(
        "creatinine",
        "Serum Creatinine",
        FeatureFamily.LABS_CHEMISTRY,
        "labevents",
        (50912,),
        "mg/dL",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=20.0,
        clip_low=0.0,
        clip_high=15.0,
        normal_value=0.9,
        include_missingness_flag=True,
        description="Creatinine; renal SOFA component; elevated in AKI associated with sepsis.",
    ),
    _spec(
        "bun",
        "Blood Urea Nitrogen",
        FeatureFamily.LABS_CHEMISTRY,
        "labevents",
        (51006,),
        "mg/dL",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=300.0,
        clip_low=0.0,
        clip_high=200.0,
        normal_value=14.0,
        include_missingness_flag=True,
        description="BUN; additional renal function marker; rises with catabolism and fluid shifts.",
    ),
    _spec(
        "bilirubin_total",
        "Total Bilirubin",
        FeatureFamily.LABS_CHEMISTRY,
        "labevents",
        (50885,),
        "mg/dL",
        AggregationRule.LAST,
        MissingStrategy.MEDIAN_TRAIN,
        valid_low=0.0,
        valid_high=50.0,
        clip_low=0.0,
        clip_high=40.0,
        normal_value=0.6,
        include_missingness_flag=True,
        description="Total bilirubin; hepatic SOFA component; jaundice signals severe liver dysfunction.",
    ),
    _spec(
        "albumin",
        "Serum Albumin",
        FeatureFamily.LABS_CHEMISTRY,
        "labevents",
        (50862,),
        "g/dL",
        AggregationRule.LAST,
        MissingStrategy.MEDIAN_TRAIN,
        valid_low=0.5,
        valid_high=6.0,
        clip_low=0.5,
        clip_high=5.5,
        normal_value=4.0,
        include_missingness_flag=True,
        description="Albumin; marker of nutritional status and capillary leak; hypoalbuminaemia worsens oedema.",
    ),
    _spec(
        "sodium",
        "Serum Sodium",
        FeatureFamily.LABS_CHEMISTRY,
        "labevents",
        (50983, 50824),
        "mEq/L",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=100.0,
        valid_high=180.0,
        clip_low=110.0,
        clip_high=170.0,
        normal_value=140.0,
        include_missingness_flag=True,
        description="Sodium; hyper- and hyponatraemia common in sepsis-related fluid dysregulation.",
    ),
    _spec(
        "potassium",
        "Serum Potassium",
        FeatureFamily.LABS_CHEMISTRY,
        "labevents",
        (50971, 50822),
        "mEq/L",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=1.0,
        valid_high=10.0,
        clip_low=1.5,
        clip_high=9.0,
        normal_value=4.0,
        include_missingness_flag=True,
        description="Potassium; dysregulation risks fatal arrhythmias; monitored closely in ICU.",
    ),
    _spec(
        "glucose",
        "Blood Glucose",
        FeatureFamily.LABS_CHEMISTRY,
        "labevents",
        (50931, 50809),
        "mg/dL",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=20.0,
        valid_high=1500.0,
        clip_low=20.0,
        clip_high=1000.0,
        normal_value=110.0,
        include_missingness_flag=True,
        description="Glucose; hyperglycaemia and hypoglycaemia are both harmful in critical illness.",
    ),
    _spec(
        "inr",
        "International Normalised Ratio",
        FeatureFamily.LABS_CHEMISTRY,
        "labevents",
        (51237,),
        "ratio",
        AggregationRule.LAST,
        MissingStrategy.MEDIAN_TRAIN,
        valid_low=0.5,
        valid_high=20.0,
        clip_low=0.5,
        clip_high=15.0,
        normal_value=1.0,
        include_missingness_flag=True,
        description="INR; coagulation SOFA component; coagulopathy is a hallmark of sepsis-induced DIC.",
    ),
    _spec(
        "ptt",
        "Partial Thromboplastin Time",
        FeatureFamily.LABS_CHEMISTRY,
        "labevents",
        (51275,),
        "seconds",
        AggregationRule.LAST,
        MissingStrategy.MEDIAN_TRAIN,
        valid_low=10.0,
        valid_high=200.0,
        clip_low=10.0,
        clip_high=150.0,
        normal_value=30.0,
        include_missingness_flag=True,
        description="PTT; additional coagulation marker; prolonged PTT indicates intrinsic pathway dysfunction.",
    ),
)

# ── Haematology ─────────────────────────────────────────────────────────────

_register(
    _spec(
        "wbc",
        "White Blood Cell Count",
        FeatureFamily.LABS_HAEMATOLOGY,
        "labevents",
        (51301, 51300),
        "×10³/µL",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=500.0,
        clip_low=0.0,
        clip_high=400.0,
        normal_value=8.0,
        include_missingness_flag=True,
        description="WBC; leukocytosis and leucopenia are Sepsis-3 SIRS surrogates.",
    ),
    _spec(
        "haemoglobin",
        "Haemoglobin",
        FeatureFamily.LABS_HAEMATOLOGY,
        "labevents",
        (51222,),
        "g/dL",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=25.0,
        clip_low=1.0,
        clip_high=20.0,
        normal_value=12.0,
        include_missingness_flag=True,
        description="Haemoglobin; anaemia impairs oxygen delivery; relevant for transfusion decisions.",
    ),
    _spec(
        "haematocrit",
        "Haematocrit",
        FeatureFamily.LABS_HAEMATOLOGY,
        "labevents",
        (51221,),
        "%",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=100.0,
        clip_low=5.0,
        clip_high=65.0,
        normal_value=38.0,
        include_missingness_flag=False,
        description="Haematocrit; complements haemoglobin for haemoconcentration / dilution assessment.",
    ),
    _spec(
        "platelets",
        "Platelet Count",
        FeatureFamily.LABS_HAEMATOLOGY,
        "labevents",
        (51265,),
        "×10³/µL",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=3000.0,
        clip_low=0.0,
        clip_high=2000.0,
        normal_value=200.0,
        include_missingness_flag=True,
        description="Platelets; coagulation SOFA component; thrombocytopaenia signals consumptive coagulopathy.",
    ),
)

# ── Treatments (state signal — NOT action) ──────────────────────────────────
#
# These features represent cumulative treatment exposure observed in the
# state vector.  Action-level encoding (vasopressor bins, fluid bins) is
# handled by Phase 5.

_register(
    _spec(
        "cum_iv_fluid_ml",
        "Cumulative IV Fluid Volume",
        FeatureFamily.TREATMENTS,
        "inputevents",
        (
            220949,
            220950,
            225158,
            225159,
            225161,
            225168,
            225828,
            225823,
            225825,
            226089,
            220952,
            220955,
        ),
        "mL",
        AggregationRule.CUMULATIVE,
        MissingStrategy.ZERO,
        valid_low=0.0,
        valid_high=50000.0,
        clip_low=0.0,
        clip_high=30000.0,
        normal_value=0.0,
        include_missingness_flag=False,
        description=(
            "Cumulative IV crystalloid and colloid volume from episode start; "
            "captures overall resuscitation burden."
        ),
    ),
    _spec(
        "cum_vasopressor_dose_nor_equiv",
        "Cumulative Vasopressor Dose (noradrenaline equivalent)",
        FeatureFamily.TREATMENTS,
        "inputevents",
        (221906, 221289, 222315, 221662),
        "µg/kg/min·hours",
        AggregationRule.CUMULATIVE,
        MissingStrategy.ZERO,
        valid_low=0.0,
        valid_high=1000.0,
        clip_low=0.0,
        clip_high=500.0,
        normal_value=0.0,
        include_missingness_flag=False,
        description=(
            "Cumulative vasopressor exposure normalised to noradrenaline equivalents; "
            "reflects haemodynamic support burden."
        ),
    ),
    _spec(
        "urine_output_4h",
        "Urine Output (4-hour window)",
        FeatureFamily.TREATMENTS,
        "outputevents",
        (
            226559,
            226560,
            226561,
            226584,
            226563,
            226564,
            226565,
            226567,
            226557,
            226558,
        ),
        "mL",
        AggregationRule.SUM,
        MissingStrategy.ZERO,
        valid_low=0.0,
        valid_high=2000.0,
        clip_low=0.0,
        clip_high=1500.0,
        normal_value=0.0,
        include_missingness_flag=True,
        description=(
            "Urine output summed over the 4-hour window; "
            "oliguria < 0.5 mL/kg/h is a renal SOFA criterion."
        ),
    ),
)

# ── Demographics ────────────────────────────────────────────────────────────

_register(
    _spec(
        "age_years",
        "Patient Age",
        FeatureFamily.DEMOGRAPHICS,
        "patients",
        (),
        "years",
        AggregationRule.LAST,
        MissingStrategy.MEDIAN_TRAIN,
        valid_low=18.0,
        valid_high=120.0,
        clip_low=18.0,
        clip_high=110.0,
        normal_value=None,
        include_missingness_flag=False,
        description="Patient age at ICU admission; age is an independent predictor of sepsis mortality.",
    ),
    _spec(
        "weight_kg",
        "Admission Body Weight",
        FeatureFamily.DEMOGRAPHICS,
        "chartevents",
        (226512, 224639),
        "kg",
        AggregationRule.LAST,
        MissingStrategy.MEDIAN_TRAIN,
        valid_low=20.0,
        valid_high=400.0,
        clip_low=20.0,
        clip_high=300.0,
        normal_value=None,
        include_missingness_flag=False,
        description="Body weight; required for weight-normalised vasopressor dose and fluid calculations.",
    ),
)

# ── Derived ─────────────────────────────────────────────────────────────────

_register(
    _spec(
        "pf_ratio",
        "PaO2 / FiO2 Ratio",
        FeatureFamily.DERIVED,
        "derived",
        (),
        "mmHg",
        AggregationRule.MIN,
        MissingStrategy.MEDIAN_TRAIN,
        valid_low=10.0,
        valid_high=700.0,
        clip_low=20.0,
        clip_high=600.0,
        normal_value=400.0,
        include_missingness_flag=True,
        description=(
            "PaO2/FiO2 ratio (Berlin/SOFA); worst-case within window. "
            "Computed from pao2 / fio2_vent; requires both parent features present."
        ),
    ),
    _spec(
        "shock_index",
        "Shock Index",
        FeatureFamily.DERIVED,
        "derived",
        (),
        "ratio",
        AggregationRule.LAST,
        MissingStrategy.FORWARD_FILL,
        valid_low=0.0,
        valid_high=10.0,
        clip_low=0.0,
        clip_high=5.0,
        normal_value=0.6,
        include_missingness_flag=False,
        description=(
            "Heart rate / systolic BP; simple bedside haemodynamic instability index. "
            "Computed from heart_rate / sbp."
        ),
    ),
    _spec(
        "hours_since_onset",
        "Hours Since Sepsis Onset",
        FeatureFamily.DERIVED,
        "derived",
        (),
        "hours",
        AggregationRule.LAST,
        MissingStrategy.NORMAL_VALUE,
        valid_low=-24.0,
        valid_high=48.0,
        clip_low=-24.0,
        clip_high=48.0,
        normal_value=0.0,
        include_missingness_flag=False,
        description=(
            "Relative time position within the episode window "
            "(onset − 24h → 0 h → onset + 48h); encodes temporal context."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------


def load_feature_registry(
    config: dict[str, Any] | None = None,
) -> dict[str, FeatureSpec]:
    """Return a filtered and validated feature registry.

    Parameters
    ----------
    config:
        Optional mapping loaded from ``configs/features/default.yaml``.
        Supports the following keys:

        ``include_features`` : list[str] | None
            If provided, only these feature IDs are returned.
            Unknown IDs raise ``ValueError``.
        ``exclude_features`` : list[str]
            Feature IDs to drop from the registry.
        ``missingness_flags_default`` : bool
            Override ``include_missingness_flag`` for *all* features
            when explicitly set.

    Returns
    -------
    dict[str, FeatureSpec]
        Ordered subset of ``FEATURE_REGISTRY`` matching the config.
    """
    registry = dict(FEATURE_REGISTRY)

    if config is None:
        return registry

    include = config.get("include_features")
    exclude = set(config.get("exclude_features", []))
    flag_override: bool | None = config.get("missingness_flags_default")

    if include is not None:
        unknown = set(include) - set(registry)
        if unknown:
            raise ValueError(
                f"unknown feature IDs in include_features: {sorted(unknown)}"
            )
        registry = {fid: registry[fid] for fid in include if fid in registry}

    if exclude:
        unknown_ex = exclude - set(FEATURE_REGISTRY)
        if unknown_ex:
            logger.warning(
                "exclude_features references unknown IDs (ignored): %s",
                sorted(unknown_ex),
            )
        registry = {fid: spec for fid, spec in registry.items() if fid not in exclude}

    if flag_override is not None:
        registry = {
            fid: FeatureSpec(
                feature_id=spec.feature_id,
                display_name=spec.display_name,
                family=spec.family,
                source_table=spec.source_table,
                item_ids=spec.item_ids,
                unit=spec.unit,
                aggregation=spec.aggregation,
                missing_strategy=spec.missing_strategy,
                valid_low=spec.valid_low,
                valid_high=spec.valid_high,
                clip_low=spec.clip_low,
                clip_high=spec.clip_high,
                normal_value=spec.normal_value,
                include_missingness_flag=flag_override,
                description=spec.description,
            )
            for fid, spec in registry.items()
        }

    return registry


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_spec(spec: FeatureSpec) -> list[str]:
    """Return a list of validation error strings for a single spec."""
    errors: list[str] = []
    fid = spec.feature_id

    if not fid:
        errors.append("feature_id must not be empty.")
    if not spec.display_name:
        errors.append(f"{fid}: display_name must not be empty.")
    if not spec.unit:
        errors.append(f"{fid}: unit must not be empty.")
    if not spec.description:
        errors.append(f"{fid}: description must not be empty.")
    if spec.valid_low is not None and spec.valid_high is not None:
        if spec.valid_low >= spec.valid_high:
            errors.append(
                f"{fid}: valid_low ({spec.valid_low}) must be < valid_high ({spec.valid_high})."
            )
    if spec.clip_low is not None and spec.clip_high is not None:
        if spec.clip_low >= spec.clip_high:
            errors.append(
                f"{fid}: clip_low ({spec.clip_low}) must be < clip_high ({spec.clip_high})."
            )
    if (
        spec.missing_strategy == MissingStrategy.NORMAL_VALUE
        and spec.normal_value is None
    ):
        errors.append(
            f"{fid}: normal_value must be set when missing_strategy=NORMAL_VALUE."
        )
    if spec.family in (
        FeatureFamily.VITALS,
        FeatureFamily.LABS_BLOOD_GAS,
        FeatureFamily.LABS_CHEMISTRY,
        FeatureFamily.LABS_HAEMATOLOGY,
        FeatureFamily.TREATMENTS,
    ):
        if spec.source_table == "derived" and spec.item_ids:
            errors.append(f"{fid}: derived features should not declare item_ids.")
    return errors


def validate_registry(
    registry: dict[str, FeatureSpec] | None = None,
) -> tuple[bool, list[str]]:
    """Validate all specs in the registry.

    Parameters
    ----------
    registry:
        Feature registry to validate.  Defaults to ``FEATURE_REGISTRY``.

    Returns
    -------
    (passed, errors)
        ``passed`` is True when there are no errors.
    """
    reg = registry or FEATURE_REGISTRY
    all_errors: list[str] = []

    ids = list(reg.keys())
    if len(ids) != len(set(ids)):
        all_errors.append("Registry contains duplicate feature_ids.")

    for spec in reg.values():
        all_errors.extend(_validate_spec(spec))

    passed = len(all_errors) == 0
    return passed, all_errors


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def registry_summary(registry: dict[str, FeatureSpec] | None = None) -> dict[str, Any]:
    """Return a structured summary dict for reporting."""
    reg = registry or FEATURE_REGISTRY
    families: dict[str, int] = {}
    tables: dict[str, int] = {}
    agg_rules: dict[str, int] = {}
    missing_strategies: dict[str, int] = {}
    flag_features: list[str] = []

    for spec in reg.values():
        families[spec.family.value] = families.get(spec.family.value, 0) + 1
        tables[spec.source_table] = tables.get(spec.source_table, 0) + 1
        agg_rules[spec.aggregation.value] = agg_rules.get(spec.aggregation.value, 0) + 1
        missing_strategies[spec.missing_strategy.value] = (
            missing_strategies.get(spec.missing_strategy.value, 0) + 1
        )
        if spec.include_missingness_flag:
            flag_features.append(spec.feature_id)

    return {
        "spec_version": FEATURE_SPEC_VERSION,
        "total_features": len(reg),
        "total_missingness_flag_features": len(flag_features),
        "families": families,
        "source_tables": tables,
        "aggregation_rules": agg_rules,
        "missing_strategies": missing_strategies,
        "missingness_flag_features": flag_features,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mimic_sepsis_rl.mdp.features.dictionary",
        description="Validate and inspect the feature dictionary contract.",
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to features YAML config (configs/features/default.yaml).",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run spec validation and exit with non-zero status on failure.",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print a JSON summary of the registry.",
    )
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: D401
    """CLI entry point — validate and/or summarise the feature registry."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load optional config
    registry: dict[str, FeatureSpec] | None = None
    if args.config:
        try:
            import yaml  # type: ignore[import]

            with open(args.config) as fh:
                cfg = yaml.safe_load(fh)
            registry = load_feature_registry(cfg)
            logger.info(
                "Loaded config from %s; %d features active.", args.config, len(registry)
            )
        except FileNotFoundError:
            logger.error("Config file not found: %s", args.config)
            return 1
    else:
        registry = dict(FEATURE_REGISTRY)

    exit_code = 0

    if args.validate or (not args.summary):
        passed, errors = validate_registry(registry)
        if passed:
            logger.info("Validation PASSED — %d features in registry.", len(registry))
        else:
            logger.error("Validation FAILED — %d error(s):", len(errors))
            for err in errors:
                logger.error("  • %s", err)
            exit_code = 1

    if args.summary:
        summary = registry_summary(registry)
        print(json.dumps(summary, indent=2))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
