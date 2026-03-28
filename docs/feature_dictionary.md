# Feature Dictionary — MIMIC Sepsis Offline RL

**Phase:** 04 — State Representation Pipeline  
**Plan:** 04-01 — Feature Contract  
**Spec version:** 1.0.0  
**Last updated:** 2026-04-01  
**Executable contract:** `src/mimic_sepsis_rl/mdp/features/dictionary.py`

---

## 1. Purpose

This document is the human-readable companion to the executable feature dictionary
(`dictionary.py`).  Every state feature that appears in the continuous state vector
of the MDP is defined here with its clinical rationale, MIMIC-IV source, unit,
within-window aggregation rule, and missingness handling policy.

The feature set is drawn from the Sepsis-3 / SOFA literature and the MIMIC-IV
schema.  Features are grouped into six clinical families:

| Family | Count | Description |
|--------|------:|-------------|
| Vitals | 9 | Bedside monitor readings |
| Labs — Blood Gas | 6 | Arterial blood gas panel |
| Labs — Chemistry | 9 | Electrolytes, renal, hepatic, coagulation |
| Labs — Haematology | 4 | Full blood count |
| Treatments | 3 | Cumulative fluid and vasopressor exposure, urine output |
| Demographics | 2 | Static per-episode patient attributes |
| Derived | 3 | Computed from base features |
| **Total** | **36** | |

---

## 2. Terminology

| Term | Definition |
|------|-----------|
| **Feature ID** | Stable snake_case column name used throughout the pipeline. Never renamed after v1 ships. |
| **Aggregation rule** | How multiple measurements within a 4-hour window are collapsed to a scalar. |
| **Valid range** | Physiologically plausible bounds.  Values outside this range are dropped *before* aggregation. |
| **Clip range** | Hard bounds applied *after* aggregation, *before* normalisation. |
| **Missing strategy** | Imputation fallback when no valid measurement exists in the window. |
| **Missingness flag** | Binary column `{feature_id}_missing` indicating the value was imputed (1) or observed (0). |

### Aggregation rules

| Rule | Meaning |
|------|---------|
| `last` | Most recent value before window end — reflects current clinical status. |
| `mean` | Arithmetic mean of all valid measurements in the window. |
| `max` | Worst-case high value (e.g. peak lactate). |
| `min` | Worst-case low value (e.g. minimum SpO2). |
| `sum` | Total quantity within the 4-hour window (e.g. urine output). |
| `cumulative` | Running total from episode start to window end (e.g. cumulative IV fluids). |

### Imputation strategies (applied in order)

| Strategy | When applied |
|----------|-------------|
| `forward_fill` | Carry last observed value from an earlier step in the same episode; falls back to `median_train` if no prior exists. |
| `median_train` | Feature-level median computed on the training split only (leakage-safe). |
| `zero` | Impute as zero — used only for cumulative treatment quantities where absence genuinely means zero dose. |
| `normal_value` | Clinically normal reference value defined per feature — last resort fallback. |

---

## 3. MIMIC-IV Source Tables

| Table | Content |
|-------|---------|
| `chartevents` | Bedside vital signs, GCS components, ventilator settings (item IDs from `d_items`). |
| `labevents` | Blood gas, chemistry, haematology results (item IDs from `d_labitems`). |
| `inputevents` | IV fluid and vasopressor infusion records (amount in mL or mcg/kg/min). |
| `outputevents` | Urine and other output records (value in mL). |
| `patients` | Static demographics: anchor age, gender. |
| `derived` | Computed inside the pipeline from base features — no MIMIC-IV table. |

Item IDs listed in each feature entry correspond to MIMIC-IV v2.2
`d_items` / `d_labitems` dictionaries.  Where multiple IDs map to the
same clinical concept they are merged before aggregation.

---

## 4. Feature Definitions

### 4.1 Vitals

All vitals are sourced from `chartevents`.

---

#### `heart_rate` — Heart Rate

| Attribute | Value |
|-----------|-------|
| Display name | Heart Rate |
| Source table | chartevents |
| Item IDs | 220045 |
| Unit | bpm |
| Aggregation | `last` |
| Valid range | [0, 350] bpm |
| Clip range | [0, 300] bpm |
| Missing strategy | `forward_fill` → normal value 75 bpm |
| Missingness flag | No |

**Clinical rationale:**  Instantaneous heart rate is the most frequently charted
vital sign in the ICU.  Elevated (>100 bpm) and depressed (<60 bpm) values flag
haemodynamic instability and are Sepsis-3 SIRS surrogates.

---

#### `map` — Mean Arterial Pressure

| Attribute | Value |
|-----------|-------|
| Display name | Mean Arterial Pressure |
| Source table | chartevents |
| Item IDs | 220052, 52, 220181, 224 |
| Unit | mmHg |
| Aggregation | `last` |
| Valid range | [0, 300] mmHg |
| Clip range | [0, 200] mmHg |
| Missing strategy | `forward_fill` → normal value 80 mmHg |
| Missingness flag | **Yes** |

**Clinical rationale:**  MAP ≥ 65 mmHg is the primary haemodynamic target in
septic shock management (Surviving Sepsis Campaign guidelines).  Multiple item IDs
cover arterial-line and non-invasive cuff measurements.

---

#### `sbp` — Systolic Blood Pressure

| Attribute | Value |
|-----------|-------|
| Display name | Systolic Blood Pressure |
| Source table | chartevents |
| Item IDs | 220179, 51 |
| Unit | mmHg |
| Aggregation | `last` |
| Valid range | [0, 300] mmHg |
| Clip range | [0, 250] mmHg |
| Missing strategy | `forward_fill` → normal value 120 mmHg |
| Missingness flag | No |

**Clinical rationale:**  SBP supplements MAP for identifying hypotension patterns
and is required for computing the shock index derived feature.

---

#### `dbp` — Diastolic Blood Pressure

| Attribute | Value |
|-----------|-------|
| Display name | Diastolic Blood Pressure |
| Source table | chartevents |
| Item IDs | 220180, 8368 |
| Unit | mmHg |
| Aggregation | `last` |
| Valid range | [0, 250] mmHg |
| Clip range | [0, 200] mmHg |
| Missing strategy | `forward_fill` → normal value 70 mmHg |
| Missingness flag | No |

**Clinical rationale:**  DBP alongside SBP enables pulse-pressure estimation;
wide pulse pressure is a marker of vasodilatory shock.

---

#### `resp_rate` — Respiratory Rate

| Attribute | Value |
|-----------|-------|
| Display name | Respiratory Rate |
| Source table | chartevents |
| Item IDs | 220210, 618 |
| Unit | breaths/min |
| Aggregation | `last` |
| Valid range | [0, 100] breaths/min |
| Clip range | [0, 80] breaths/min |
| Missing strategy | `forward_fill` → normal value 16 breaths/min |
| Missingness flag | No |

**Clinical rationale:**  Tachypnoea (>22/min) is a Sepsis-3 clinical criterion
and an indirect marker of hypoxaemia and metabolic acidosis.

---

#### `temperature` — Body Temperature

| Attribute | Value |
|-----------|-------|
| Display name | Body Temperature |
| Source table | chartevents |
| Item IDs | 223762, 676 |
| Unit | °C |
| Aggregation | `last` |
| Valid range | [25, 45] °C |
| Clip range | [25, 42] °C |
| Missing strategy | `forward_fill` → normal value 37.0 °C |
| Missingness flag | No |

**Clinical rationale:**  Both fever (>38.3 °C) and hypothermia (<36 °C) are
prognostically relevant in sepsis; MIMIC-IV stores temperatures in Celsius
(item 223762) and Fahrenheit (item 676 — converted by extractor).

---

#### `spo2` — Peripheral Oxygen Saturation

| Attribute | Value |
|-----------|-------|
| Display name | Peripheral Oxygen Saturation |
| Source table | chartevents |
| Item IDs | 220277, 646 |
| Unit | % |
| Aggregation | `min` |
| Valid range | [50, 100] % |
| Clip range | [50, 100] % |
| Missing strategy | `forward_fill` → normal value 98 % |
| Missingness flag | **Yes** |

**Clinical rationale:**  Worst-case SpO2 within the window captures the nadir
of oxygenation — a single hypoxic episode is clinically significant even if
other measurements in the window are normal.

---

#### `gcs_total` — Glasgow Coma Scale Total

| Attribute | Value |
|-----------|-------|
| Display name | Glasgow Coma Scale — Total |
| Source table | chartevents |
| Item IDs | 220739, 223900, 223901 |
| Unit | score [3–15] |
| Aggregation | `last` |
| Valid range | [3, 15] |
| Clip range | [3, 15] |
| Missing strategy | `forward_fill` → normal value 15 |
| Missingness flag | **Yes** |

**Clinical rationale:**  GCS is the neurological SOFA component; declining
scores indicate septic encephalopathy or sedation-confounded mentation.
Item IDs cover eye, verbal, and motor sub-scores summed to total.

---

#### `fio2_vent` — FiO2 (ventilator)

| Attribute | Value |
|-----------|-------|
| Display name | Fraction of Inspired Oxygen (ventilator) |
| Source table | chartevents |
| Item IDs | 223835, 3420 |
| Unit | fraction [0.21–1.0] |
| Aggregation | `last` |
| Valid range | [0.21, 1.0] |
| Clip range | [0.21, 1.0] |
| Missing strategy | `forward_fill` → normal value 0.21 |
| Missingness flag | **Yes** |

**Clinical rationale:**  Required as the denominator for the PaO2/FiO2 ratio
(P/F ratio) — the SOFA respiratory score component.

---

#### `peep` — Positive End-Expiratory Pressure

| Attribute | Value |
|-----------|-------|
| Display name | Positive End-Expiratory Pressure |
| Source table | chartevents |
| Item IDs | 220339, 224700 |
| Unit | cmH₂O |
| Aggregation | `last` |
| Valid range | [0, 40] cmH₂O |
| Clip range | [0, 35] cmH₂O |
| Missing strategy | `forward_fill` → normal value 0.0 cmH₂O |
| Missingness flag | **Yes** |

**Clinical rationale:**  PEEP is a key ventilator setting; high PEEP impairs
venous return and cardiac output, directly affecting haemodynamic management
in ARDS-complicated sepsis.

---

### 4.2 Labs — Blood Gas

All blood gas features are sourced from `labevents`.

---

#### `pao2` — PaO2

| Attribute | Value |
|-----------|-------|
| Display name | Partial Pressure of Arterial Oxygen |
| Source table | labevents |
| Item IDs | 50821 |
| Unit | mmHg |
| Aggregation | `last` |
| Valid range | [20, 600] mmHg |
| Clip range | [20, 550] mmHg |
| Missing strategy | `forward_fill` → normal value 95 mmHg |
| Missingness flag | **Yes** |

**Clinical rationale:**  Combined with FiO2 to compute the PaO2/FiO2 ratio;
sparse measurement schedule makes missingness flag informative.

---

#### `paco2` — PaCO2

| Attribute | Value |
|-----------|-------|
| Display name | Partial Pressure of Arterial CO2 |
| Source table | labevents |
| Item IDs | 50818 |
| Unit | mmHg |
| Aggregation | `last` |
| Valid range | [5, 200] mmHg |
| Clip range | [5, 150] mmHg |
| Missing strategy | `forward_fill` → normal value 40 mmHg |
| Missingness flag | **Yes** |

**Clinical rationale:**  Reflects ventilatory status and metabolic compensation;
hypercapnia indicates ventilatory failure and hypocapnia indicates compensatory
hyperventilation for metabolic acidosis.

---

#### `arterial_ph` — Arterial pH

| Attribute | Value |
|-----------|-------|
| Display name | Arterial pH |
| Source table | labevents |
| Item IDs | 50820 |
| Unit | pH units |
| Aggregation | `last` |
| Valid range | [6.5, 8.0] |
| Clip range | [6.8, 7.8] |
| Missing strategy | `forward_fill` → normal value 7.4 |
| Missingness flag | **Yes** |

**Clinical rationale:**  Acidaemia (pH < 7.35) is independently associated with
worse outcomes in septic shock; required for base excess interpretation.

---

#### `bicarbonate` — Serum Bicarbonate

| Attribute | Value |
|-----------|-------|
| Display name | Serum Bicarbonate |
| Source table | labevents |
| Item IDs | 50882, 50803 |
| Unit | mEq/L |
| Aggregation | `last` |
| Valid range | [0, 60] mEq/L |
| Clip range | [2, 50] mEq/L |
| Missing strategy | `median_train` → normal value 24 mEq/L |
| Missingness flag | **Yes** |

**Clinical rationale:**  HCO3 is a primary marker of metabolic acidosis; low
bicarbonate with elevated lactate signals tissue hypoperfusion.

---

#### `base_excess` — Base Excess

| Attribute | Value |
|-----------|-------|
| Display name | Base Excess |
| Source table | labevents |
| Item IDs | 50802 |
| Unit | mEq/L |
| Aggregation | `last` |
| Valid range | [−30, 30] mEq/L |
| Clip range | [−25, 25] mEq/L |
| Missing strategy | `median_train` → normal value 0.0 mEq/L |
| Missingness flag | **Yes** |

**Clinical rationale:**  Base excess is a sensitive early marker of tissue
hypoperfusion and guides resuscitation endpoints.

---

#### `lactate` — Serum Lactate

| Attribute | Value |
|-----------|-------|
| Display name | Serum Lactate |
| Source table | labevents |
| Item IDs | 50813 |
| Unit | mmol/L |
| Aggregation | `max` |
| Valid range | [0, 30] mmol/L |
| Clip range | [0, 20] mmol/L |
| Missing strategy | `forward_fill` → normal value 1.0 mmol/L |
| Missingness flag | **Yes** |

**Clinical rationale:**  Lactate ≥ 2 mmol/L (or ≥ 4 mmol/L with hypotension)
is a cornerstone Sepsis-3 organ dysfunction marker.  Worst-case within the
window is preferred to detect transient hypoperfusion episodes.

---

### 4.3 Labs — Chemistry

All chemistry features are sourced from `labevents`.

---

#### `creatinine` — Serum Creatinine

| Attribute | Value |
|-----------|-------|
| Display name | Serum Creatinine |
| Item IDs | 50912 |
| Unit | mg/dL |
| Aggregation | `last` |
| Valid range | [0, 20] mg/dL |
| Clip range | [0, 15] mg/dL |
| Missing strategy | `forward_fill` → normal value 0.9 mg/dL |
| Missingness flag | **Yes** |

**Clinical rationale:**  Renal SOFA component; AKI is the most common organ
dysfunction in sepsis and directly informs fluid and vasopressor decisions.

---

#### `bun` — Blood Urea Nitrogen

| Attribute | Value |
|-----------|-------|
| Display name | Blood Urea Nitrogen |
| Item IDs | 51006 |
| Unit | mg/dL |
| Aggregation | `last` |
| Valid range | [0, 300] mg/dL |
| Clip range | [0, 200] mg/dL |
| Missing strategy | `forward_fill` → normal value 14 mg/dL |
| Missingness flag | **Yes** |

**Clinical rationale:**  BUN rises with catabolism and GI bleeding in addition
to renal dysfunction; provides complementary information to creatinine.

---

#### `bilirubin_total` — Total Bilirubin

| Attribute | Value |
|-----------|-------|
| Display name | Total Bilirubin |
| Item IDs | 50885 |
| Unit | mg/dL |
| Aggregation | `last` |
| Valid range | [0, 50] mg/dL |
| Clip range | [0, 40] mg/dL |
| Missing strategy | `median_train` → normal value 0.6 mg/dL |
| Missingness flag | **Yes** |

**Clinical rationale:**  Hepatic SOFA component; jaundice (bilirubin > 2 mg/dL)
signals severe hepatic dysfunction in sepsis.

---

#### `albumin` — Serum Albumin

| Attribute | Value |
|-----------|-------|
| Display name | Serum Albumin |
| Item IDs | 50862 |
| Unit | g/dL |
| Aggregation | `last` |
| Valid range | [0.5, 6.0] g/dL |
| Clip range | [0.5, 5.5] g/dL |
| Missing strategy | `median_train` → normal value 4.0 g/dL |
| Missingness flag | **Yes** |

**Clinical rationale:**  Hypoalbuminaemia (< 3.5 g/dL) worsens capillary
oncotic pressure, exacerbates oedema, and reflects nutritional depletion in
critical illness.

---

#### `sodium` — Serum Sodium

| Attribute | Value |
|-----------|-------|
| Display name | Serum Sodium |
| Item IDs | 50983, 50824 |
| Unit | mEq/L |
| Aggregation | `last` |
| Valid range | [100, 180] mEq/L |
| Clip range | [110, 170] mEq/L |
| Missing strategy | `forward_fill` → normal value 140 mEq/L |
| Missingness flag | **Yes** |

**Clinical rationale:**  Dysnatraemia is common in sepsis-related fluid shifts
and ADH dysregulation; both extremes are associated with adverse outcomes.

---

#### `potassium` — Serum Potassium

| Attribute | Value |
|-----------|-------|
| Display name | Serum Potassium |
| Item IDs | 50971, 50822 |
| Unit | mEq/L |
| Aggregation | `last` |
| Valid range | [1, 10] mEq/L |
| Clip range | [1.5, 9.0] mEq/L |
| Missing strategy | `forward_fill` → normal value 4.0 mEq/L |
| Missingness flag | **Yes** |

**Clinical rationale:**  Hyperkalaemia and hypokalaemia risk fatal arrhythmias;
potassium is closely monitored in the ICU.

---

#### `glucose` — Blood Glucose

| Attribute | Value |
|-----------|-------|
| Display name | Blood Glucose |
| Item IDs | 50931, 50809 |
| Unit | mg/dL |
| Aggregation | `last` |
| Valid range | [20, 1500] mg/dL |
| Clip range | [20, 1000] mg/dL |
| Missing strategy | `forward_fill` → normal value 110 mg/dL |
| Missingness flag | **Yes** |

**Clinical rationale:**  Both hyperglycaemia and hypoglycaemia are independently
harmful in critical illness; glucose management is a core ICU nursing task.

---

#### `inr` — International Normalised Ratio

| Attribute | Value |
|-----------|-------|
| Display name | International Normalised Ratio |
| Item IDs | 51237 |
| Unit | ratio |
| Aggregation | `last` |
| Valid range | [0.5, 20] |
| Clip range | [0.5, 15] |
| Missing strategy | `median_train` → normal value 1.0 |
| Missingness flag | **Yes** |

**Clinical rationale:**  Coagulation SOFA component; coagulopathy is a hallmark
of sepsis-induced disseminated intravascular coagulation (DIC).

---

#### `ptt` — Partial Thromboplastin Time

| Attribute | Value |
|-----------|-------|
| Display name | Partial Thromboplastin Time |
| Item IDs | 51275 |
| Unit | seconds |
| Aggregation | `last` |
| Valid range | [10, 200] seconds |
| Clip range | [10, 150] seconds |
| Missing strategy | `median_train` → normal value 30 s |
| Missingness flag | **Yes** |

**Clinical rationale:**  PTT measures the intrinsic coagulation pathway;
prolonged PTT confirms DIC progression and guides anticoagulation decisions.

---

### 4.4 Labs — Haematology

All haematology features are sourced from `labevents`.

---

#### `wbc` — White Blood Cell Count

| Attribute | Value |
|-----------|-------|
| Display name | White Blood Cell Count |
| Item IDs | 51301, 51300 |
| Unit | ×10³/µL |
| Aggregation | `last` |
| Valid range | [0, 500] ×10³/µL |
| Clip range | [0, 400] ×10³/µL |
| Missing strategy | `forward_fill` → normal value 8.0 ×10³/µL |
| Missingness flag | **Yes** |

**Clinical rationale:**  WBC > 12 or < 4 ×10³/µL are Sepsis-3 SIRS criteria
surrogates; severe leucopenia may indicate bone marrow suppression.

---

#### `haemoglobin` — Haemoglobin

| Attribute | Value |
|-----------|-------|
| Display name | Haemoglobin |
| Item IDs | 51222 |
| Unit | g/dL |
| Aggregation | `last` |
| Valid range | [0, 25] g/dL |
| Clip range | [1, 20] g/dL |
| Missing strategy | `forward_fill` → normal value 12.0 g/dL |
| Missingness flag | **Yes** |

**Clinical rationale:**  Anaemia impairs oxygen delivery (DO2); transfusion
decisions and vasopressor titration are directly affected.

---

#### `haematocrit` — Haematocrit

| Attribute | Value |
|-----------|-------|
| Display name | Haematocrit |
| Item IDs | 51221 |
| Unit | % |
| Aggregation | `last` |
| Valid range | [0, 100] % |
| Clip range | [5, 65] % |
| Missing strategy | `forward_fill` → normal value 38 % |
| Missingness flag | No |

**Clinical rationale:**  Haematocrit complements haemoglobin for diagnosing
haemoconcentration (dehydration) or haemodilution (aggressive fluid resuscitation).

---

#### `platelets` — Platelet Count

| Attribute | Value |
|-----------|-------|
| Display name | Platelet Count |
| Item IDs | 51265 |
| Unit | ×10³/µL |
| Aggregation | `last` |
| Valid range | [0, 3000] ×10³/µL |
| Clip range | [0, 2000] ×10³/µL |
| Missing strategy | `forward_fill` → normal value 200 ×10³/µL |
| Missingness flag | **Yes** |

**Clinical rationale:**  Coagulation SOFA component; thrombocytopaenia < 100
×10³/µL signals consumptive coagulopathy and is prognostically important.

---

### 4.5 Treatments

Treatment features represent the *observed* cumulative drug and fluid exposure
in the state vector.  They are **not** action features — action encoding
(vasopressor bins and fluid bins) is handled by Phase 5.

---

#### `cum_iv_fluid_ml` — Cumulative IV Fluid Volume

| Attribute | Value |
|-----------|-------|
| Display name | Cumulative IV Fluid Volume |
| Source table | inputevents |
| Item IDs | 220949, 220950, 225158, 225159, 225161, 225168, 225828, 225823, 225825, 226089, 220952, 220955 |
| Unit | mL |
| Aggregation | `cumulative` (episode start → window end) |
| Valid range | [0, 50 000] mL |
| Clip range | [0, 30 000] mL |
| Missing strategy | `zero` |
| Missingness flag | No |

**Clinical rationale:**  Cumulative fluid balance is a key determinant of
treatment intensity and outcome in sepsis; captures resuscitation burden
from episode start.  Covers crystalloid (NS, LR, D5W) and colloid (albumin,
hetastarch) item IDs.

---

#### `cum_vasopressor_dose_nor_equiv` — Cumulative Vasopressor Dose

| Attribute | Value |
|-----------|-------|
| Display name | Cumulative Vasopressor Dose (noradrenaline equivalent) |
| Source table | inputevents |
| Item IDs | 221906 (norepinephrine), 221289 (epinephrine), 222315 (vasopressin), 221662 (dopamine) |
| Unit | µg/kg/min · hours |
| Aggregation | `cumulative` |
| Valid range | [0, 1 000] |
| Clip range | [0, 500] |
| Missing strategy | `zero` |
| Missingness flag | No |

**Clinical rationale:**  Cumulative vasopressor exposure quantifies haemodynamic
support burden.  Doses are normalised to noradrenaline equivalents using standard
clinical equivalence tables (see `configs/features/default.yaml`).

---

#### `urine_output_4h` — Urine Output (4-hour window)

| Attribute | Value |
|-----------|-------|
| Display name | Urine Output (4-hour window) |
| Source table | outputevents |
| Item IDs | 226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558 |
| Unit | mL |
| Aggregation | `sum` |
| Valid range | [0, 2 000] mL |
| Clip range | [0, 1 500] mL |
| Missing strategy | `zero` |
| Missingness flag | **Yes** |

**Clinical rationale:**  Oliguria (< 0.5 mL/kg/h) is a renal SOFA criterion;
urine output is one of the most actionable signals for fluid management.

---

### 4.6 Demographics

Demographics are static per-episode values — they do not change across steps.

---

#### `age_years` — Patient Age

| Attribute | Value |
|-----------|-------|
| Source table | patients |
| Unit | years |
| Aggregation | `last` (static) |
| Valid range | [18, 120] years |
| Clip range | [18, 110] years |
| Missing strategy | `median_train` |
| Missingness flag | No |

**Clinical rationale:**  Age is an independent predictor of sepsis mortality
and affects treatment response; inclusion allows the policy to learn
age-stratified treatment patterns.

---

#### `weight_kg` — Admission Body Weight

| Attribute | Value |
|-----------|-------|
| Source table | chartevents |
| Item IDs | 226512, 224639 |
| Unit | kg |
| Aggregation | `last` |
| Valid range | [20, 400] kg |
| Clip range | [20, 300] kg |
| Missing strategy | `median_train` |
| Missingness flag | No |

**Clinical rationale:**  Weight is required for weight-normalised vasopressor
dose calculations (µg/kg/min) and fluid-per-kg targets.

---

### 4.7 Derived Features

Derived features are computed inside the pipeline from base features already
extracted in the same step.  They have no direct MIMIC-IV source table.

---

#### `pf_ratio` — PaO2/FiO2 Ratio

| Attribute | Value |
|-----------|-------|
| Computed from | `pao2` / `fio2_vent` |
| Unit | mmHg |
| Aggregation | `min` (worst-case across step) |
| Valid range | [10, 700] mmHg |
| Clip range | [20, 600] mmHg |
| Missing strategy | `median_train` → normal value 400 mmHg |
| Missingness flag | **Yes** |

**Clinical rationale:**  The P/F ratio is the SOFA respiratory score component:
P/F < 300 → mild ARDS, < 200 → moderate, < 100 → severe.  Returns `None`
when either `pao2` or `fio2_vent` is missing this step.

---

#### `shock_index` — Shock Index

| Attribute | Value |
|-----------|-------|
| Computed from | `heart_rate` / `sbp` |
| Unit | ratio |
| Aggregation | `last` |
| Valid range | [0, 10] |
| Clip range | [0, 5] |
| Missing strategy | `forward_fill` → normal value 0.6 |
| Missingness flag | No |

**Clinical rationale:**  Shock index > 1.0 is a simple bedside marker of
haemodynamic compromise; captures the combined effect of tachycardia and
hypotension in a single scalar.

---

#### `hours_since_onset` — Hours Since Sepsis Onset

| Attribute | Value |
|-----------|-------|
| Computed from | Episode step metadata |
| Unit | hours |
| Aggregation | `last` (deterministic) |
| Valid range | [−24, 48] hours |
| Clip range | [−24, 48] hours |
| Missing strategy | `normal_value` = 0.0 (onset time) |
| Missingness flag | No |

**Clinical rationale:**  Temporal position within the episode encodes the
clinical phase (pre-onset stabilisation vs. acute management vs. recovery)
and allows the policy to learn time-dependent treatment patterns.

---

## 5. Missingness Summary

Features for which a `{feature_id}_missing` binary flag is emitted alongside
the imputed value:

| Feature ID | Clinical reason for flag |
|------------|--------------------------|
| map | Arterial line placement varies; gap = potential unmeasured hypotension |
| spo2 | Probe detachment / limb perfusion gap; missingness is informative |
| gcs_total | Sedated / intubated patients may not have scored GCS |
| fio2_vent | Not on mechanical ventilation |
| peep | Not on mechanical ventilation |
| pao2, paco2, arterial_ph | ABG is ordered on clinical indication — sparse by design |
| bicarbonate, base_excess | Sparse ABG ordering schedule |
| lactate | Ordered on clinical suspicion; absence itself may be informative |
| creatinine, bun | Lab drawn on daily schedule; within-step gap is common |
| bilirubin_total, albumin | Less frequently ordered; gap signals no active hepatic concern |
| sodium, potassium | Daily or more; short gaps can occur |
| glucose | Variable measurement frequency |
| inr, ptt | Ordered on clinical suspicion; coagulation panel not routine in all patients |
| wbc, haemoglobin | Daily CBC; gap in step window is common |
| platelets | Daily CBC |
| pf_ratio | Derived — missing when either parent is missing |
| urine_output_4h | Urinary catheter may not be present |

---

## 6. Leakage Controls

All preprocessing decisions that could introduce data leakage are documented here:

1. **Train-only medians** — `median_train` fallback values are computed on the
   training split only and written to `data/processed/features/train_medians.json`
   by the Phase 4 Plan 02 normalisation step.  They are never re-fit on validation
   or test data.

2. **Clip and normalisation bounds** — all scaler fit is deferred to Phase 4 Plan 02
   and applied train-split only.

3. **Cumulative aggregation boundary** — cumulative features (`cum_iv_fluid_ml`,
   `cum_vasopressor_dose_nor_equiv`) aggregate from episode start, not from the
   MIMIC-IV admission start, ensuring no future information leaks back in time.

4. **Derived features** — `pf_ratio` and `shock_index` are computed from features
   already extracted in the *same* step; they carry no cross-step lookahead.

---

## 7. State Vector Layout

The state vector column order follows clinical convention and matches the
registry insertion order in `dictionary.py`:

```
stay_id | step_index
-- Vitals (9) --
heart_rate | map | sbp | dbp | resp_rate | temperature | spo2 | gcs_total | fio2_vent | peep
-- Blood Gas (6) --
pao2 | paco2 | arterial_ph | bicarbonate | base_excess | lactate
-- Chemistry (9) --
creatinine | bun | bilirubin_total | albumin | sodium | potassium | glucose | inr | ptt
-- Haematology (4) --
wbc | haemoglobin | haematocrit | platelets
-- Treatments (3) --
cum_iv_fluid_ml | cum_vasopressor_dose_nor_equiv | urine_output_4h
-- Demographics (2) --
age_years | weight_kg
-- Derived (3) --
pf_ratio | shock_index | hours_since_onset
-- Missingness flags (for flagged features) --
map_missing | spo2_missing | gcs_total_missing | fio2_vent_missing | peep_missing
pao2_missing | paco2_missing | arterial_ph_missing | bicarbonate_missing | base_excess_missing
lactate_missing | creatinine_missing | bun_missing | bilirubin_total_missing | albumin_missing
sodium_missing | potassium_missing | glucose_missing | inr_missing | ptt_missing
wbc_missing | haemoglobin_missing | platelets_missing
urine_output_4h_missing | pf_ratio_missing
```

Total base features: **36**  
Total with all missingness flags: **61 columns** (excluding stay_id, step_index).

---

## 8. References

- Singer M et al. (2016). *The Third International Consensus Definitions for
  Sepsis and Septic Shock (Sepsis-3).* JAMA 315(8):801–810.
- Vincent JL et al. (1996). *The SOFA (Sepsis-related Organ Failure Assessment)
  score.* Intensive Care Med 22(7):707–710.
- Komorowski M et al. (2018). *The Artificial Intelligence Clinician learns
  optimal treatment strategies for sepsis in intensive care.* Nature Medicine
  24:1716–1720.
- Johnson AEW et al. (2023). *MIMIC-IV, a freely accessible electronic health
  record dataset.* Scientific Data 10:1.
- Rhodes A et al. (2017). *Surviving Sepsis Campaign: International Guidelines
  for Management of Sepsis and Septic Shock: 2016.* Intensive Care Med 43:304–377.

---

*Executable contract: `src/mimic_sepsis_rl/mdp/features/dictionary.py`*  
*Config: `configs/features/default.yaml`*  
*Phase: 04-state-representation-pipeline / Plan 04-01*