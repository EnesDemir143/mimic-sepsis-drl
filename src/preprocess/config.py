"""
MIMIC-IV Sepsis DRL — Feature & Path Configuration
===================================================
Tüm item ID'leri ve dosya yolları burada tanımlıdır.
"""

from __future__ import annotations

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # mimic-sepsis-drl/
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "physionet.org" / "files" / "mimiciv" / "3.1"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PARQUET = OUT_DIR / "mimic_hourly_binned.parquet"

# Kaynak tablolar (sıkıştırılmış CSV)
ICUSTAYS_CSV  = RAW_DIR / "icu"  / "icustays.csv.gz"
CHARTEVENTS_CSV = RAW_DIR / "icu"  / "chartevents.csv.gz"
LABEVENTS_CSV = RAW_DIR / "hosp" / "labevents.csv.gz"
OUTPUTEVENTS_CSV = RAW_DIR / "icu" / "outputevents.csv.gz"
INPUTEVENTS_CSV = RAW_DIR / "icu" / "inputevents.csv.gz"
PATIENTS_CSV = RAW_DIR / "hosp" / "patients.csv.gz"
ADMISSIONS_CSV = RAW_DIR / "hosp" / "admissions.csv.gz"

# ─── Vital Signs (chartevents) ────────────────────────────────────
#   key   = çıktıdaki sütun adı
#   value = o sütuna ait itemid listesi (birden fazla kaynak birleştirilir)
VITALS: dict[str, list[int]] = {
    "heart_rate": [220045],
    "sbp":        [220050, 220179],   # Arterial + Non-Invasive
    "dbp":        [220051, 220180],
    "mbp":        [220052, 220181],
    "resp_rate":  [220210],
    "spo2":       [220277],
    "temp_c":     [223762],
    "fio2":       [223835],
}

# ─── Lab Results (labevents) ──────────────────────────────────────
LABS: dict[str, list[int]] = {
    "lactate":         [50813],
    "creatinine":      [50912],
    "bilirubin_total": [50885],
    "platelet":        [51265],
    "wbc":             [51301],
    "bun":             [51006],
    "glucose":         [50931],
    "sodium":          [50983],
    "potassium":       [50971],
    "hemoglobin":      [51222],
    "hematocrit":      [51221],
    "bicarbonate":     [50882],
    "chloride":        [50902],
    "anion_gap":       [50868],
    "inr":             [51237],
    "pao2":            [50821],
    "paco2":           [50818],
    "ph":              [50820],
}

# Pivot-free strateji için tüm itemid → feature adı eşleme tablosu
ALL_VITALS_IDS: list[int] = [iid for ids in VITALS.values() for iid in ids]
ALL_LABS_IDS: list[int]   = [iid for ids in LABS.values()   for iid in ids]

# ─── Urine Output (outputevents) ──────────────────────────────────
URINE_OUTPUT: dict[str, list[int]] = {
    "urine_output": [226557, 226558, 226559, 226560, 226561, 226563, 226564, 226565, 226566, 226567]
}

# ─── Vasopressors (inputevents) ───────────────────────────────────
# Not: Bu itemid'ler carevue ve metavision için ayrı, hepsini ekle
VASOPRESSORS: dict[str, list[int]] = {
    "norepinephrine": [221906, 229617, 300050, 300051],  # mcg/kg/min veya mcg/min
    "epinephrine": [221289, 229618, 300054, 300055],
    "phenylephrine": [221749, 229616, 300052, 300053],
    "vasopressin": [222315, 300057, 300058],  # units/min
    "dopamine": [221662, 229619, 300056, 300064],
    "dobutamine": [221653, 300059, 300061],
}

# ─── IV Fluids / Crystalloids (inputevents) ───────────────────────
# 4 saatteki toplam sıvı miktarı için (bolus + maintenance)
CRYSTALLOIDS: dict[str, list[int]] = {
    "crystalloid_ml": [
        21007,   # Normal Saline
        220995,  # Lactated Ringers
        220862,  # Normal Saline (bag)
        220996,  # D5W
        220997,  # D5NS
        221001,  # NS
        221002,  # LR
        225158,  # NaCl 0.9%
        225828,  # LR
        225944,  # NS
        225797,  # Free Water
    ]
}

# ─── GCS (Glasgow Coma Scale) ─────────────────────────────────────
GCS: dict[str, list[int]] = {
    "gcs_eye": [220739],
    "gcs_motor": [223901], 
    "gcs_verbal": [223900],
}

ALL_URINE_IDS: list[int] = [iid for ids in URINE_OUTPUT.values() for iid in ids]
ALL_VASO_IDS: list[int] = [iid for ids in VASOPRESSORS.values() for iid in ids]
ALL_FLUID_IDS: list[int] = [iid for ids in CRYSTALLOIDS.values() for iid in ids]
ALL_GCS_IDS: list[int] = [iid for ids in GCS.values() for iid in ids]
