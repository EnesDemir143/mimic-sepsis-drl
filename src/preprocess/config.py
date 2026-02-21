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
