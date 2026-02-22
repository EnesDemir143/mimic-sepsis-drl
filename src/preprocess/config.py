"""
MIMIC-IV Sepsis DRL — Configuration
=====================================
Dosya yolları, MIMIC item ID'leri, Elixhauser ICD mapping ve
48-feature state vektörü tanımları.
"""
from __future__ import annotations
from pathlib import Path

# ─────────────────────────────────────────────
# Dizin Yolları
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "physionet.org" / "files" / "mimiciv" / "3.1"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Girdi
HOURLY_BINNED_PATH = PROCESSED_DIR / "mimic_hourly_binned.parquet"

# Raw MIMIC dosyaları
LABEVENTS_PATH = RAW_DIR / "hosp" / "labevents.csv.gz"
CHARTEVENTS_PATH = RAW_DIR / "icu" / "chartevents.csv.gz"
ICUSTAYS_PATH = RAW_DIR / "icu" / "icustays.csv.gz"
DIAGNOSES_ICD_PATH = RAW_DIR / "hosp" / "diagnoses_icd.csv.gz"
ADMISSIONS_PATH = RAW_DIR / "hosp" / "admissions.csv.gz"
OMR_PATH = RAW_DIR / "hosp" / "omr.csv.gz"

# Çıktı
STATE_PARQUET_PATH = PROCESSED_DIR / "state.parquet"

# ─────────────────────────────────────────────
# Lab Item ID'leri — labevents.csv.gz'den çekilecek
# ─────────────────────────────────────────────
EXTRA_LAB_ITEMS: dict[str, list[int]] = {
    "sgot":             [50878],          # AST
    "sgpt":             [50861],          # ALT
    "albumin":          [50862],          # Albumin (Blood)
    "calcium_total":    [50893],          # Calcium, Total (Blood)
    "ionized_calcium":  [50808],          # Free Calcium (Blood Gas)
    "magnesium":        [50960],          # Magnesium (Blood)
    "ptt":              [51275],          # Partial Thromboplastin Time
    "pt":               [51274],          # Prothrombin Time
    "base_excess":      [50802],          # Base Excess (Blood Gas)
    "co2_total":        [50804],          # Calculated Total CO2
}

# ─────────────────────────────────────────────
# Weight (Ağırlık) — chartevents item ID'leri
# ─────────────────────────────────────────────
WEIGHT_CHART_ITEMIDS: list[int] = [
    226512,   # Admission Weight (Kg)
    224639,   # Daily Weight
]

# ─────────────────────────────────────────────
# Vazopressör Norepinefrin Eşdeğer Oranları
# ─────────────────────────────────────────────
VASOPRESSOR_CONVERSION: dict[str, float] = {
    "norepinephrine_dose": 1.0,
    "epinephrine_dose":    0.1,
    "phenylephrine_dose":  0.1,
    "vasopressin_dose":    0.4,
    "dopamine_dose":       0.01,
    "dobutamine_dose":     0.0,    # İnotrop — vazo eşdeğerine dahil değil
}

# ─────────────────────────────────────────────
# Elixhauser ICD-9 Komorbidite Grupları
# ─────────────────────────────────────────────
# Quan et al. (2005) ve Gasparini (2018) tabanlı basitleştirilmiş mapping.
# Her kategori: (başlangıç_kodu, bitiş_kodu) tuple listesi.
# ICD-9 kodları string olarak eşleştirilir (prefix match).
ELIXHAUSER_ICD9: dict[str, list[str]] = {
    "chf":              ["39891", "4254", "4255", "4257", "4258", "4259", "428"],
    "arrhythmia":       ["4260", "42613", "4267", "4269", "4270", "4271", "4272",
                         "4273", "4274", "4276", "4278", "4279", "7850", "V450",
                         "V533"],
    "valve":            ["0932", "394", "395", "396", "397", "424", "7463",
                         "7464", "7465", "7466", "V422", "V433"],
    "pulmonary_circ":   ["4150", "4## 151", "416", "4170", "4178", "4179"],
    "pvd":              ["4400", "4401", "4402", "4403", "4404", "4405", "4406",
                         "4407", "4408", "4409", "4412", "4414", "4417", "4419",
                         "4431", "4432", "4438", "4439", "4471", "5571", "5579",
                         "V434"],
    "htn_uncomp":       ["401"],
    "htn_comp":         ["402", "403", "404", "405"],
    "paralysis":        ["3341", "342", "343", "3440", "3441", "3442", "3443",
                         "3444", "3445", "3446", "3449"],
    "neuro_other":      ["3319", "3320", "3321", "3334", "3335", "334", "335",
                         "3362", "340", "341", "345", "3481", "3483", "7803",
                         "7843"],
    "chronic_pulm":     ["4168", "4169", "490", "491", "492", "493", "494",
                         "495", "496", "500", "501", "502", "503", "504", "505",
                         "5064", "5081", "5088"],
    "dm_uncomp":        ["2500", "2501", "2502", "2503"],
    "dm_comp":          ["2504", "2505", "2506", "2507", "2508", "2509"],
    "hypothyroid":      ["2409", "243", "244", "2461", "2468"],
    "renal":            ["585", "586", "V420", "V451", "V56"],
    "liver":            ["07022", "07023", "07032", "07033", "07044", "07054",
                         "0706", "0709", "4560", "4561", "4562", "570", "571",
                         "5722", "5723", "5724", "5728", "5733", "5734", "5738",
                         "5739", "V427"],
    "pud":              ["5317", "5319", "5327", "5329", "5337", "5339",
                         "5347", "5349"],
    "hiv":              ["042", "043", "044"],
    "lymphoma":         ["200", "201", "202", "2030", "2386"],
    "mets":             ["196", "197", "198", "199"],
    "solid_tumor":      ["140", "141", "142", "143", "144", "145", "146",
                         "147", "148", "149", "150", "151", "152", "153",
                         "154", "155", "156", "157", "158", "159", "160",
                         "161", "162", "163", "164", "165", "170", "171",
                         "172", "174", "175", "176", "179", "180", "181",
                         "182", "183", "184", "185", "186", "187", "188",
                         "189", "190", "191", "192", "193", "194", "195"],
    "rheumatoid":       ["446", "7010", "7100", "7101", "7102", "7103",
                         "7104", "7108", "7109", "7112", "714", "7193",
                         "720", "725", "7285", "72889", "72930"],
    "coagulopathy":     ["286", "2871", "2873", "2874", "2875"],
    "obesity":          ["2780"],
    "weight_loss":      ["260", "261", "262", "263", "7832", "7994"],
    "fluid_electrolyte": ["2536", "276"],
    "blood_loss":       ["2800"],
    "deficiency_anemia": ["2801", "2808", "2809", "281"],
    "alcohol":          ["2652", "2911", "2912", "2913", "2915", "2918",
                         "2919", "3030", "3039", "3050", "3575", "4255",
                         "5353", "5710", "5711", "5712", "5713", "980",
                         "V113"],
    "drug":             ["292", "304", "3052", "3053", "3054", "3055",
                         "3056", "3057", "3058", "3059", "V6542"],
    "psychoses":        ["2938", "2950", "2951", "2952", "2953", "2954",
                         "2955", "2956", "2957", "2958", "2959", "297"],
    "depression":       ["2962", "2963", "2965", "3004", "309", "311"],
}

# ICD-10 basitleştirilmiş prefix mapping (en yaygın kodlar)
ELIXHAUSER_ICD10: dict[str, list[str]] = {
    "chf":              ["I099", "I110", "I130", "I132", "I255", "I420",
                         "I425", "I426", "I427", "I428", "I429", "I43", "I50",
                         "P290"],
    "arrhythmia":       ["I441", "I442", "I443", "I456", "I459", "I47",
                         "I48", "I49", "R000", "R001", "R008", "T821",
                         "Z450", "Z950"],
    "valve":            ["A520", "I05", "I06", "I07", "I08", "I091",
                         "I098", "I34", "I35", "I36", "I37", "I38", "I39",
                         "Q230", "Q231", "Q232", "Q233", "Z952", "Z953",
                         "Z954"],
    "pulmonary_circ":   ["I26", "I27", "I280", "I288", "I289"],
    "pvd":              ["I70", "I71", "I731", "I738", "I739", "I771",
                         "I790", "I792", "K551", "K558", "K559", "Z958",
                         "Z959"],
    "htn_uncomp":       ["I10"],
    "htn_comp":         ["I11", "I12", "I13", "I15"],
    "paralysis":        ["G041", "G114", "G801", "G802", "G81", "G82",
                         "G830", "G831", "G832", "G833", "G834", "G839"],
    "neuro_other":      ["G10", "G11", "G12", "G13", "G20", "G21", "G22",
                         "G254", "G255", "G312", "G318", "G319", "G32",
                         "G35", "G36", "G37", "G40", "G41", "G931",
                         "G934", "R470", "R56"],
    "chronic_pulm":     ["I278", "I279", "J40", "J41", "J42", "J43",
                         "J44", "J45", "J46", "J47", "J60", "J61", "J62",
                         "J63", "J64", "J65", "J66", "J67", "J684",
                         "J701", "J703"],
    "dm_uncomp":        ["E100", "E101", "E109", "E110", "E111", "E119",
                         "E120", "E121", "E129", "E130", "E131", "E139",
                         "E140", "E141", "E149"],
    "dm_comp":          ["E102", "E103", "E104", "E105", "E106", "E107",
                         "E108", "E112", "E113", "E114", "E115", "E116",
                         "E117", "E118", "E122", "E123", "E124", "E125",
                         "E126", "E127", "E128", "E132", "E133", "E134",
                         "E135", "E136", "E137", "E138", "E142", "E143",
                         "E144", "E145", "E146", "E147", "E148"],
    "hypothyroid":      ["E00", "E01", "E02", "E03", "E890"],
    "renal":            ["I120", "I131", "N18", "N19", "N250", "Z490",
                         "Z491", "Z492", "Z940", "Z992"],
    "liver":            ["B18", "I85", "I864", "I982", "K70", "K711",
                         "K713", "K714", "K715", "K717", "K72", "K73",
                         "K74", "K760", "K762", "K763", "K764", "K765",
                         "K766", "K767", "K768", "K769", "Z944"],
    "pud":              ["K257", "K259", "K267", "K269", "K277", "K279",
                         "K287", "K289"],
    "hiv":              ["B20", "B21", "B22", "B24"],
    "lymphoma":         ["C81", "C82", "C83", "C84", "C85", "C88", "C96",
                         "C900", "C902"],
    "mets":             ["C77", "C78", "C79", "C80"],
    "solid_tumor":      ["C00", "C01", "C02", "C03", "C04", "C05", "C06",
                         "C07", "C08", "C09", "C10", "C11", "C12", "C13",
                         "C14", "C15", "C16", "C17", "C18", "C19", "C20",
                         "C21", "C22", "C23", "C24", "C25", "C26", "C30",
                         "C31", "C32", "C33", "C34", "C37", "C38", "C39",
                         "C40", "C41", "C43", "C45", "C46", "C47", "C48",
                         "C49", "C50", "C51", "C52", "C53", "C54", "C55",
                         "C56", "C57", "C58", "C60", "C61", "C62", "C63",
                         "C64", "C65", "C66", "C67", "C68", "C69", "C70",
                         "C71", "C72", "C73", "C74", "C75", "C76", "C97"],
    "rheumatoid":       ["L940", "L941", "L943", "M05", "M06", "M08",
                         "M120", "M123", "M30", "M310", "M311", "M312",
                         "M313", "M32", "M33", "M34", "M35", "M45",
                         "M461", "M468", "M469"],
    "coagulopathy":     ["D65", "D66", "D67", "D68", "D691", "D693",
                         "D694", "D695", "D696"],
    "obesity":          ["E66"],
    "weight_loss":      ["E40", "E41", "E42", "E43", "E44", "E45", "E46",
                         "R634", "R64"],
    "fluid_electrolyte": ["E222", "E86", "E87"],
    "blood_loss":       ["D500"],
    "deficiency_anemia": ["D508", "D509", "D51", "D52", "D53"],
    "alcohol":          ["F10", "E52", "G621", "I426", "K292", "K700",
                         "K703", "K709", "T51", "Z502", "Z714", "Z721"],
    "drug":             ["F11", "F12", "F13", "F14", "F15", "F16", "F17",
                         "F18", "F19", "Z715", "Z722"],
    "psychoses":        ["F20", "F22", "F23", "F24", "F25", "F28", "F29",
                         "F302", "F312", "F315"],
    "depression":       ["F204", "F313", "F314", "F315", "F32", "F33",
                         "F341", "F412", "F432"],
}

# ─────────────────────────────────────────────
# Final 48 Feature Sırası (state.parquet)
# ─────────────────────────────────────────────
STATE_FEATURES: list[str] = [
    # Demografi (5)
    "age",
    "gender",
    "weight_kg",
    "elixhauser_score",
    "icu_readmission",
    # Skorlar / Vitals (11)
    "sofa_score",
    "sirs_score",
    "gcs_total",
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "shock_index",
    "spo2",
    "temp_c",
    "resp_rate",
    # Labs (28)
    "potassium",
    "sodium",
    "chloride",
    "glucose",
    "creatinine",
    "magnesium",
    "calcium_total",
    "ionized_calcium",
    "sgot",
    "sgpt",
    "bilirubin_total",
    "albumin",
    "hemoglobin",
    "wbc",
    "platelet",
    "ptt",
    "pt",
    "inr",
    "bun",
    "ph",
    "pao2",
    "paco2",
    "base_excess",
    "bicarbonate",
    "co2_total",
    "lactate",
    "hco3",
    "pf_ratio",
    # Solunum ve Sıvı (4)
    "mechanical_ventilation",
    "fio2",
    "urine_output",
    "cumulative_fluid_balance",
]

# Meta sütunlar (parquet'ta tutulur ama state vektörüne değil)
META_COLUMNS: list[str] = ["stay_id", "hour_bin"]
