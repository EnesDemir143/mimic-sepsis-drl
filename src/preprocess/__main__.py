"""
MIMIC-IV Preprocessing — CLI Entry Point
=========================================
Kullanım:
    uv run python -m src.preprocess
"""

from src.preprocess.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
