"""
MIMIC-IV Preprocessing — CLI Entry Point
=========================================
Kullanım:
    uv run python -m src.preprocess
"""

from src.preprocess.state import run

if __name__ == "__main__":
    run()
