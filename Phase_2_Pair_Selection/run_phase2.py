"""
Phase 2 Launcher.
Runs the pair selection pipeline located in scripts/run_phase2.py.
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_PATH = os.path.join(SCRIPT_DIR, "scripts")
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from run_phase2 import run_pipeline

if __name__ == "__main__":
    run_pipeline()
