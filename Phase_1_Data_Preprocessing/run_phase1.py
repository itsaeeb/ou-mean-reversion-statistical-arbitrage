"""
Phase 1 Convenience Launcher.
Runs the pipeline located in scripts/run_phase1.py.
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_PATH = os.path.join(SCRIPT_DIR, "scripts")
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from run_phase1 import run_pipeline

if __name__ == "__main__":
    force = "--force-download" in sys.argv
    run_pipeline(force_download=force)
