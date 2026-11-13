#!/usr/bin/env python3
"""
Top-level runner for Phase 6: Performance Metrics & Benchmark Comparison.
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, "scripts")
sys.path.insert(0, scripts_dir)

from run_phase6 import run_phase6_pipeline

if __name__ == "__main__":
    run_phase6_pipeline()

