"""Run the whole Task 2.1 pipeline in order.

Usage (from the Task 2.1 folder):
    python run_all.py              # everything
    python run_all.py --data-only  # only the dataset scripts
    python run_all.py --refresh    # also re-query OpenStreetMap for stadium locations
"""

from pathlib import Path
import subprocess
import sys
import time


BASE_DIR = Path(__file__).resolve().parent

data_scripts = sorted((BASE_DIR / "scripts").glob("[0-9][0-9]_*.py"))
model_scripts = sorted((BASE_DIR / "modelling").glob("[0-9][0-9]_*.py"))

to_run = data_scripts if "--data-only" in sys.argv else data_scripts + model_scripts

for script in to_run:
    arguments = [sys.executable, str(script)]
    if "--refresh" in sys.argv and script.name.startswith("04_add_travel"):
        arguments.append("--refresh")

    print("=" * 70)
    print("Running", script.relative_to(BASE_DIR))
    print("=" * 70)
    start = time.time()
    result = subprocess.run(arguments, cwd=script.parent)
    if result.returncode != 0:
        sys.exit(f"Stopped: {script.name} failed")
    print(f"({time.time() - start:.0f} s)\n")

print("All scripts finished")
