"""
======================================================================
 Public Transport Demand Predictor — Pipeline Runner
======================================================================
 Runs the entire pipeline in one command:
   1. Generate synthetic dataset
   2. Train and evaluate models (which internally calls preprocess)
 Usage:  python run_pipeline.py
         OR
         & "C:\Users\OMKAR\AppData\Local\Python\bin\python.exe" run_pipeline.py
======================================================================
"""

import subprocess, sys, os

# Use the same Python executable that is running this script
PYTHON = sys.executable
SRC    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

steps = [
    ("STEP 1 — Generating Synthetic Dataset", os.path.join(SRC, "generate_dataset.py")),
    ("STEP 2 — Training ML Models",           os.path.join(SRC, "train_model.py")),
]

for label, script in steps:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([PYTHON, script], check=True)

print("\n" + "="*60)
print("  ✅  PIPELINE COMPLETE")
print()
print("  Launch the dashboard with:")
print(f"  {PYTHON} -m streamlit run app.py")
print()
print("  Or simply:  streamlit run app.py")
print("="*60 + "\n")
