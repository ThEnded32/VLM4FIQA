import subprocess
import sys

analysis = [
    "analysis_evr.py",
    "analysis_scface.py",
    "analysis_prompt_ablation.py",
    "analysis_internal_consistency.py",
    "analysis_mix_degradation.py",
]

reports = [
    "report_evr_curve.py",
    "report_evr_metric.py",
    "report_scface.py",
    "report_internal.py",
    "report_consistency.py",
    "report_synthetic_mix.py",
]

for s in  analysis + reports:
    print(f"\n=== {s} ===")
    subprocess.run([sys.executable, s], check=True)
