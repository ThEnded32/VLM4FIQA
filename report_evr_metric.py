import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_DIR = Path("analysis_metadata") / "evr"
INPUT_FILE = INPUT_DIR / "metrics_summary.csv"

OUTPUT_ROOT = Path("paper_figures") / "evr_metrics_multimodel"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

METRICS = ["pauc_1", "pauc_5", "pauc_10", "pauc_20"]

TARGET_FMR = 1e-3
MODE = "recompute_threshold"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

def identify_family(model_key):
    key = str(model_key).lower()
    if "sota" in key or "faceqan" in key or "fiqa" in key:
        return "SOTA"
    return "VLM"

def clean_model_name(model_key):
    key = str(model_key)
    suffix = ""
    if "classification" in key.lower() and "real" in key.lower():
        suffix = " (Classif.)"
    elif "utility" in key.lower():
        suffix = " (Utility)"
    elif "reliability" in key.lower():
        suffix = " (Reliability)"
    
    for prefix in ["SOTA__", "VLM__", "real_simple__", "real_classification__", "single_degradation__"]:
        key = key.replace(prefix, "")
    key = key.replace("_utility", "").replace("_reliability", "").replace("_classification", "")
    
    if "ediffiqa" in key: return "eDifFIQA (Large)"
    if "faceqan" in key: return "FaceQAN"
    if "sddfiqa" in key: return "SDD-FIQA"
    if "vitfiqa" in key: return "ViT-FIQA (T)"
    
    name = key.replace("qwen2p5", "Qwen2.5").replace("qwen2", "Qwen2")
    name = name.replace("gemma3", "Gemma-3").replace("phi4", "Phi-4").replace("idefics", "Idefics")
    name = name.replace("_32b", "-32B").replace("_72b", "-72B").replace("_7b", "-7B")
    name = name.replace("_", "-")
    
    if name.lower().startswith("qwen"): name = "Qwen" + name[4:]
    if name.lower().startswith("phi"): name = "Phi" + name[3:]
    
    return name + suffix

def generate_report_for_dataframe(df, output_dir, title_suffix=""):
    os.makedirs(output_dir, exist_ok=True)
    if "pauc_20" in df.columns:
        df = df.sort_values("pauc_20", ascending=True)
    
    display_cols = ["Family", "Model"]
    for m in METRICS:
        if m in df.columns:
            display_cols.append(m)
        else:
            print(f"  [Warning] Metric {m} not found in data.")
            
    out_csv = output_dir / f"metrics_table_{title_suffix}.csv"
    df[display_cols].to_csv(out_csv, index=False)

    plt.figure(figsize=(10, max(6, len(df)*0.3)))
    palette = {"SOTA": "tab:red", "VLM": "tab:blue"}
    
    plot_metric = "pauc_20" if "pauc_20" in df.columns else METRICS[-1]
    
    sns.barplot(data=df, x=plot_metric, y="Model", hue="Family", dodge=False, palette=palette)
    plt.title(f"Accumulated Biometric Error ({plot_metric}) - {title_suffix}")
    plt.xlabel(f"{plot_metric} (Lower is Better)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / f"plot_{plot_metric}_{title_suffix}.png", dpi=300)
    plt.close()

def main():
    if not INPUT_FILE.exists():
        print(f"[Error] Summary file not found: {INPUT_FILE}")
        return

    print("Loading metrics summary...")
    df_all = pd.read_csv(INPUT_FILE)
    df_all = df_all[(df_all["target_fmr"] == TARGET_FMR) & (df_all["mode"] == MODE)].copy()

    if "pauc_20" not in df_all.columns:
        print("[Warning] 'pauc_20' not found. Please run updated 'evr_analyze_runall.py' first.")
        
    variants = df_all["variant"].unique()
    print(f"Found {len(variants)} Variants.")

    for variant in variants:
        out_sub = OUTPUT_ROOT / variant
        df_sub = df_all[df_all["variant"] == variant].copy()
        df_sub["Family"] = df_sub["model_key"].apply(identify_family)
        df_sub["Model"] = df_sub["model_key"].apply(clean_model_name)
        generate_report_for_dataframe(df_sub, out_sub, title_suffix=variant)

    if len(variants) > 1:
        print("\nProcessing Averaged Report...")
        
        numeric_cols = [c for c in df_all.columns if "pauc" in c or "auc" in c]
        df_avg = df_all.groupby("model_key")[numeric_cols].mean().reset_index()
        
        df_avg["Family"] = df_avg["model_key"].apply(identify_family)
        df_avg["Model"] = df_avg["model_key"].apply(clean_model_name)
        
        if "pauc_20" in df_avg.columns:
            df_avg = df_avg.sort_values("pauc_20")
        
        out_sub = OUTPUT_ROOT / "AVERAGE_ALL_MODELS"
        print(f"Location: {out_sub.resolve()}")
        generate_report_for_dataframe(df_avg, out_sub, title_suffix="AVERAGE")

    print("Done.")

if __name__ == "__main__":
    main()