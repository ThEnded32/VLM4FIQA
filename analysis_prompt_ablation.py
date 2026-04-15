import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import loader_utils

OUTPUT_DIR = "analysis_metadata/prompt_consistency"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIANTS = ["reliability", "utility"]

def calculate_metrics(df, col_a, col_b):

    try:
        pearson_r, _ = pearsonr(df[col_a], df[col_b])
    except:
        pearson_r = np.nan

    mae = np.mean(np.abs(df[col_a] - df[col_b]))
    bias = np.mean(df[col_b] - df[col_a])
    return pearson_r, mae, bias


def analyze_format_consistency():
    print("--- Part 1: Analyzing Format Consistency (Simple vs Classification) ---")
    results = []
    
    simple_root = os.path.join(loader_utils.DATASET_ROOT, "real_simple")
    class_root = os.path.join(loader_utils.DATASET_ROOT, "real_classification")
    
    if not os.path.exists(simple_root) or not os.path.exists(class_root):
        print("[Error] real_simple or real_classification folder missing.")
        return

    models = os.listdir(simple_root)
    
    for model_name in models:
        if any(v in model_name for v in VARIANTS):
            continue
            
        class_model_dir = os.path.join(class_root, model_name)
        if not os.path.exists(class_model_dir):
            continue

        simple_model_dir = os.path.join(simple_root, model_name)
        datasets = [f for f in os.listdir(simple_model_dir) if f.endswith(".parquet")]
        
        for ds_file in datasets:
            ds_name = ds_file.replace(".parquet", "").replace("_mtcnn_aligned", "")
            
            df_simple = loader_utils.load_vlm_predictions(model_name, ds_name, "real_simple")
            if df_simple is None: continue
            df_simple = df_simple.rename(columns={'predicted_score': 'score_simple'})
            
            df_class = loader_utils.load_vlm_predictions(model_name, ds_name, "real_classification")
            if df_class is None: continue
            df_class = df_class.rename(columns={'predicted_score': 'score_class'})
            
            merged = pd.merge(df_simple, df_class, on='filename', how='inner')
            if len(merged) < 50: continue
            
            pearson_r, mae, bias = calculate_metrics(merged, 'score_simple', 'score_class')
            
            results.append({
                "Model": model_name,
                "Dataset": ds_name,
                "Comparison": "Simple_vs_Class",
                "Pearson_r": pearson_r,
                "MAE": mae,
                "Bias (Class - Simple)": bias,
                "N_Samples": len(merged)
            })

    if results:
        df_out = pd.DataFrame(results)
        save_path = os.path.join(OUTPUT_DIR, "results_format_consistency.csv")
        df_out.to_csv(save_path, index=False)
        print(f"  -> Saved Format Consistency: {save_path}")
    else:
        print("  No matching models found for Format Consistency.")


def analyze_prompt_ablation():
    print("\n--- Part 2: Analyzing Prompt Ablation (Quality vs Variants) ---")
    results = []
    
    simple_root = os.path.join(loader_utils.DATASET_ROOT, "real_simple")
    if not os.path.exists(simple_root): return

    all_models = os.listdir(simple_root)
    groups = {} 
    
    for m in all_models:
        is_variant = False
        for v in VARIANTS:
            if m.endswith(f"_{v}"):
                base = m.replace(f"_{v}", "")
                if base not in groups: groups[base] = []
                groups[base].append(m)
                is_variant = True
                break
        if not is_variant:
            if m not in groups: groups[m] = []

    valid_groups = {k: v for k, v in groups.items() if v}
    
    if not valid_groups:
        print("  No ablation variants found.")
        return

    for base_model, variant_list in valid_groups.items():
        print(f"  Processing group: {base_model}")
        
        base_dir = os.path.join(simple_root, base_model)
        if not os.path.exists(base_dir): continue
        
        datasets = [f for f in os.listdir(base_dir) if f.endswith(".parquet")]
        
        for ds_file in datasets:
            ds_name = ds_file.replace(".parquet", "").replace("_mtcnn_aligned", "")
            
            df_base = loader_utils.load_vlm_predictions(base_model, ds_name, "real_simple")
            if df_base is None: continue
            df_base = df_base.rename(columns={'predicted_score': 'score_base'})
            
            for var_model in variant_list:
                v_type = "Utility" if "utility" in var_model else "Reliability"
                
                df_var = loader_utils.load_vlm_predictions(var_model, ds_name, "real_simple")
                if df_var is None: continue
                df_var = df_var.rename(columns={'predicted_score': 'score_variant'})
                
                merged = pd.merge(df_base, df_var, on='filename', how='inner')
                if len(merged) < 50: continue
                
                pearson_r, mae, bias = calculate_metrics(merged, 'score_base', 'score_variant')
                
                results.append({
                    "Base_Model": base_model,
                    "Variant_Model": var_model,
                    "Variant_Type": v_type,
                    "Dataset": ds_name,
                    "Pearson_r": pearson_r,
                    "MAE": mae,
                    "Bias (Variant - Base)": bias,
                    "N_Samples": len(merged)
                })

    if results:
        df_out = pd.DataFrame(results)
        df_out = df_out.sort_values(by=['Base_Model', 'Variant_Type', 'Dataset'])
        
        save_path = os.path.join(OUTPUT_DIR, "results_prompt_ablation.csv")
        df_out.to_csv(save_path, index=False)
        print(f"  -> Saved Prompt Ablation: {save_path}")
    else:
        print("  No valid ablation comparisons made.")

def main():
    analyze_format_consistency()
    analyze_prompt_ablation()
    print("\nConsistency Analysis Complete.")

if __name__ == "__main__":
    main()