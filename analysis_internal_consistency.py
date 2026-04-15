import pandas as pd
import os
import loader_utils  

OUTPUT_DIR = "analysis_metadata/internal_consistency"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTRIBUTES = ["Sharpness", "Resolution", "Compression", "Lighting"]

def get_classification_files():
    found = []
    target_folders = ["real_classification", "single_degradation_classification"]
    
    for folder in target_folders:
        base_path = os.path.join(loader_utils.DATASET_ROOT, folder)
        if not os.path.exists(base_path): continue
        
        for model in os.listdir(base_path):
            model_path = os.path.join(base_path, model)
            if not os.path.isdir(model_path): continue
            
            for f in os.listdir(model_path):
                if f.endswith(".parquet"):
                    dataset = f.replace("_mtcnn_aligned", "").replace(".parquet", "")
                    if "original" in dataset: dataset = dataset.replace("_original", "")
                    
                    found.append({
                        "Model": model,
                        "Dataset": dataset,
                        "Path": os.path.join(model_path, f),
                        "Source_Folder": folder
                    })
    return found

def analyze_model_consistency(file_info):
    path = file_info["Path"]
    model = file_info["Model"]
    dataset = file_info["Dataset"]
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[Err] Failed to read {path}: {e}")
        return []

    score_col = None
    for cand in ['Quality Score', 'quality_score', 'score', 'Quality_Score']:
        if cand in df.columns:
            score_col = cand
            break
            
    if not score_col:
        return []

    df['numeric_score'] = pd.to_numeric(df[score_col], errors='coerce')
    df = df.dropna(subset=['numeric_score'])
    
    results = []

    for attr in ATTRIBUTES:
        if attr not in df.columns:
            continue
            
        mapping = loader_utils.TEXT_TO_NUMERIC.get(attr, {})
        
        sub_df = df[df[attr].isin(mapping.keys())].copy()
        
        if len(sub_df) < 5:
            continue

        groups = sub_df.groupby(attr)['numeric_score']
        
        stats = groups.describe(percentiles=[.25, .5, .75])
        
        for label_val, row in stats.iterrows():
            ord_val = mapping.get(label_val, -1)
            
            results.append({
                "Model": model,
                "Dataset": dataset,
                "Attribute": attr,
                "Label_Text": label_val,
                "Label_Ordinal": ord_val,
                "Count": int(row['count']),
                "Mean_Score": row['mean'],
                "Std_Dev": row['std'],
                "Min": row['min'],
                "Q1_25%": row['25%'],
                "Median_50%": row['50%'],
                "Q3_75%": row['75%'],
                "Max": row['max']
            })
            
    return results

def main():
    print("--- Starting Internal Consistency Analysis (Distributions Only) ---")
    files = get_classification_files()
    
    files_by_model = {}
    for item in files:
        m = item["Model"]
        if m not in files_by_model: files_by_model[m] = []
        files_by_model[m].append(item)

    for model_name, file_list in files_by_model.items():
        print(f"Processing {model_name}...")
        all_model_results = []
        
        for file_info in file_list:
            rows = analyze_model_consistency(file_info)
            all_model_results.extend(rows)
            
        if all_model_results:
            out_df = pd.DataFrame(all_model_results)
            out_df = out_df.sort_values(
                by=['Dataset', 'Attribute', 'Label_Ordinal'], 
                ascending=[True, True, True] 
            )
            
            save_path = os.path.join(OUTPUT_DIR, f"{model_name}_internal.csv")
            out_df.to_csv(save_path, index=False)
            print(f"  -> Saved {save_path}")

    print("\nInternal Consistency Analysis Complete.")

if __name__ == "__main__":
    main()