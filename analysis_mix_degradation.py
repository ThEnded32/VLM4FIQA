import os
import glob
import json
import pandas as pd
import numpy as np
import loader_utils

INPUT_DIR = os.path.join(loader_utils.DATASET_ROOT, "mix_degradation")
OUTPUT_DIR = "analysis_metadata/mix_degradation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GT_JSON_PATH = os.path.join(INPUT_DIR, "hard_mix_labels.json")

ARTIFACT_MAP = {
    "blur": 0,
    "noise": 1,
    "res": 2, "downsample": 2, "resolution": 2,
    "jpeg": 3, "compression": 3, "compress": 3,
    "light": 4, "dark": 4, "exposure": 4
}
ARTIFACT_NAMES = ["Blur", "Noise", "LowRes", "Compression", "Lighting"]

def parse_gt_labels(label_list):

    vec = [0] * 5
    hard_art = None
    
    for label in label_list:
        lbl_lower = label.lower()
        
        idx = -1
        for key, mapped_idx in ARTIFACT_MAP.items():
            if key in lbl_lower:
                idx = mapped_idx
                break
        
        if idx != -1:
            vec[idx] = 1
            if "(hard)" in lbl_lower:
                hard_art = ARTIFACT_NAMES[idx]
                
    return vec, hard_art

def load_gt_data():
    if not os.path.exists(GT_JSON_PATH):
        print(f"[Error] GT JSON not found at: {GT_JSON_PATH}")
        return {}
    
    print(f"Loading GT from: {GT_JSON_PATH}")
    with open(GT_JSON_PATH, 'r') as f:
        data = json.load(f)
        
    lookup = {}
    
    items = data if isinstance(data, list) else []
    
    for item in items:
        raw_key = item.get("path") or item.get("filename")
        if not raw_key: continue
        
        clean_key = raw_key.strip().lower()
        
        lvl = str(item.get("level", "L0"))
        if lvl == "0": lvl = "L0"
        if lvl == "2": lvl = "L2"
        
        labels = item.get("labels", [])
        vec, hard = parse_gt_labels(labels)
        
        lookup[clean_key] = {
            "Level": lvl,
            "GT_Vector": vec,
            "GT_Hard_Artifact": hard
        }
    
    print(f"Loaded {len(lookup)} GT items.")
    return lookup

def process_csv(csv_path, gt_lookup):
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        print(f"[Skip] 'filename' column missing in {os.path.basename(csv_path)}")
        return None

    base_name = os.path.basename(csv_path)
    model_name = base_name.replace("_detection.csv", "").replace(".csv", "")

    records = []
    pred_cols = ["has_blur", "has_noise", "has_low_resolution", "has_compression", "has_poor_lighting"]
    
    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        clean_key = fname.lower()
        
        gt = gt_lookup.get(clean_key)
        if not gt:
            continue
            
        pred_vec = [0] * 5
        for i, col in enumerate(pred_cols):
            val = row.get(col, False)
            if str(val).lower() in ["true", "1", "yes"]:
                pred_vec[i] = 1
                
        records.append({
            "Model": model_name,
            "Filename": fname,
            "Level": gt["Level"],
            "GT_Vector": gt["GT_Vector"],
            "GT_Hard_Artifact": gt["GT_Hard_Artifact"],
            "Pred_Vector": pred_vec,
            "Quality_Score": row.get("quality_score", np.nan)
        })

    if not records:
        return None
        
    return pd.DataFrame(records)

def main():
    print("=== Mix Degradation Analysis v8 (CSV Mode) ===")
    
    gt_lookup = load_gt_data()
    if not gt_lookup: return

    csv_pattern = os.path.join(INPUT_DIR, "*_detection.csv")
    files = glob.glob(csv_pattern)
    
    if not files:
        files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
        
    print(f"Found {len(files)} detection CSVs in {INPUT_DIR}")

    count = 0
    for f in files:
        model_name = os.path.basename(f).replace("_detection.csv", "").replace(".csv", "")
        print(f"Processing {model_name}...", end="\r")
        
        df_out = process_csv(f, gt_lookup)
        if df_out is not None:
            save_path = os.path.join(OUTPUT_DIR, f"{model_name}_mix_predictions.csv")
            df_out.to_csv(save_path, index=False)
            count += 1
            
    print(f"\nSuccessfully processed {count} models.")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()