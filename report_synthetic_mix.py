import os
import glob
import ast
import numpy as np
import pandas as pd

PRED_DIR = "analysis_metadata/mix_degradation"
TAB_DIR = "paper_tables"
os.makedirs(TAB_DIR, exist_ok=True)

def parse_vector(vec_data):
    if isinstance(vec_data, str):
        return ast.literal_eval(vec_data)
    return vec_data

def load_data():
    files = glob.glob(os.path.join(PRED_DIR, "*_mix_predictions.csv"))
    if not files:
        print(f"[Error] No analysis files found in {PRED_DIR}.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["GT_Vector"] = df["GT_Vector"].apply(parse_vector)
        df["Pred_Vector"] = df["Pred_Vector"].apply(parse_vector)
        dfs.append(df)
            
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def save_table(df, fname):
    path = os.path.join(TAB_DIR, fname)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(df.head().to_string(index=False))
    print("-" * 60)

def report_l0_analysis(df):
    print("\n--- 1. L0 (Clean) Analysis ---")
    sub = df[df["Level"] == "L0"].copy()
    if sub.empty: return

    sub["Hallucinations"] = sub["Pred_Vector"].apply(sum)
    
    results = []
    for m in sub["Model"].unique():
        d = sub[sub["Model"] == m]
        total = len(d)
        if total == 0: continue
        
        row = {"Model": m}

        h0 = d[d["Hallucinations"] == 0]
        row["Clean (0) %"] = (len(h0) / total) * 100
        row["Clean (0) QS"] = h0["Quality_Score"].mean() if not h0.empty else np.nan
        
        h1 = d[d["Hallucinations"] == 1]
        row["1 Halluc. %"] = (len(h1) / total) * 100
        row["1 Halluc. QS"] = h1["Quality_Score"].mean() if not h1.empty else np.nan
        
        h2 = d[d["Hallucinations"] == 2]
        row["2 Halluc. %"] = (len(h2) / total) * 100
        row["2 Halluc. QS"] = h2["Quality_Score"].mean() if not h2.empty else np.nan
        
        h3 = d[d["Hallucinations"] >= 3]
        row[">=3 Halluc. %"] = (len(h3) / total) * 100
        row[">=3 Halluc. QS"] = h3["Quality_Score"].mean() if not h3.empty else np.nan
        
        results.append(row)
        
    df_out = pd.DataFrame(results).sort_values("Clean (0) %", ascending=False)
    save_table(df_out, "Table_Mix_L0.csv")

def report_l2_general(df):
    print("\n--- 2. L2 General (Soft Recall) ---")
    sub = df[df["Level"] == "L2"].copy()
    if sub.empty: return

    def count_found(r):
        gt = np.array(r["GT_Vector"])
        pred = np.array(r["Pred_Vector"])
        return np.sum(gt & pred) 

    sub["Found_Count"] = sub.apply(count_found, axis=1)

    results = []
    for m in sub["Model"].unique():
        d = sub[sub["Model"] == m]
        total = len(d)
        if total == 0: continue
        
        row = {"Model": m}
        
        for i in range(4):
            subset = d[d["Found_Count"] == i]
            pct = (len(subset) / total) * 100
            qs = subset["Quality_Score"].mean() if not subset.empty else np.nan
            
            row[f"Found {i}/3 (%)"] = pct
            row[f"Found {i}/3 (QS)"] = qs
            
        results.append(row)
        
    df_out = pd.DataFrame(results).sort_values("Found 3/3 (%)", ascending=False)
    save_table(df_out, "Table_Mix_L2_General.csv")

def report_l2_extreme(df):
    print("\n--- 3. L2 Extreme Recall ---")
    sub = df[(df["Level"] == "L2") & (df["GT_Hard_Artifact"].notna())].copy()
    if sub.empty: return
    
    ART_MAP = {"Blur":0, "Noise":1, "LowRes":2, "Compression":3, "Lighting":4}
    
    def check_hit(r):
        idx = ART_MAP.get(r["GT_Hard_Artifact"])
        if idx is not None and idx < len(r["Pred_Vector"]):
            return r["Pred_Vector"][idx] == 1
        return False

    sub["Hit"] = sub.apply(check_hit, axis=1)
    
    results = []
    for m in sub["Model"].unique():
        d = sub[sub["Model"] == m]
        total = len(d)
        hits = d[d["Hit"] == True]
        
        row = {
            "Model": m,
            "Extreme Recall (%)": (len(hits)/total*100) if total else 0,
            "Extreme Recall (QS)": hits["Quality_Score"].mean() if not hits.empty else np.nan
        }
        results.append(row)
        
    df_out = pd.DataFrame(results).sort_values("Extreme Recall (%)", ascending=False)
    save_table(df_out, "Table_Mix_L2_Extreme.csv")

def report_hamming(df):
    print("\n--- 4. Hamming Distance Breakdown ---")
    sub = df[df["Level"] == "L2"].copy()
    if sub.empty: return

    sub["Hamming"] = sub.apply(lambda r: sum(abs(a-b) for a,b in zip(r["GT_Vector"], r["Pred_Vector"])), axis=1)
    
    results = []
    for m in sub["Model"].unique():
        d = sub[sub["Model"] == m]
        n = len(d)
        
        grp = d.groupby("Hamming").agg(Count=("Quality_Score", "size"), QS=("Quality_Score", "mean"))
        
        row = {"Model": m}
        for i in range(6):
            if i in grp.index:
                row[f"D{i} (%)"] = (grp.loc[i, "Count"] / n) * 100
                row[f"D{i} (QS)"] = grp.loc[i, "QS"]
            else:
                row[f"D{i} (%)"] = 0.0
                row[f"D{i} (QS)"] = np.nan
                
        results.append(row)

    df_out = pd.DataFrame(results).sort_values("D0 (%)", ascending=False)
    save_table(df_out, "Table_Mix_Hamming.csv")

def main():
    print("=== NEW REPORT GENERATION (Per-Column QS) ===")
    df = load_data()
    if df.empty: return
    
    report_l0_analysis(df)
    report_l2_general(df)
    report_l2_extreme(df)
    report_hamming(df)
    
    print("Done. Tables saved in 'paper_tables/'.")

if __name__ == "__main__":
    main()