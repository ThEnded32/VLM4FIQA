import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

INPUT_DIR = "analysis_metadata/internal_consistency"
FIG_DIR = "paper_figures/internal_consistency"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14, 'figure.autolayout': True})

VLM_NAME_MAP = {
    "qwen2p5_32b": "Qwen2.5-32B",
    "qwen2p5_7b": "Qwen2.5-7B",
    "qwen2p5_72b": "Qwen2.5-72B",
    "qwen2_7b": "Qwen2-7B",
    "phi4": "Phi-4",
    "idefics": "Idefics",
    "gemma3": "Gemma-3"
}

DATASET_NAME_MAP = {
    "lfw": "LFW",
    "ijbb": "IJB-B",
    "celeba": "CelebA",
    "scface": "SCFace"
}

REAL_DATASETS = ["lfw", "ijbb", "celeba", "scface"]

BASE_ATTRIBUTES = ["Sharpness", "Resolution", "Compression", "Lighting"]

def load_and_clean_data():
    files = glob.glob(os.path.join(INPUT_DIR, "*_internal.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    if not dfs: return pd.DataFrame()
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['Dataset_Key'] = full_df['Dataset'].astype(str).str.lower().str.strip()
    full_df['Dataset_Key'] = full_df['Dataset_Key'].str.replace("_original", "").str.replace("_mtcnn_aligned", "")
    full_df = full_df[full_df['Dataset_Key'].isin(REAL_DATASETS)].copy()
    full_df['Dataset_Display'] = full_df['Dataset_Key'].map(DATASET_NAME_MAP)
    
    return full_df

def get_plot_data_for_model(df, model_name):

    model_df = df[df['Model'] == model_name].copy()
    if model_df.empty: return {}

    plot_groups = {}

    def add_global_average(sub_df):
        global_avg = sub_df.groupby('Plot_Level')['Mean_Score'].mean().reset_index()
        global_avg['Dataset_Display'] = "Global Average"
        global_avg['Std_Dev'] = 0 
        
        return pd.concat([sub_df, global_avg], ignore_index=True)

    for attr in ["Sharpness", "Resolution", "Compression"]:
        sub = model_df[model_df['Attribute'] == attr].copy()
        if not sub.empty:
            sub['Plot_Level'] = sub['Label_Ordinal']
            plot_groups[attr] = add_global_average(sub)

    lighting = model_df[model_df['Attribute'] == "Lighting"].copy()
    if not lighting.empty:
        baseline = lighting[lighting['Label_Ordinal'] == 0].copy()
        if not baseline.empty:
            baseline['Plot_Level'] = 0

        under = lighting[lighting['Label_Ordinal'] < 0].copy()
        if not under.empty:
            under['Plot_Level'] = under['Label_Ordinal'].abs()
            combined = pd.concat([baseline, under], ignore_index=True)
            plot_groups["Lighting (Under-Exposure)"] = add_global_average(combined.sort_values('Plot_Level'))

        over = lighting[lighting['Label_Ordinal'] > 0].copy()
        if not over.empty:
            over['Plot_Level'] = over['Label_Ordinal']
            combined = pd.concat([baseline, over], ignore_index=True)
            plot_groups["Lighting (Over-Exposure)"] = add_global_average(combined.sort_values('Plot_Level'))

    return plot_groups

def plot_single_attribute(df, model_disp, attr_name, filename):
    plt.figure(figsize=(8, 6))
    
    datasets = sorted([d for d in df['Dataset_Display'].unique() if d != "Global Average"])
    palette = sns.color_palette("husl", len(datasets))
    color_map = dict(zip(datasets, palette))
    color_map["Global Average"] = "black"

    style_map = {d: (1, 0) for d in datasets}
    style_map["Global Average"] = (4, 1.5)

    sns.lineplot(
        data=df,
        x='Plot_Level',
        y='Mean_Score',
        hue='Dataset_Display',
        style='Dataset_Display',
        dashes=style_map,
        palette=color_map,
        linewidth=2.5,
        markers=True,
        markersize=8
    )

    plt.title(f"{model_disp}: {attr_name}", fontsize=16)
    plt.xlabel("Predicted Degradation Level", fontsize=14)
    plt.ylabel("Mean Quality Score", fontsize=14)
    plt.ylim(0, 100)
    plt.xticks([0, 1, 2, 3], ["0 (Clear)", "1 (Mild)", "2 (Mod)", "3 (Severe)"])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title="Dataset", fontsize=11)
    
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def main():
    full_df = load_and_clean_data()
    if full_df.empty:
        print("No internal consistency data found.")
        return

    models = full_df['Model'].unique()
    
    print("--- Generating Internal Consistency Plots (5 Lines per Plot) ---")

    for model in models:
        if "utility" in model or "reliability" in model: continue
        
        model_disp = VLM_NAME_MAP.get(model, model)
        plot_groups = get_plot_data_for_model(full_df, model)

        for attr_name, df_attr in plot_groups.items():
            attr_clean = attr_name.replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
            fname = f"Internal_{model}_{attr_clean}.png"
            
            plot_single_attribute(df_attr, model_disp, attr_name, fname)

    print(f"\nPart 3 Complete. Figures in {FIG_DIR}")

if __name__ == "__main__":
    main()