import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

INPUT_DIR = "analysis_metadata/prompt_consistency"

FIG_DIR = "paper_figures/consistency"
TAB_DIR = "paper_tables"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

VLM_NAME_MAP = {
    "qwen2p5_32b": "Qwen2.5-32B",
    "qwen2p5_7b": "Qwen2.5-7B",
    "qwen2p5_72b": "Qwen2.5-72B",
    "qwen2_7b": "Qwen2-7B",
    "phi4": "Phi-4",
    "idefics": "Idefics",
    "gemma3": "Gemma-3"
}

def load_and_prep_data():
    data_dir = INPUT_DIR
    
    print(f"Loading data from: {data_dir}")

    df_fmt = pd.DataFrame()
    fmt_files = glob.glob(os.path.join(data_dir, "*format*.csv"))
    if fmt_files:
        df_fmt = pd.read_csv(fmt_files[0])
        df_fmt = df_fmt.rename(columns={
            'Model': 'Model',
            'Pearson_r': 'Pearson',
            'Bias (Class - Simple)': 'Bias'
        })
        df_fmt['Variant_Type'] = 'Classification'
    else:
        print("[Warn] No format consistency data found.")

    df_sem = pd.DataFrame()
    sem_files = glob.glob(os.path.join(data_dir, "*ablation*.csv"))
    if sem_files:
        df_sem = pd.read_csv(sem_files[0])
        # Standardize
        df_sem = df_sem.rename(columns={
            'Base_Model': 'Model',
            'Pearson_r': 'Pearson',
            'Bias (Variant - Base)': 'Bias'
        })
    else:
        print("[Warn] No semantic consistency data found.")

    if df_fmt.empty and df_sem.empty:
        return pd.DataFrame()
    
    full_df = pd.concat([df_fmt, df_sem], ignore_index=True)
    
    full_df['Model_Display'] = full_df['Model'].map(VLM_NAME_MAP).fillna(full_df['Model'])
    
    return full_df

def generate_long_format_table(df):

    print("--- Generating Unified Long-Format Table ---")
    
    grouped = df.groupby(['Model_Display', 'Variant_Type']).agg({
        'MAE': 'mean',
        'Pearson': 'mean',
        'Bias': 'mean'
    }).reset_index()
    
    grouped = grouped.rename(columns={
        'Model_Display': 'Model',
        'Variant_Type': 'Prompt Combination'
    })
    
    grouped['MAE'] = grouped['MAE'].round(3)
    grouped['Pearson'] = grouped['Pearson'].round(3)
    grouped['Bias'] = grouped['Bias'].round(3)
    
    grouped = grouped.sort_values(['Model', 'Prompt Combination'])
    
    csv_path = os.path.join(TAB_DIR, "Table_Consistency_Long.csv")
    grouped.to_csv(csv_path, index=False)
    

    print(f"  Saved Table: {csv_path}")

def plot_format_consistency(df):
    print("--- Generating Format Consistency Plot ---")
    
    sub = df[df['Variant_Type'] == 'Classification'].copy()
    if sub.empty: return
    
    agg = sub.groupby('Model_Display').agg({'MAE': 'mean', 'Pearson': 'mean', 'Bias': 'mean'}).reset_index()
    
    fig, axes = plt.subplots(3, 1, figsize=(6, 18))
    
    sns.barplot(data=agg, x='Model_Display', y='MAE', ax=axes[0], palette="Blues_d", edgecolor='black')
    axes[0].set_title("Inconsistency (MAE)")
    axes[0].set_ylabel("MAE (Lower is Better)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=agg, x='Model_Display', y='Pearson', ax=axes[1], palette="Greens_d", edgecolor='black')
    axes[1].set_title("Correlation (Pearson r)")
    axes[1].set_ylabel("Pearson Correlation (Higher is Better)")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis='x', rotation=45)

    sns.barplot(data=agg, x='Model_Display', y='Bias', ax=axes[2], palette="RdBu", edgecolor='black')
    axes[2].set_title("Bias (Classification - Simple)")
    axes[2].set_ylabel("Bias (Positive = Class Higher)")
    axes[2].set_xlabel("")
    axes[2].axhline(0, color='black', linewidth=1)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle("Format Consistency: Simple vs. Classification (JSON)", fontsize=16)
    
    save_path = os.path.join(FIG_DIR, "Fig_Format_Consistency.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ablation_consistency(df):
    print("--- Generating Prompt Ablation Plot ---")
    
    ablation_models = df[df['Variant_Type'].isin(['Utility', 'Reliability'])]['Model_Display'].unique()
    
    if len(ablation_models) == 0:
        return

    sub = df[df['Model_Display'].isin(ablation_models)].copy()
    
    agg = sub.groupby(['Model_Display', 'Variant_Type']).agg({'MAE': 'mean', 'Pearson': 'mean', 'Bias': 'mean'}).reset_index()
    
    fig, axes = plt.subplots(3, 1, figsize=(6, 18))
    
    sns.barplot(data=agg, x='Model_Display', y='MAE', hue='Variant_Type', ax=axes[0], palette="viridis", edgecolor='black')
    axes[0].set_title("Inconsistency (MAE)")
    axes[0].set_ylabel("MAE (Lower is Better)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=agg, x='Model_Display', y='Pearson', hue='Variant_Type', ax=axes[1], palette="viridis", edgecolor='black')
    axes[1].set_title("Correlation (Pearson r)")
    axes[1].set_ylabel("Pearson Correlation (Higher is Better)")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=agg, x='Model_Display', y='Bias', hue='Variant_Type', ax=axes[2], palette="coolwarm", edgecolor='black')
    axes[2].set_title("Bias (Variant - Base)")
    axes[2].set_ylabel("Bias (Positive = Variant Higher)")
    axes[2].set_xlabel("")
    axes[2].axhline(0, color='black', linewidth=1)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle("Prompt Ablation: Classification vs. Utility vs. Reliability", fontsize=16)
    
    save_path = os.path.join(FIG_DIR, "Fig_Prompt_Ablation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df = load_and_prep_data()
    
    if df.empty:
        print("No consistency data found.")
        return

    generate_long_format_table(df)
    plot_format_consistency(df)
    plot_ablation_consistency(df)
    
    print(f"\nPart 4 Complete. Figures in {FIG_DIR}, Table in {TAB_DIR}")

if __name__ == "__main__":
    main()