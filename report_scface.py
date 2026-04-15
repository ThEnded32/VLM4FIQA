from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = Path("analysis_metadata") / "scface_full"
FIG_DIR = Path("paper_figures") / "scface_full"
TAB_DIR = Path("paper_tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

VLM_NAME_MAP = {
    "qwen2p5_32b": "Qwen2.5-32B",
    "qwen2p5_7b": "Qwen2.5-7B",
    "qwen2p5_72b": "Qwen2.5-72B",
    "qwen2_7b": "Qwen2-7B",
    "phi4": "Phi-4",
    "idefics": "Idefics",
    "gemma3": "Gemma-3",
}

DIST_ORDER_IDS = ["1", "2", "3"]
XTICKS_SHORT = ["d1 (Far)", "d2 (Med)", "d3 (Close)"]
DIST_NAME_FOR_COL = {"1": "Far", "2": "Med", "3": "Close"}

Y_LIM = (0, 100)

ATTRIBUTES = ["Sharpness", "Resolution", "Compression", "Lighting"]

LEVEL_ORDER = {
    "Sharpness": ["Clear", "Slightly Blurred", "Moderately Blurred", "Strongly Blurred"],
    "Resolution": ["High", "Medium", "Low", "Very Low"],
    "Compression": ["None", "Minimal", "Moderate", "Severe"],
}

def _pretty_model(m: str) -> str:
    return VLM_NAME_MAP.get(m, m)


def _load():
    long_path = INPUT_DIR / "scface_long.csv"
    sum_path = INPUT_DIR / "scface_scores_summary.csv"
    dist_path = INPUT_DIR / "scface_labels_distribution.csv"

    if not long_path.exists() or not sum_path.exists():
        raise FileNotFoundError(f"Missing inputs in {INPUT_DIR}. Run scface_analyze_runall_full.py first.")

    long_df = pd.read_csv(long_path)
    sum_df = pd.read_csv(sum_path)
    dist_df = pd.read_csv(dist_path) if dist_path.exists() else pd.DataFrame()

    for df in (long_df, sum_df, dist_df):
        if not df.empty and "Distance_ID" in df.columns:
            df["Distance_ID"] = pd.Categorical(df["Distance_ID"].astype(str), categories=DIST_ORDER_IDS, ordered=True)
    return long_df, sum_df, dist_df


def generate_distance_table(summary_df: pd.DataFrame):
    df = summary_df.copy()
    df["Model_Display"] = df["Model"].astype(str).apply(_pretty_model)

    pivot = df.pivot_table(
        index="Model_Display",
        columns=["Prompt_Type", "Distance_ID"],
        values="Mean",
        aggfunc="mean",
    )

    col_order = []
    for prompt in ["Simple", "Classification"]:
        for did in ["3", "2", "1"]:
            if (prompt, did) in pivot.columns:
                col_order.append((prompt, did))
    pivot = pivot[col_order]

    pivot.columns = [f"{p}_{DIST_NAME_FOR_COL[str(d)]}" for (p, d) in pivot.columns]
    out = pivot.reset_index()
    for c in out.columns[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    csv_path = TAB_DIR / "Table_SCFace_Distance_Scores.csv"
    out.to_csv(csv_path, index=False)

    print(f"[OK] Saved table: {csv_path} (+ .tex)")


def plot_score_vs_distance_global(long_df: pd.DataFrame):
    df = long_df.copy()
    if df.empty:
        return

    per_model = df.groupby(["Model", "Prompt_Type", "Distance_ID"])["predicted_score"].mean().reset_index()
    per_model["Model_Display"] = per_model["Model"].astype(str).apply(_pretty_model)

    for prompt, fname, marker, linestyle in [
        ("Simple", "Score_vs_Dist_Simple.png", "o", "-"),
        ("Classification", "Score_vs_Dist_Classification.png", "s", "--"),
    ]:
        sub = per_model[per_model["Prompt_Type"] == prompt].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))
        for model in sorted(sub["Model_Display"].unique().tolist()):
            mm = sub[sub["Model_Display"] == model].sort_values("Distance_ID")
            x = np.arange(len(DIST_ORDER_IDS))
            y = []
            for did in DIST_ORDER_IDS:
                r = mm[mm["Distance_ID"].astype(str) == did]
                y.append(float(r["predicted_score"].iloc[0]) if not r.empty else np.nan)
            ax.plot(x, y, linewidth=2.5, marker=marker, linestyle=linestyle, label=model)

        ax.set_xticks(np.arange(len(DIST_ORDER_IDS)))
        ax.set_xticklabels(XTICKS_SHORT, fontsize=14)
        ax.set_ylabel("Mean Quality Score", fontsize=14)
        ax.set_title(f"SCFace: Quality Score vs Distance ({prompt} Prompt)", fontsize=18, pad=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=10)
        if Y_LIM:
            ax.set_ylim(*Y_LIM)

        out = FIG_DIR / fname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: {out}")


def plot_per_model_comparison(long_df: pd.DataFrame):
    df = long_df.copy()
    if df.empty:
        return

    models = sorted(df["Model"].dropna().unique().tolist())
    for model in models:
        sub = df[df["Model"] == model].copy()
        if sub.empty:
            continue

        g = sub.groupby(["Prompt_Type", "Distance_ID"])["predicted_score"].agg(["mean", "std"]).reset_index()
        prompts = set(g["Prompt_Type"].unique().tolist())
        if len(prompts) < 2:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        for prompt, marker, linestyle in [("Classification", "o", "-"), ("Simple", "o", "-")]:
            gg = g[g["Prompt_Type"] == prompt].sort_values("Distance_ID")
            if gg.empty:
                continue

            x = np.arange(len(DIST_ORDER_IDS))
            y = []
            s = []
            for did in DIST_ORDER_IDS:
                r = gg[gg["Distance_ID"].astype(str) == did]
                if r.empty:
                    y.append(np.nan); s.append(np.nan)
                else:
                    y.append(float(r["mean"].iloc[0])); s.append(float(r["std"].iloc[0]))
            y = np.array(y, dtype=float)
            s = np.array(s, dtype=float)

            ax.plot(x, y, linewidth=3, marker=marker, linestyle=linestyle, label=prompt)
            if np.isfinite(s).any():
                ax.fill_between(x, y - s, y + s, alpha=0.15)

        ax.set_xticks(np.arange(len(DIST_ORDER_IDS)))
        ax.set_xticklabels(XTICKS_SHORT, fontsize=14)
        ax.set_ylabel("Mean Quality Score", fontsize=14)
        ax.set_title(f"{_pretty_model(model)}: Quality vs Distance (SCFace)", fontsize=18, pad=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=12)
        if Y_LIM:
            ax.set_ylim(*Y_LIM)

        out = FIG_DIR / f"SCFace_DistCompare_{model}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Saved per-model comparisons to: {FIG_DIR}")


def plot_level_occurrence(long_df: pd.DataFrame):
    df = long_df.copy()
    df = df[df["Prompt_Type"] == "Classification"].copy()
    if df.empty:
        print("[Info] No classification data for level occurrence.")
        return

    for model in sorted(df["Model"].unique().tolist()):
        mdf = df[df["Model"] == model].copy()
        if mdf.empty:
            continue

        for attr in ATTRIBUTES:
            if attr not in mdf.columns:
                continue

            counts = mdf.groupby(["Distance_ID", attr]).size().unstack(fill_value=0)
            counts = counts.reindex(DIST_ORDER_IDS, fill_value=0)

            denom = counts.sum(axis=1).replace(0, np.nan)
            props = counts.div(denom, axis=0) * 100.0
            props = props.fillna(0.0)

            if attr in LEVEL_ORDER:
                cols = [c for c in LEVEL_ORDER[attr] if c in props.columns] + [c for c in props.columns if c not in LEVEL_ORDER[attr]]
            else:
                cols = sorted(props.columns.tolist())
            props = props[cols] if cols else props

            if props.empty:
                continue

            fig, ax = plt.subplots(figsize=(4, 4))
            bottom = np.zeros(len(DIST_ORDER_IDS), dtype=float)
            x = np.arange(len(DIST_ORDER_IDS))

            for col in props.columns:
                y = props[col].to_numpy(dtype=float)
                ax.bar(x, y, bottom=bottom, edgecolor="black", width=0.7, label=str(col))
                bottom += y

            ax.set_xticks(x)
            ax.set_xticklabels(XTICKS_SHORT, rotation=0)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Percentage (%)")
            ax.set_title(f"{_pretty_model(model)}: {attr} vs Distance")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.legend(title="Level", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)

            out = FIG_DIR / f"Level_Dist_{model}_{attr}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print(f"[OK] Saved level occurrence plots to: {FIG_DIR}")


def main():
    print("=== SCFace report (full old-style) ===")
    print(f"INPUT_DIR: {INPUT_DIR.resolve()}")
    print(f"FIG_DIR:   {FIG_DIR.resolve()}")
    print(f"TAB_DIR:   {TAB_DIR.resolve()}")

    long_df, summary_df, _ = _load()

    generate_distance_table(summary_df)
    plot_score_vs_distance_global(long_df)
    plot_per_model_comparison(long_df)
    plot_level_occurrence(long_df)

    print("Done.")


if __name__ == "__main__":
    main()
