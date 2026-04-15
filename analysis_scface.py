import os
import re
from pathlib import Path
import pandas as pd
import loader_utils 

OUTPUT_DIR = Path("analysis_metadata") / "scface_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCAN_REAL_SIMPLE = True
SCAN_REAL_CLASSIFICATION = True

SCAN_SYNTHETIC_SINGLE = False
SCAN_SYNTHETIC_CLASSIFICATION = False
SCAN_MIX = False

ATTRIBUTES = ["Sharpness", "Resolution", "Compression", "Lighting"]

DIST_ID_TO_LABEL_LONG = {"1": "Far (4.2m)", "2": "Med (2.6m)", "3": "Close (1.0m)"}
DIST_ID_TO_LABEL_SHORT = {"1": "d1 (Far)", "2": "d2 (Med)", "3": "d3 (Close)"}
DIST_ORDER_IDS = ["1", "2", "3"]

def _basename(x):
    return os.path.basename(str(x))


def parse_scface_distance(filename):

    base = _basename(filename).lower().strip()

    for tok, did in [("d1", "1"), ("d2", "2"), ("d3", "3")]:
        if re.search(rf"(?:^|[_\-]){tok}(?:[_\-]|\.)", base):
            return did, DIST_ID_TO_LABEL_LONG[did], DIST_ID_TO_LABEL_SHORT[did]

    m = re.search(r"_([1-3])(?=\.[a-z0-9]+$)", base)
    if m:
        did = m.group(1)
        return did, DIST_ID_TO_LABEL_LONG[did], DIST_ID_TO_LABEL_SHORT[did]

    return None


def _pick_scface_parquet(model_dir):
    if not model_dir.exists():
        return None
    files = sorted([p for p in model_dir.iterdir() if p.is_file() and p.suffix.lower() == ".parquet"])
    sc = [p for p in files if "scface" in p.name.lower()]
    return sc[0] if sc else None


def _ensure_filename(raw):
    if "filename" in raw.columns:
        return raw["filename"].astype(str).apply(os.path.basename)

    for c in raw.columns:
        cl = c.lower()
        if cl == "image_path" or ("path" in cl) or ("img" in cl) or ("name" in cl):
            return raw[c].astype(str).apply(os.path.basename)

    if "index" in raw.columns:
        return raw["index"].astype(str).apply(os.path.basename)

    return None


def _extract_score(raw):

    numeric_candidates = [
        "Quality Score", "quality_score", "Quality_Score", "overall_score", "score", "prediction",
        "Reliability Score", "reliability_score", "Reliability",
        "Utility Score", "utility_score", "Utility",
    ]
    for cand in numeric_candidates:
        if cand in raw.columns:
            s = pd.to_numeric(raw[cand], errors="coerce")
            if s.notna().sum() > 0:
                return s

    if "Sharpness" in raw.columns:
        mapping = loader_utils.TEXT_TO_NUMERIC.get("Sharpness", {})
        s = raw["Sharpness"].map(mapping)
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() > 0:
            return s

    return None


def iter_model_categories():
    models = []
    root = Path(loader_utils.DATASET_ROOT)

    def add_dir(dirname):
        d = root / dirname
        if d.exists():
            for m in sorted([p.name for p in d.iterdir() if p.is_dir()]):
                models.append((m, dirname))

    if SCAN_REAL_SIMPLE:
        add_dir("real_simple")
    if SCAN_REAL_CLASSIFICATION:
        add_dir("real_classification")
    if SCAN_SYNTHETIC_SINGLE:
        add_dir("single_degradation_simple")
    if SCAN_SYNTHETIC_CLASSIFICATION:
        add_dir("single_degradation_classification")
    if SCAN_MIX:
        add_dir("mix_degradation")

    seen = set()
    out = []
    for m, c in models:
        if (m, c) in seen:
            continue
        seen.add((m, c))
        out.append((m, c))
    return out


def prompt_type_from_category(category):
    return "Classification" if "classification" in category else "Simple"


def normalize_model_name(model):
    return (
        model.replace("_classification", "")
             .replace("_utility", "")
             .replace("_reliability", "")
    )


def load_scface_long_for_model(model, category):
    model_dir = Path(loader_utils.DATASET_ROOT) / category / model
    p = _pick_scface_parquet(model_dir)
    if p is None:
        return pd.DataFrame()

    raw = pd.read_parquet(p)

    fn = _ensure_filename(raw)
    if fn is None:
        return pd.DataFrame()

    score = _extract_score(raw)
    if score is None:
        return pd.DataFrame()

    df = pd.DataFrame({
        "filename": fn.to_numpy(),
        "predicted_score": score.to_numpy(),
    })

    df["predicted_score"] = pd.to_numeric(df["predicted_score"], errors="coerce")

    dist = df["filename"].apply(parse_scface_distance)
    df["Distance_ID"] = dist.apply(lambda x: x[0] if x else None)
    df["Distance_Long"] = dist.apply(lambda x: x[1] if x else None)
    df["Distance_Short"] = dist.apply(lambda x: x[2] if x else None)

    df = df.dropna(subset=["predicted_score", "Distance_ID", "Distance_Long", "Distance_Short"])
    if df.empty:
        return pd.DataFrame()

    df["Model"] = normalize_model_name(model)
    df["Category"] = category
    df["Prompt_Type"] = prompt_type_from_category(category)

    for attr in ATTRIBUTES:
        if attr in raw.columns:
            df[attr] = raw[attr].astype(str).to_numpy()

    df["Distance_ID"] = pd.Categorical(df["Distance_ID"].astype(str), categories=DIST_ORDER_IDS, ordered=True)

    keep_cols = [
        "Model", "Category", "Prompt_Type",
        "filename", "predicted_score",
        "Distance_ID", "Distance_Long", "Distance_Short",
    ] + [a for a in ATTRIBUTES if a in df.columns]

    return df[keep_cols].copy()


def build_long():
    all_rows = []
    for model, category in iter_model_categories():
        cur = load_scface_long_for_model(model, category)
        if not cur.empty:
            all_rows.append(cur)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["Model", "Prompt_Type", "Distance_ID", "filename"]).reset_index(drop=True)
    return out


def summarize_scores(long_df):
    df = long_df.copy()

    df["predicted_score"] = pd.to_numeric(df["predicted_score"], errors="coerce")
    df = df.dropna(subset=["predicted_score", "Model", "Category", "Prompt_Type", "Distance_ID", "Distance_Long", "Distance_Short"])
    if df.empty:
        return pd.DataFrame()

    g = df.groupby(
        ["Model", "Category", "Prompt_Type", "Distance_ID", "Distance_Long", "Distance_Short"],
        observed=True
    )["predicted_score"]
    s = g.agg(["count", "mean", "std", "min", "max"]).reset_index()
    s = s.rename(columns={"count": "Count", "mean": "Mean", "std": "Std_Dev", "min": "Min", "max": "Max"})
    return s.sort_values(["Model", "Prompt_Type", "Distance_ID"]).reset_index(drop=True)


def label_distribution(long_df):
    df = long_df[long_df["Prompt_Type"] == "Classification"].copy()
    if df.empty:
        return pd.DataFrame()

    results = []
    for attr in ATTRIBUTES:
        if attr not in df.columns:
            continue

        counts = df.groupby(["Model", "Distance_ID", "Distance_Long", attr], observed=True).size().reset_index(name="Count")
        totals = df.groupby(["Model", "Distance_ID", "Distance_Long"], observed=True).size().reset_index(name="Total_Images_At_Dist")
        merged = pd.merge(counts, totals, on=["Model", "Distance_ID", "Distance_Long"], how="left")
        merged["Frequency_Percent"] = merged["Count"] / merged["Total_Images_At_Dist"] * 100.0
        merged = merged.rename(columns={attr: "Label_Value"})
        merged["Attribute"] = attr
        results.append(
            merged[["Model", "Attribute", "Distance_ID", "Distance_Long", "Label_Value",
                    "Count", "Total_Images_At_Dist", "Frequency_Percent"]]
        )

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    return out.sort_values(["Model", "Attribute", "Distance_ID", "Label_Value"]).reset_index(drop=True)


def main():
    print("=== SCFace analyze (full old-style, v2 fix) ===")
    print(f"OUTPUT_DIR:   {OUTPUT_DIR.resolve()}")
    print(f"DATASET_ROOT: {Path(loader_utils.DATASET_ROOT).resolve()}")

    long_df = build_long()
    if long_df.empty:
        raise SystemExit("No SCFace data found. Check parquet_dataset and that scface parquets exist.")

    long_path = OUTPUT_DIR / "scface_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"[OK] wrote {long_path}  rows={len(long_df)}")

    summary = summarize_scores(long_df)
    sum_path = OUTPUT_DIR / "scface_scores_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"[OK] wrote {sum_path}  rows={len(summary)}")
    if summary.empty:
        print("[Debug] summary is empty. Here are quick checks:")
        print("  predicted_score non-null:", int(pd.to_numeric(long_df["predicted_score"], errors="coerce").notna().sum()))
        print("  Distance_ID counts:\n", long_df["Distance_ID"].value_counts(dropna=False))

    dist = label_distribution(long_df)
    if not dist.empty:
        dist_path = OUTPUT_DIR / "scface_labels_distribution.csv"
        dist.to_csv(dist_path, index=False)
        print(f"[OK] wrote {dist_path}  rows={len(dist)}")
    else:
        print("[Info] No classification label distribution (attributes missing or no classification parquets).")

    print("Done.")


if __name__ == "__main__":
    main()
