import pandas as pd
import numpy as np
import os
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "parquet_dataset")

TEXT_TO_NUMERIC = {
    "Sharpness": {
        "Clear": 0,
        "Slightly Blurred": 1,
        "Moderately Blurred": 2,
        "Strongly Blurred": 3
    },
    "Resolution": {
        "High": 0,
        "Medium": 1,
        "Low": 2,
        "Very Low": 3
    },
    "Compression": {
        "None": 0,
        "Minimal": 1,
        "Moderate": 2,
        "Severe": 3
    },
    "Lighting": {
        "Balanced": 0,
        "Slightly Dark": -1,     "Slightly Bright": 1,
        "Moderately Dark": -2,  "Moderately Bright": 2,
        "Very Dark": -3,        "Very Bright": 3
    }
}

def get_all_real_models():
    models = []
    for cat in ["real_simple", "real_classification"]:
        cat_path = os.path.join(DATASET_ROOT, cat)
        if not os.path.exists(cat_path):
            continue

        # Deterministic ordering
        for model_name in sorted(os.listdir(cat_path)):
            full_path = os.path.join(cat_path, model_name)
            if os.path.isdir(full_path):
                models.append((model_name, cat))
    return models

def get_all_synthetic_models():
    models = []
    target_cats = ["single_degradation_simple", "single_degradation_classification"]

    for cat in target_cats:
        cat_path = os.path.join(DATASET_ROOT, cat)
        if not os.path.exists(cat_path):
            continue

        for model_name in sorted(os.listdir(cat_path)):
            full_path = os.path.join(cat_path, model_name)
            if os.path.isdir(full_path):
                models.append((model_name, cat))
    return models

def get_mix_models():

    models = []
    cat = "mix_degradation"
    cat_path = os.path.join(DATASET_ROOT, cat)

    if os.path.exists(cat_path):
        for model_name in sorted(os.listdir(cat_path)):
            full_path = os.path.join(cat_path, model_name)
            if os.path.isdir(full_path):
                models.append((model_name, cat))
    return models

def _pick_best_dataset_parquet(files, dataset_name, context=""):

    dn = dataset_name.lower().strip()
    candidates = [f for f in files if f.lower().endswith(".parquet") and dn in f.lower()]
    if not candidates:
        return None

    def rank_key(fname):
        stem = os.path.splitext(fname)[0].lower()
        exact = 0 if stem == dn else 1
        starts = 0 if stem.startswith(dn) else 1
        return (exact, starts, len(stem), stem)

    candidates_sorted = sorted(candidates, key=rank_key)
    best = candidates_sorted[0]

    if len(candidates_sorted) > 1:
        warnings.warn(
            f"[{context}] Multiple parquet matches for dataset '{dataset_name}'. "
            f"Using '{best}'. Candidates: {candidates_sorted}"
        )

    return best

def load_vlm_predictions(model_name, dataset_name, category):

    model_dir = os.path.join(DATASET_ROOT, category, model_name)
    if not os.path.exists(model_dir):
        return None

    files = sorted(os.listdir(model_dir))

    best = _pick_best_dataset_parquet(files, dataset_name, context=f"VLM:{category}/{model_name}")
    if not best:
        return None

    found_file = os.path.join(model_dir, best)

    df = pd.read_parquet(found_file)

    if 'filename' not in df.columns:
        path_cols = [c for c in df.columns if ('path' in c.lower()) or ('img' in c.lower()) or ('name' in c.lower())]
        if path_cols:
            df['filename'] = df[path_cols[0]].astype(str).apply(os.path.basename)
        elif 'index' in df.columns:
            df['filename'] = df['index'].astype(str).apply(os.path.basename)
        else:
            return None
    else:
        df['filename'] = df['filename'].astype(str).apply(os.path.basename)

    df['predicted_score'] = np.nan

    numeric_candidates = [
        'Quality Score', 'quality_score', 'Quality_Score', 'overall_score', 'score', 'prediction',
        'Reliability Score', 'reliability_score', 'Reliability',
        'Utility Score', 'utility_score', 'Utility'
    ]

    for cand in numeric_candidates:
        if cand in df.columns:
            df['predicted_score'] = pd.to_numeric(df[cand], errors='coerce')
            if df['predicted_score'].notna().sum() > 0:
                break

    if df['predicted_score'].isna().all() and 'Sharpness' in df.columns:
        df['predicted_score'] = df['Sharpness'].map(TEXT_TO_NUMERIC["Sharpness"])

    df = df.dropna(subset=['predicted_score'])

    return df[['filename', 'predicted_score']]

def load_sota_scores(dataset_name):

    sota_root = os.path.join(DATASET_ROOT, "sota")
    sota_data = {}

    if not os.path.exists(sota_root):
        return {}

    for method in sorted(os.listdir(sota_root)):
        method_path = os.path.join(sota_root, method)
        if not os.path.isdir(method_path):
            continue

        files = sorted(os.listdir(method_path))

        best = _pick_best_dataset_parquet(files, dataset_name, context=f"SOTA:{method}")
        if not best:
            continue

        target_file = os.path.join(method_path, best)

        df = pd.read_parquet(target_file)

        if 'filename' not in df.columns:
            cols = [c for c in df.columns if ('path' in c.lower()) or ('img' in c.lower()) or ('name' in c.lower())]
            if cols:
                df['filename'] = df[cols[0]].astype(str).apply(os.path.basename)
            elif 'index' in df.columns:
                df['filename'] = df['index'].astype(str).apply(os.path.basename)
            else:
                continue
        else:
            df['filename'] = df['filename'].astype(str).apply(os.path.basename)

        numeric_cols = []
        for c in df.columns:
            if c == 'filename':
                continue
            if pd.to_numeric(df[c], errors='coerce').notna().any():
                numeric_cols.append(c)

        if not numeric_cols:
            continue

        chosen = None
        if len(numeric_cols) == 1:
            chosen = numeric_cols[0]
        else:
            score_like = [c for c in numeric_cols if ('score' in c.lower()) or ('quality' in c.lower())]
            if score_like:
                chosen = sorted(score_like, key=lambda c: (0 if 'quality' in c.lower() else 1, c.lower()))[0]
            else:
                chosen = sorted(numeric_cols)[0]

            warnings.warn(
                f"[SOTA:{method}] Multiple numeric columns for '{dataset_name}'. "
                f"Using '{chosen}'. Candidates: {sorted(numeric_cols)}"
            )

        df = df.rename(columns={chosen: 'sota_score'})
        df['sota_score'] = pd.to_numeric(df['sota_score'], errors='coerce')
        df = df.dropna(subset=['sota_score'])

        sota_data[method] = df[['filename', 'sota_score']]

    return sota_data
