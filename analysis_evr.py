import os
import math
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

OUTPUT_DIR = PROJECT_ROOT / "analysis_metadata" / "evr"

DATASETS = {
    "lfw_arcface": {
        "score_dataset": "lfw",
        "embeddings_dir": PROJECT_ROOT / "saved_embeddings" / "arcface_ir50" / "lfw_mtcnn_aligned",
        "pairs_file": PROJECT_ROOT / "lfw_pairs.txt",
    },
    "lfw_lvface": {
        "score_dataset": "lfw",
        "embeddings_dir": PROJECT_ROOT / "saved_embeddings" / "LVFaceS" / "lfw_mtcnn_aligned",
        "pairs_file": PROJECT_ROOT / "lfw_pairs.txt",
    },
    "lfw_transface": {
        "score_dataset": "lfw",
        "embeddings_dir": PROJECT_ROOT / "saved_embeddings" / "transFaceS" / "lfw_mtcnn_aligned",
        "pairs_file": PROJECT_ROOT / "lfw_pairs.txt",
    },
}

TARGET_FMRS = [1e-3]

REJECT_RATES = np.unique(np.concatenate([
    np.arange(0.0, 0.101, 0.005),
    np.arange(0.11, 0.501, 0.05),
])).astype(float)

TIE_RUNS = 10
RANDOM_SEED = 42

import loader_utils  

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def normalize_key(name):
    base = os.path.basename(str(name))
    if base.lower().endswith(".npy"):
        base = base[:-4]
    root, ext = os.path.splitext(base)
    if ext.lower() in IMG_EXTS:
        base = root
    return base

def _load_embedding_map(embeddings_dir):
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings dir not found: {embeddings_dir}")
    emb_map = {}
    for p in sorted(embeddings_dir.glob("*.npy")):
        key = normalize_key(p.name)
        v = np.load(p).reshape(-1)
        if not np.isfinite(v).all():
            v = np.nan_to_num(v)
        d = v.shape[0] // 2
        v = v[:d] + v[d:]
        n = np.linalg.norm(v)
        if n > 0: v = v / n
        emb_map[key] = v
    return emb_map

def _parse_lfw_pairs(pairs_file):
    if not pairs_file.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    lines = pairs_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    parsed = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            parsed.append((f"{parts[0]}_{int(parts[1]):04d}", f"{parts[0]}_{int(parts[2]):04d}", 1))
        elif len(parts) == 4:
            parsed.append((f"{parts[0]}_{int(parts[1]):04d}", f"{parts[2]}_{int(parts[3]):04d}", 0))
    return parsed

def build_pairs_df(variant_name, embeddings_dir, pairs_file, variant_cache_dir):
    variant_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = variant_cache_dir / f"{variant_name}__pairs_cache.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    emb_map = _load_embedding_map(embeddings_dir)
    pairs = _parse_lfw_pairs(pairs_file)
    rows = []
    for f1, f2, label in pairs:
        if f1 in emb_map and f2 in emb_map:
            rows.append((f1, f2, int(label), float(np.dot(emb_map[f1], emb_map[f2]))))
    
    if not rows: raise RuntimeError(f"No pairs for {variant_name}")
    df = pd.DataFrame(rows, columns=["f1", "f2", "label", "sim_score"])
    df.to_csv(cache_path, index=False)
    return df

def threshold_at_target_fmr(sims, labels, target_fmr):
    imp = sims[labels == 0]
    imp = imp[np.isfinite(imp)]
    if imp.size < 10: return float("nan")
    return float(np.quantile(imp, 1.0 - target_fmr))

def compute_fmr_fnmr(sims, labels, th):
    mask = np.isfinite(sims)
    s, l = sims[mask], labels[mask]
    imp, gen = (l == 0), (l == 1)
    if not np.any(imp) or not np.any(gen): return float("nan"), float("nan")
    pred = s >= th
    return float(np.sum(pred & imp) / np.sum(imp)), float(np.sum((~pred) & gen) / np.sum(gen))

def _select_keep_indices(qualities, rate, rng):
    N = len(qualities)
    k = int(math.floor(rate * N))
    if k <= 0: return np.ones(N, dtype=bool)
    if k >= N: return np.zeros(N, dtype=bool)
    
    order = np.argsort(qualities, kind="mergesort")
    cutoff = qualities[order[k-1]]
    
    lower = qualities < cutoff
    equal = qualities == cutoff
    need = k - int(np.sum(lower))
    
    if need > 0:
        eq_idx = np.flatnonzero(equal)
        chosen = rng.choice(eq_idx, size=need, replace=False)
        lower[chosen] = True
    
    return ~lower

def _auc_trapz(x, y):
    if x.size < 2: return float("nan")
    return float(np.trapz(y, x))

def _pauc(x, y, xmax):
    if x.size < 2: return float("nan"), float("nan")
    xmax = float(xmax)
    if x[-1] < xmax:
        j = np.searchsorted(x, xmax)
        y_end = np.interp(xmax, x, y)
        x = np.append(x[:j], xmax)
        y = np.append(y[:j], y_end)
    else:
        mask = x <= xmax
        x, y = x[mask], y[mask]
    
    area = _auc_trapz(x, y)
    norm = area / xmax if xmax > 0 else float("nan")
    return area, norm

def compute_evr_for_model(pairs_df, image_quality, model_key, variant_name, target_fmr, tie_runs, seed):
    df = pairs_df.copy()
    df["q1"] = df["f1"].map(image_quality)
    df["q2"] = df["f2"].map(image_quality)
    df = df.dropna(subset=["q1", "q2"]).reset_index(drop=True)
    if df.empty: raise RuntimeError("No pairs with quality")

    pair_q = np.minimum(df["q1"], df["q2"])
    sims = df["sim_score"].values
    labels = df["label"].values
    base_th = threshold_at_target_fmr(sims, labels, target_fmr)

    agg = {"fixed": {"fnmr": [], "n": []}, "recompute": {"fnmr": [], "n": []}}

    for run in range(tie_runs):
        rng = np.random.default_rng(seed + run)
        fn_fix, fn_rec, n_lst = [], [], []
        
        for r in REJECT_RATES:
            keep = _select_keep_indices(pair_q, r, rng)
            s_k, l_k = sims[keep], labels[keep]
            n_lst.append(np.sum(keep))
            
            _, fn1 = compute_fmr_fnmr(s_k, l_k, base_th)
            fn_fix.append(fn1)
            
            th2 = threshold_at_target_fmr(s_k, l_k, target_fmr)
            _, fn2 = compute_fmr_fnmr(s_k, l_k, th2)
            fn_rec.append(fn2)
            
        agg["fixed"]["fnmr"].append(fn_fix)
        agg["fixed"]["n"].append(n_lst)
        agg["recompute"]["fnmr"].append(fn_rec)
        agg["recompute"]["n"].append(n_lst)

    metrics_rows, curves_rows = [], []
    rr = REJECT_RATES

    for mode_key, mode_name in [("fixed", "fixed_threshold"), ("recompute", "recompute_threshold")]:
        mat_fnmr = np.array(agg[mode_key]["fnmr"])
        if mat_fnmr.ndim != 2: continue
        
        mean_fnmr = np.mean(mat_fnmr, axis=0)
        mean_n = np.mean(agg[mode_key]["n"], axis=0)

        for i, r in enumerate(rr):
            curves_rows.append({
                "variant": variant_name, "model_key": model_key, "target_fmr": target_fmr,
                "mode": mode_name, "reject_rate": r, "fnmr": mean_fnmr[i], "pairs": mean_n[i]
            })

        auc = _auc_trapz(rr, mean_fnmr)
        p1, np1 = _pauc(rr, mean_fnmr, 0.01)
        p5, np5 = _pauc(rr, mean_fnmr, 0.05)
        p10, np10 = _pauc(rr, mean_fnmr, 0.10)
        p20, np20 = _pauc(rr, mean_fnmr, 0.20)

        metrics_rows.append({
            "variant": variant_name, "model_key": model_key, "target_fmr": target_fmr,
            "mode": mode_name, "auc": auc,
            "pauc_1": p1, "npauc_1": np1,
            "pauc_5": p5, "npauc_5": np5,
            "pauc_10": p10, "npauc_10": np10,
            "pauc_20": p20, "npauc_20": np20
        })

    return pd.DataFrame(curves_rows), pd.DataFrame(metrics_rows)


def _quality_series(df, score_col):
    df["key"] = df["filename"].astype(str).apply(normalize_key)
    return df.set_index("key")[score_col]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    global_curves, global_metrics = [], []
    vlm_models = []
    for f in [loader_utils.get_all_real_models, loader_utils.get_all_synthetic_models, loader_utils.get_mix_models]: vlm_models.extend(f())
    vlm_models = list(set(vlm_models))

    for variant_name, dcfg in DATASETS.items():
        score_ds = dcfg.get("score_dataset", variant_name)
        v_cache = OUTPUT_DIR / "variants" / variant_name / "cache"
        try:
            pairs_df = build_pairs_df(variant_name, Path(dcfg["embeddings_dir"]), Path(dcfg["pairs_file"]), v_cache)
        except Exception as e:
            print(f"[Skip] {variant_name}: {e}")
            continue
            
        sota = loader_utils.load_sota_scores(score_ds) or {}
        for m_name, df in sota.items():
            print(f"Processing SOTA: {m_name}")
            c, m = compute_evr_for_model(pairs_df, _quality_series(df, "sota_score"), f"SOTA__{m_name}", variant_name, TARGET_FMRS[0], TIE_RUNS, RANDOM_SEED)
            global_curves.append(c); global_metrics.append(m)

        for m_name, cat in vlm_models:
            df = loader_utils.load_vlm_predictions(m_name, score_ds, cat)
            if df is not None and not df.empty:
                print(f"Processing VLM: {m_name}")
                c, m = compute_evr_for_model(pairs_df, _quality_series(df, "predicted_score"), f"VLM__{cat}__{m_name}", variant_name, TARGET_FMRS[0], TIE_RUNS, RANDOM_SEED)
                global_curves.append(c); global_metrics.append(m)

    if global_metrics:
        pd.concat(global_metrics).to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
        pd.concat(global_curves).to_csv(OUTPUT_DIR / "curves_all.csv", index=False)
        print(f"Done. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()