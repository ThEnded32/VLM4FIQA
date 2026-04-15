from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

INPUT_DIR = PROJECT_ROOT / "analysis_metadata" / "evr"

FIG_DIR = PROJECT_ROOT / "paper_figures" / "evr_curves"

INCLUDE_VARIANTS = []   
INCLUDE_TARGET_FMRS = [] 

X_MAX = 0.2
Y_MAX = 0.12  

MODES_TO_PLOT = ["recompute_threshold"] # ["fixed_threshold", "recompute_threshold"]

PLOT_PROTOCOL_COMPARE = True

DPI = 300
FONT_SIZE = 14

plt.rcParams.update({"font.size": FONT_SIZE, "figure.autolayout": True})

VLM_NAME_MAP = {
    "qwen2p5_32b": "Qwen2.5-32B",
    "qwen2p5_7b": "Qwen2.5-7B",
    "qwen2p5_72b": "Qwen2.5-72B",
    "qwen2_7b": "Qwen2-7B",
    "phi4": "Phi-4",
    "idefics": "Idefics",
    "gemma3": "Gemma-3",
}

SOTA_NAME_MAP = {
    "ediffiqa_large": "eDifFIQA",
    "faceqan": "FaceQAN",
    "sddfiqa": "SDD-FIQA",
    "vitfiqa_T": "ViT-FIQA",
}

COLOR_PALETTE = {
    "qwen2p5_7b": "#e377c2",  
    "qwen2p5_32b": "#9467bd", 
    "qwen2p5_72b": "#8c564b", 
    "qwen2_7b": "#d62728",    
    "phi4": "#2ca02c",        
    "gemma3": "#1f77b4",      
    "idefics": "#ff7f0e",     
    "ediffiqa_large": "#ff7f0e",
    "faceqan": "#2ca02c",
    "sddfiqa": "#d62728",
    "vitfiqa_T": "#9467bd",
}

def _safe_filename(s):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))


def _pretty_base_name(base):
    if base in VLM_NAME_MAP:
        return VLM_NAME_MAP[base]
    if base in SOTA_NAME_MAP:
        return SOTA_NAME_MAP[base]
    return base


def _load_curves_all():

    p = INPUT_DIR / "curves_all.csv"
    if p.exists():
        return pd.read_csv(p)

    rows = []
    for f in sorted(INPUT_DIR.rglob("*__curves.csv")):
        rows.append(pd.read_csv(f))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _infer_kind(model_key):
    if model_key.startswith("SOTA__"):
        return "SOTA"
    if model_key.startswith("VLM__"):
        return "VLM"
    return "OTHER"


def _parse_model_key(model_key):

    kind = _infer_kind(model_key)

    if kind == "SOTA":
        base = model_key.split("SOTA__", 1)[1]
        return kind, base, "simple"

    if kind == "VLM":
        parts = model_key.split("__", 2)
        if len(parts) < 3:
            return kind, model_key, "other"
        _, category, model_name = parts

        variant = "simple"
        base = model_name

        if "classification" in category:
            variant = "classification"

        if model_name.endswith("_utility"):
            variant = "utility"
            base = model_name[: -len("_utility")]
        elif model_name.endswith("_reliability"):
            variant = "reliability"
            base = model_name[: -len("_reliability")]

        return kind, base, variant

    return kind, model_key, "other"


def _extract_xy(df):
    g = df.sort_values("reject_rate")
    x = g["reject_rate"].to_numpy(dtype=float)
    y = g["fnmr"].to_numpy(dtype=float)
    return x, y


def _plot_overlay(items, title, out_path, color_map=COLOR_PALETTE):

    if not items:
        return

    plt.figure(figsize=(9, 6))
    
    for label, dfi in items:
        x, y = _extract_xy(dfi)
        
        line_color = None
        if color_map and not dfi.empty:
            mk = dfi["model_key"].iloc[0]
            _, base_name, _ = _parse_model_key(mk)
            line_color = color_map.get(base_name)

        plt.plot(
            x, y, 
            linewidth=2.5, 
            label=label, 
            color=line_color 
        )

    plt.xlim(0.0, X_MAX)
    plt.ylim(0.0, Y_MAX)
    plt.xlabel("Rejection Rate (Discard Ratio)", fontweight="bold")
    plt.ylabel("FNMR", fontweight="bold")
    plt.title(title, pad=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()

def _variant_label(base, vtag):
    disp = _pretty_base_name(base)
    if vtag == "classification":
        return f"{disp} (Cls)"
    if vtag == "utility":
        return f"{disp} (Util)"
    if vtag == "reliability":
        return f"{disp} (Rel)"
    return disp


def _plot_vlm_simple_overlay(curves, variant, target_fmr, mode, out_dir):
    items = []
    for mk, g in curves.groupby("model_key"):
        kind, base, vtag = _parse_model_key(mk)
        if kind != "VLM" or vtag != "simple":
            continue
        items.append((_pretty_base_name(base), g))

    title = f"EvR Curves (VLMs - Simple Prompt)\n{variant} | target FMR={target_fmr:g} | {mode}"
    out = out_dir / f"EvR_VLMs_Simple__{mode}.png"
    _plot_overlay(items, title, out)


def _plot_sota_overlay(curves, variant, target_fmr, mode, out_dir):
    items = []
    for mk, g in curves.groupby("model_key"):
        kind, base, vtag = _parse_model_key(mk)
        if kind != "SOTA":
            continue
        items.append((_pretty_base_name(base), g))

    title = f"EvR Curves (SOTA FIQA Methods)\n{variant} | target FMR={target_fmr:g} | {mode}"
    out = out_dir / f"EvR_SOTAs__{mode}.png"
    _plot_overlay(items, title, out)


def _plot_per_vlm_vs_sota(curves, variant, target_fmr, mode, out_dir):
    sota_items = []
    for mk, g in curves.groupby("model_key"):
        kind, base, _ = _parse_model_key(mk)
        if kind == "SOTA":
            sota_items.append((_pretty_base_name(base), g))

    if not sota_items:
        return

    vlm_simple = []
    for mk, g in curves.groupby("model_key"):
        kind, base, vtag = _parse_model_key(mk)
        if kind == "VLM" and vtag == "simple":
            vlm_simple.append((base, mk, g))

    for base, mk, g in sorted(vlm_simple, key=lambda t: _pretty_base_name(t[0])):
        items = [(_pretty_base_name(base), g)] + sota_items
        title = f"EvR: {_pretty_base_name(base)} vs SOTA\n{variant} | target FMR={target_fmr:g} | {mode}"
        out = out_dir / "per_vlm_vs_sota" / f"EvR_{_safe_filename(base)}_vs_SOTA__{mode}.png"
        _plot_overlay(items, title, out)


def _plot_prompt_variations(curves, variant, target_fmr, mode, out_dir):
    by_base = {}

    for mk, g in curves.groupby("model_key"):
        kind, base, vtag = _parse_model_key(mk)
        if kind != "VLM":
            continue
        by_base.setdefault(base, {})[vtag] = g

    desired_order = ["simple", "classification", "utility", "reliability"]
    for base, dd in sorted(by_base.items(), key=lambda kv: _pretty_base_name(kv[0])):
        available = [v for v in desired_order if v in dd]
        if len(available) < 2:
            continue

        items = [(_variant_label(base, v), dd[v]) for v in available]
        title = f"EvR Variants: {_pretty_base_name(base)}\n{variant} | target FMR={target_fmr:g} | {mode}"
        out = out_dir / "prompt_variations" / f"EvR_{_safe_filename(base)}_Variants__{mode}.png"
        _plot_overlay(items, title, out)


def _plot_protocol_compare(curves_all, variant, target_fmr, out_dir):
    df = curves_all[
        (curves_all["variant"] == variant) &
        (curves_all["target_fmr"].astype(float) == float(target_fmr))
    ].copy()

    if df.empty:
        return

    df = df[df["model_key"].astype(str).str.startswith("VLM__")]

    for mk, g0 in df.groupby("model_key"):
        kind, base, _ = _parse_model_key(mk)
        if kind != "VLM":
            continue

        plt.figure(figsize=(8, 6))
        for mode, g in g0.groupby("mode"):
            g = g.sort_values("reject_rate")
            plt.plot(g["reject_rate"], g["fnmr"], linewidth=2.5, label=mode)

        plt.xlim(0.0, X_MAX)
        plt.ylim(0.0, Y_MAX)
        plt.xlabel("Rejection Rate (Discard Ratio)", fontweight="bold")
        plt.ylabel("FNMR", fontweight="bold")
        plt.title(f"{variant} | {_pretty_base_name(base)} | target FMR={target_fmr:g}", pad=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        out = out_dir / "protocol_compare" / f"EvR_{_safe_filename(base)}__protocol_compare.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    curves_all = _load_curves_all()
    if curves_all.empty:
        raise SystemExit(f"No curves found under {INPUT_DIR}. Run analysis first.")

    curves_all["target_fmr"] = pd.to_numeric(curves_all["target_fmr"], errors="coerce")
    curves_all["reject_rate"] = pd.to_numeric(curves_all["reject_rate"], errors="coerce")
    curves_all["fnmr"] = pd.to_numeric(curves_all["fnmr"], errors="coerce")
    curves_all = curves_all.dropna(subset=["variant", "model_key", "mode", "target_fmr", "reject_rate", "fnmr"])

    variants = sorted(curves_all["variant"].unique().tolist())
    if INCLUDE_VARIANTS:
        variants = [v for v in variants if v in set(INCLUDE_VARIANTS)]

    for variant in variants:
        dfv = curves_all[curves_all["variant"] == variant]
        tfmrs = sorted(dfv["target_fmr"].unique().tolist())
        if INCLUDE_TARGET_FMRS:
            tfmrs = [t for t in tfmrs if float(t) in set(map(float, INCLUDE_TARGET_FMRS))]

        for target_fmr in tfmrs:
            out_dir = FIG_DIR / variant / f"FMR_{target_fmr:.0e}"

            for mode in MODES_TO_PLOT:
                df = dfv[(dfv["target_fmr"] == target_fmr) & (dfv["mode"] == mode)]
                if df.empty:
                    continue

                _plot_vlm_simple_overlay(df, variant, float(target_fmr), mode, out_dir)
                _plot_sota_overlay(df, variant, float(target_fmr), mode, out_dir)
                _plot_per_vlm_vs_sota(df, variant, float(target_fmr), mode, out_dir)
                _plot_prompt_variations(df, variant, float(target_fmr), mode, out_dir)

            if PLOT_PROTOCOL_COMPARE:
                _plot_protocol_compare(curves_all, variant, float(target_fmr), out_dir)

    print(f"Done. Figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
