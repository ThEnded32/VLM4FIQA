import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

try:
    from qwen_vl_utils import process_vision_info
except Exception as e:
    process_vision_info = None
    print("[warn] qwen-vl-utils not found. Install with: pip install qwen-vl-utils")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SYSTEM_PROMPT = (
    "You are an expert image quality assessor for face images. "
    "Return STRICT JSON only with the following single key:\n"
    '{"Quality Score": <0-100 number>}' 
)

USER_TEXT = (
    "Evaluate the image quality for facial analysis. "
    "Only return the JSON object with the single key 'Quality Score'."
)

def find_image_dirs(root) :
    dirs = []
    for p in sorted([d for d in root.iterdir() if d.is_dir()]):
        has_img = any(p.rglob("*.jpg")) or any(p.rglob("*.jpeg")) or any(p.rglob("*.png"))
        if has_img:
            dirs.append(p)
    return dirs

def list_images(folder) :
    files = []
    for ext in IMG_EXTS:
        files.extend(sorted(folder.rglob(f"*{ext}")))
    return files

def load_model_and_processor(model_name, device, use_flash_attn):
    kwargs = {}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
        **kwargs,
    )
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.num_beams = 1
    model.generation_config.use_cache = True

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    if process_vision_info is None:
        raise RuntimeError("qwen-vl-utils is required. Please `pip install qwen-vl-utils`.")
    return model, processor

def build_messages(img):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": USER_TEXT},
        ]}
    ]

def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def run_on_folder(folder, model, processor, device, batch_size, max_new_tokens):
    images = list_images(folder)
    if not images:
        return pd.DataFrame(columns=["image_path", "quality_score", "output_text"])

    rows = []
    for i in tqdm(range(0, len(images), batch_size)):
        batch_paths = images[i:i+batch_size]
        batch_msgs = []
        valid_paths = []
        for p in batch_paths:
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"[warn] skipping {p}: {e}")
                continue
            batch_msgs.append(build_messages(im))
            valid_paths.append(p)

        if not valid_paths:
            continue

        try:
            texts = processor.apply_chat_template(batch_msgs, add_generation_prompt=True)
        except TypeError:
            texts = [processor.apply_chat_template([m], add_generation_prompt=True) for m in batch_msgs]

        image_inputs, video_inputs = process_vision_info(batch_msgs)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, return_tensors="pt").to(device)

        eos_id = getattr(processor.tokenizer, "eos_token_id", None)
        pad_id = getattr(processor.tokenizer, "pad_token_id", eos_id)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, 
                do_sample=False,                
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )

        try:
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception:
            generated_texts = [processor.decode(ids, skip_special_tokens=True) for ids in generated_ids]

        for pth, out_text in zip(valid_paths, generated_texts):
            parsed = extract_json(out_text)
            q = None
            if isinstance(parsed, dict):
                val = parsed.get("Quality Score", None)
                try:
                    q = float(val) if val is not None else None
                except Exception:
                    q = None
            rows.append({"image_path": str(pth), "quality_score": q, "output_text": out_text})

        del inputs, generated_ids
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", type=str, default="datasets")
    ap.add_argument("--output_folder", type=str, default="csv_outputs")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--flash_attn", action="store_true", help="Use FlashAttention v2 if available", default=True)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs",default=True)
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    if not datasets_root.exists():
        raise FileNotFoundError(f"{datasets_root} does not exist")

    args.output_folder = Path(args.output_folder)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    print(f"[info] Using device: {args.device}")
    model, processor = load_model_and_processor(args.model, args.device, args.flash_attn)

    folders = find_image_dirs(datasets_root)
    if not folders:
        print("[warn] No image directories found under", datasets_root)
        return

    for folder in folders:
        out_csv = args.output_folder / f"{folder.name}.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[skip] {out_csv.name} exists. Use --overwrite to regenerate.")
            continue
        print(f"[run] {folder.name} -> {out_csv.name}")
        df = run_on_folder(folder, model, processor, args.device, args.batch_size, args.max_new_tokens)
        df.to_csv(out_csv, index=False)

    print("[done] All eligible folders processed.]")

if __name__ == "__main__":
    main()
