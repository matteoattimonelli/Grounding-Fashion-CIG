"""CLIP-Score via open_clip ViT-B/32 (laion2b_s34b_b79k).

Computes the mean cosine similarity between each generated image's
CLIP image embedding and the CLIP text embedding of the prompt that
produced it.

Usage:
    python -m eval.clip_score \\
        --datasets_root ${DATASETS_ROOT} \\
        --dataset fashionvc \\
        --generated_dir results/styleflow/FashionVC/test/prompt_detailed \\
        --prompt_column detailed
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from styleflow.data import DATASET_SPECS, dataset_label


DIF_TEMPLATE = "A photo of a {category}, on white background, high quality."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASET_SPECS.keys()))
    ap.add_argument("--generated_dir", required=True)
    ap.add_argument("--prompt_column", default="detailed",
                    help='Column from test_full_disj.csv to use as the text prompt. '
                         'Use "dif" for the DiFashion-style template.')
    ap.add_argument("--test_csv_name", default="test_full_disj.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    import open_clip

    print("Loading open_clip ViT-B/32 (laion2b_s34b_b79k)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.eval().to(args.device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    label = dataset_label(args.dataset)
    csv_path = os.path.join(os.path.abspath(args.datasets_root), label, "files", args.test_csv_name)
    df = pd.read_csv(csv_path)

    sims, skipped = [], 0
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CLIP-Score"):
            top, bot = row["tshirt"], row["positive_pant"]
            fp = os.path.join(args.generated_dir, f"{top}_{bot}_template.jpg")
            if not os.path.exists(fp):
                skipped += 1
                continue
            try:
                image = preprocess(Image.open(fp).convert("RGB")).unsqueeze(0).to(args.device)
            except Exception:
                skipped += 1
                continue

            if args.prompt_column == "dif":
                cat = str(row.get("Type_New", row.get("Type_Only", "garment"))).strip() or "garment"
                prompt = DIF_TEMPLATE.format(category=cat)
            else:
                val = row.get(args.prompt_column, "")
                prompt = str(val) if isinstance(val, str) and val.strip() else " "

            tok = tokenizer([prompt]).to(args.device)
            img_f = model.encode_image(image)
            txt_f = model.encode_text(tok)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            sims.append(float((img_f * txt_f).sum().item()))

    out = {
        "clip_score_mean": float(np.mean(sims)) if sims else float("nan"),
        "n_scored": len(sims),
        "n_skipped": skipped,
        "prompt_column": args.prompt_column,
        "dataset": args.dataset,
        "generated_dir": str(args.generated_dir),
    }
    print(json.dumps(out, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
