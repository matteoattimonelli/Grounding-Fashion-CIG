"""Catalog-alignment retrieval via SigLIP image embeddings.

Each generated image is ranked against the test catalog of bottom
items (defined by ``test_full_disj.csv`` for the chosen dataset). The
ground-truth bottom is the positive; every other unique bottom in the
test split is a distractor. The script reports MRR, Recall, and nDCG
at cutoffs 10 and 50, mirroring the catalog-alignment protocol in the
main paper.

Usage:
    python -m eval.siglip_catalog_alignment \\
        --datasets_root ${DATASETS_ROOT} \\
        --dataset fashionvc \\
        --generated_dir results/styleflow/FashionVC/test/prompt_detailed
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASET_SPECS.keys()))
    ap.add_argument("--generated_dir", required=True)
    ap.add_argument("--test_csv_name", default="test_full_disj.csv")
    ap.add_argument("--siglip_model", default="google/siglip-base-patch16-224")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor
    from torchmetrics.functional.retrieval import retrieval_normalized_dcg

    print(f"Loading SigLIP: {args.siglip_model}")
    processor = AutoProcessor.from_pretrained(args.siglip_model)
    model = AutoModel.from_pretrained(args.siglip_model).to(args.device).eval()

    label = dataset_label(args.dataset)
    csv_path = os.path.join(os.path.abspath(args.datasets_root), label, "files", args.test_csv_name)
    img_dir = os.path.join(os.path.abspath(args.datasets_root), label, "img")
    df = pd.read_csv(csv_path)

    unique_pants = list(df["positive_pant"].unique())

    @torch.no_grad()
    def _emb(image: Image.Image) -> torch.Tensor:
        inp = processor(images=image, return_tensors="pt").to(args.device)
        e = model.get_image_features(**inp)
        return e / e.norm(p=2, dim=-1, keepdim=True)

    # Encode the catalog (ground-truth bottoms) once.
    catalog_enc = []
    for pant in tqdm(unique_pants, desc="SigLIP catalog"):
        path = os.path.join(img_dir, f"{pant}.jpg")
        if not os.path.exists(path):
            catalog_enc.append(None)
            continue
        catalog_enc.append(_emb(Image.open(path).convert("RGB")).cpu())

    mrr10, mrr50, r10, r50, n10, n50, skipped = [], [], [], [], [], [], 0
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="SigLIP rank"):
            top, bot = row["tshirt"], row["positive_pant"]
            fp = os.path.join(args.generated_dir, f"{top}_{bot}_template.jpg")
            if not os.path.exists(fp):
                skipped += 1
                continue
            query_enc = _emb(Image.open(fp).convert("RGB"))
            try:
                pos_idx = unique_pants.index(bot)
            except ValueError:
                skipped += 1
                continue
            if catalog_enc[pos_idx] is None:
                skipped += 1
                continue
            target = catalog_enc[pos_idx].to(args.device)
            others = [e for j, e in enumerate(catalog_enc) if j != pos_idx and e is not None]
            if not others:
                continue
            all_enc = torch.cat([target, torch.cat(others, dim=0).to(args.device)], dim=0)
            sims = (query_enc * all_enc).sum(dim=-1)
            _, order = sims.topk(k=all_enc.shape[0], largest=True)
            order = order.tolist()
            pos = order.index(0) + 1
            mrr10.append(1.0 / pos if pos <= 10 else 0.0)
            mrr50.append(1.0 / pos if pos <= 50 else 0.0)
            r10.append(1.0 if pos <= 10 else 0.0)
            r50.append(1.0 if pos <= 50 else 0.0)
            labels = torch.zeros_like(sims); labels[0] = 1.0
            n10.append(float(retrieval_normalized_dcg(sims, labels, top_k=10).item()))
            n50.append(float(retrieval_normalized_dcg(sims, labels, top_k=50).item()))

    out = {
        "mrr_10": float(np.mean(mrr10)) if mrr10 else float("nan"),
        "mrr_50": float(np.mean(mrr50)) if mrr50 else float("nan"),
        "recall_10": float(np.mean(r10)) if r10 else float("nan"),
        "recall_50": float(np.mean(r50)) if r50 else float("nan"),
        "ndcg_10": float(np.mean(n10)) if n10 else float("nan"),
        "ndcg_50": float(np.mean(n50)) if n50 else float("nan"),
        "n_scored": len(mrr50),
        "n_skipped": skipped,
        "siglip_model": args.siglip_model,
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
