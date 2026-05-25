"""LPIPS paired against the ground-truth bottom for each generation.

Generated filenames follow ``<top_id>_<bottom_id>_template.jpg``; we
extract ``<bottom_id>`` and pair it with ``<reference_dir>/<bottom_id>.jpg``
for a per-sample LPIPS computation, then report the mean.

Usage:
    python -m eval.lpips_paired \\
        --reference_dir ${DATASETS_ROOT}/FashionVC/img \\
        --generated_dir results/styleflow/FashionVC/test/prompt_detailed
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True,
                    help="Directory containing <bottom_id>.jpg ground-truth images.")
    ap.add_argument("--generated_dir", required=True)
    ap.add_argument("--net", choices=("alex", "vgg", "squeeze"), default="alex")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    import lpips
    from torchvision.io import read_image
    from torchvision.transforms.functional import resize

    model = lpips.LPIPS(net=args.net, verbose=False).to(args.device)

    def _load(path):
        img = read_image(path).float() / 127.5 - 1.0
        img = resize(img, size=[args.size, args.size])
        return img.unsqueeze(0).to(args.device)

    gen_files = sorted(
        p for p in os.listdir(args.generated_dir)
        if p.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    if not gen_files:
        raise FileNotFoundError(f"No images in {args.generated_dir}")

    total, count, missing = 0.0, 0, 0
    for fn in tqdm(gen_files, desc="LPIPS"):
        stem = fn.rsplit(".", 1)[0]
        parts = stem.split("_")
        if len(parts) < 2:
            missing += 1
            continue
        bottom_id = parts[1]
        ref_path = os.path.join(args.reference_dir, f"{bottom_id}.jpg")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(args.reference_dir, f"{bottom_id}.png")
            if not os.path.exists(ref_path):
                missing += 1
                continue
        with torch.no_grad():
            d = model(_load(os.path.join(args.generated_dir, fn)), _load(ref_path))
        total += d.item()
        count += 1

    out = {
        "lpips_mean": total / max(1, count),
        "n_pairs": count,
        "n_missing_references": missing,
        "net": args.net,
        "reference_dir": str(args.reference_dir),
        "generated_dir": str(args.generated_dir),
    }
    print(json.dumps(out, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
