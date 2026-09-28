"""FID and KID via torch-fidelity.

The reference distribution is the catalog of bottom garment images for
the target dataset (typically ``${DATASETS_ROOT}/<DatasetLabel>/img``).

Usage:
    python -m eval.fid_kid \\
        --reference_dir ${DATASETS_ROOT}/FashionVC/img \\
        --generated_dir results/styleflow/FashionVC/test/prompt_detailed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True,
                    help="Catalog of ground-truth bottom images.")
    ap.add_argument("--generated_dir", required=True,
                    help="Directory of generated JPG/PNG outputs.")
    ap.add_argument("--kid_subset_size", type=int, default=100)
    ap.add_argument("--cuda", action="store_true", default=True)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    from torch_fidelity import calculate_metrics

    n_gen = (
        len(list(Path(args.generated_dir).glob("*.jpg"))) +
        len(list(Path(args.generated_dir).glob("*.png")))
    )
    kid_subset = max(10, min(args.kid_subset_size, n_gen // 2 if n_gen > 1 else 1))

    metrics = calculate_metrics(
        input1=args.reference_dir,
        input2=args.generated_dir,
        cuda=args.cuda,
        isc=False,
        fid=True,
        kid=True,
        kid_subset_size=kid_subset,
        feature_extractor_resize=True,
        verbose=False,
    )
    out = {
        "fid": float(metrics["frechet_inception_distance"]),
        "kid": float(metrics["kernel_inception_distance_mean"]),
        "kid_std": float(metrics["kernel_inception_distance_std"]),
        "n_generated": int(n_gen),
        "kid_subset_size": int(kid_subset),
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
