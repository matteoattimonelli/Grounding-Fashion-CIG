#!/usr/bin/env python
"""StyleFlow inference / generation entry point.

Loads ``StyleFlowPipeline`` from a FLUX backbone + optional LoRA
checkpoint and iterates over a test split, writing one generated image
per (top, bottom, prompt-level) triple to
``<output_root>/<DatasetLabel>/test/prompt_<level>/<top>_<bottom>_template.jpg``.

The output filenames are dedup-safe: if the same (top, bottom, level)
already exists on disk it is skipped, so the script can be re-run
incrementally.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from styleflow.data import (
    FashionPromptDataset,
    dataset_image_size,
    dataset_label,
    prompt_output_subdir,
)
from styleflow.pipeline import build_styleflow_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--checkpoint", "--checkpoint_dir", dest="checkpoint", default=None,
                   help="Path to StyleFlow LoRA weights. Either a directory "
                        "containing pytorch_lora_weights.safetensors (e.g. an output "
                        "of train.py) or a direct .safetensors / .bin file "
                        "(e.g. the released styleflow_cig_OOD_GA8_DIF_lora_weights.safetensors). "
                        "Leave unset to evaluate the zero-shot FLUX backbone.")
    p.add_argument("--dataset", required=True, choices=["fashionvc", "expreduced", "fashiontaobaotb", "fashiontaobaoTB"])
    p.add_argument("--datasets_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--prompt_keys", default="detailed,medium,low,dif")
    p.add_argument("--test_csv_name", default="test_full_disj.csv")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate even if an output file already exists.")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _image_output_path(out_dir: str, top_id: str, bottom_id: str) -> str:
    return os.path.join(out_dir, f"{top_id}_{bottom_id}_template.jpg")


def _make_dirs(root: str, prompt_keys: List[str]) -> None:
    os.makedirs(root, exist_ok=True)
    for p in prompt_keys:
        os.makedirs(os.path.join(root, prompt_output_subdir(p)), exist_ok=True)


def _subset_field(value, indices):
    if torch.is_tensor(value):
        return value[indices]
    return [value[i] for i in indices]


def _collect_pending(batch, output_dir: str, overwrite: bool):
    pending, paths = [], []
    for i, (top, bot) in enumerate(zip(batch["top_id"], batch["bottom_id"])):
        path = _image_output_path(output_dir, top, bot)
        if overwrite or not os.path.exists(path):
            pending.append(i)
            paths.append(path)
    return pending, paths


def main():
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    label = dataset_label(args.dataset)
    img_size = args.img_size or dataset_image_size(args.dataset)
    prompt_keys = [k.strip() for k in args.prompt_keys.split(",") if k.strip()]

    out_root = Path(args.output_root, label, "test")
    _make_dirs(str(out_root), prompt_keys)

    print(f"Loading StyleFlow ({args.pretrained_model_name_or_path}; "
          f"checkpoint={args.checkpoint or 'zero-shot FLUX'})")
    pipe = build_styleflow_pipeline(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint=args.checkpoint,
        torch_dtype=dtype,
        device=args.device,
    )
    pipe.set_progress_bar_config(disable=True)

    dataset = FashionPromptDataset(
        datasets_root=args.datasets_root,
        dataset=args.dataset,
        prompt_keys=prompt_keys,
        image_size=img_size,
        test_csv_name=args.test_csv_name,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset: {len(dataset)} pairs at {img_size}x{img_size}, bs={args.batch_size}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    generated = skipped = 0

    for batch in tqdm(dataloader, desc=f"gen {args.dataset}"):
        seed_tensor = batch["top"].to(args.device, dtype=torch.float32)
        for prompt_key in prompt_keys:
            sub = os.path.join(str(out_root), prompt_output_subdir(prompt_key))
            pending_idx, out_paths = _collect_pending(batch, sub, args.overwrite)
            skipped += len(batch["top_id"]) - len(pending_idx)
            if not pending_idx:
                continue
            seed_subset = _subset_field(seed_tensor, pending_idx)
            prompts = _subset_field(batch[prompt_key], pending_idx)
            with torch.no_grad():
                out = pipe(
                    prompt=prompts,
                    control_image=seed_subset,
                    height=img_size,
                    width=img_size,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                    generator=torch.Generator("cpu").manual_seed(args.seed),
                ).images
            for img, p in zip(out, out_paths):
                img.save(p, format="JPEG", quality=95)
            generated += len(out_paths)

    dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0
    print(f"Done: generated={generated} skipped={skipped} "
          f"elapsed={dt:.1f}s ({generated / max(dt, 1e-8):.2f} img/s) "
          f"peak GPU={peak:.1f} GB")


if __name__ == "__main__":
    main()
