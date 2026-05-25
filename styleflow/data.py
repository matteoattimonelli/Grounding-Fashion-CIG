"""Dataset utilities for StyleFlow training and evaluation.

Two datasets are exposed:

* :class:`FashionDataset` — joint training set over FashionVC,
  ExpReduced, and FashionTaobao-TB. Each ``__getitem__`` returns a
  (seed top, ground-truth bottom, instruction, dataset name) tuple at
  the dataset's native resolution (128 / 224 / 512).

* :class:`FashionPromptDataset` — evaluation dataset for a single
  benchmark, exposing the seed top tensor at the native resolution and
  every instruction granularity selected via ``prompt_keys``.

The two CSV files expected per dataset are described in the README.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DATASET_SPECS: Dict[str, tuple] = {
    "fashionvc": ("FashionVC", 128),
    "expreduced": ("ExpReduced", 224),
    "fashiontaobaotb": ("FashionTaobao-TB", 512),
    "fashiontaobaoTB": ("FashionTaobao-TB", 512),
}

PROMPT_COLUMNS: Dict[str, Optional[str]] = {
    "detailed": "detailed",
    "medium":   "medium",
    "low":      "low",
    "empty":    "empty",
    "dif":      None,  # built from "Type_New" with a template
}

DIF_TEMPLATE = "A photo of a {category}, on white background, high quality."


def dataset_label(dataset: str) -> str:
    return DATASET_SPECS[dataset][0]


def dataset_image_size(dataset: str) -> int:
    return DATASET_SPECS[dataset][1]


def prompt_output_subdir(prompt_key: str) -> str:
    return f"prompt_{prompt_key}"


def _build_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def _resolve_prompt(row, prompt_key: str) -> str:
    col = PROMPT_COLUMNS[prompt_key]
    if col is None:  # dif
        cat = str(row.get("Type_New", row.get("Type_Only", "garment"))).strip() or "garment"
        return DIF_TEMPLATE.format(category=cat)
    val = row.get(col, "")
    if not isinstance(val, str) or not val.strip():
        return " "
    return val


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------
class FashionDataset(Dataset):
    """Joint training set over the three CIG benchmarks.

    Each sample is returned at the dataset's native resolution
    (128/224/512) and exposes the seed top image, the ground-truth
    bottom image, the selected instruction, and the source dataset
    name. The instruction column is sampled uniformly from
    ``prompt_columns`` per draw, so a single epoch hits all prompt
    granularities for each (seed, target) pair.
    """

    def __init__(
        self,
        datasets_root: str,
        dataset_names: Sequence[str] = ("fashionvc", "expreduced", "fashiontaobaotb"),
        train_csv_name: str = "train_full_columns_dif_G.csv",
        prompt_columns: Sequence[str] = ("bottom_description",),
        max_examples_per_dataset: Optional[int] = None,
        subset_seed: int = 0,
    ):
        super().__init__()
        self.prompt_columns = tuple(prompt_columns) or ("bottom_description",)
        datasets_root = os.path.abspath(datasets_root)

        frames = []
        for name in dataset_names:
            label, _ = DATASET_SPECS[name]
            csv_path = os.path.join(datasets_root, label, "files", train_csv_name)
            df = pd.read_csv(csv_path)
            if max_examples_per_dataset and len(df) > max_examples_per_dataset:
                df = df.sample(n=max_examples_per_dataset, random_state=subset_seed).reset_index(drop=True)
            df["bottom_description"] = df["bottom_description"].fillna(" ")
            df["data_name"] = name
            df["root_path"] = os.path.join(datasets_root, label)
            for col in self.prompt_columns:
                if col not in df.columns and col != "dif":
                    df[col] = df["bottom_description"]
                if col in df.columns:
                    df[col] = df[col].fillna(" ")
            frames.append(df)
        self.data = pd.concat(frames).sample(frac=1, random_state=subset_seed).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.data.iloc[idx]
        _, size = DATASET_SPECS[row["data_name"]]
        tx = _build_transform(size)
        img_path = os.path.join(row["root_path"], "img", f"{row['positive_pant']}.jpg")
        cond_path = os.path.join(row["root_path"], "img", f"{row['tshirt']}.jpg")
        img = tx(Image.open(img_path).convert("RGB"))
        cond = tx(Image.open(cond_path).convert("RGB"))

        # Sample a prompt column for this example.
        import random
        prompt_col = random.choice(self.prompt_columns)
        prompt = str(row.get(prompt_col, " ")) if prompt_col != "dif" else _resolve_prompt(row, "dif")
        if not prompt.strip():
            prompt = " "

        return {
            "pixel_values": img,
            "conditioning_pixel_values": cond,
            "prompt": prompt,
            "data_name": row["data_name"],
        }


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(memory_format=torch.contiguous_format).float()
    cond = torch.stack([b["conditioning_pixel_values"] for b in batch]).to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": cond,
        "captions": [b["prompt"] for b in batch],
        "data_name": [b["data_name"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------
class FashionPromptDataset(Dataset):
    """Per-dataset test-time loader exposing the seed top and one or
    more instruction strings per item.

    Returns a dict with ``top_id``, ``bottom_id``, ``top`` (seed
    tensor), and one entry per requested prompt key. Filenames in the
    generation output follow ``<top_id>_<bottom_id>_template.jpg``.
    """

    def __init__(
        self,
        datasets_root: str,
        dataset: str,
        prompt_keys: Sequence[str],
        image_size: Optional[int] = None,
        test_csv_name: str = "test_full_disj.csv",
    ):
        super().__init__()
        self.dataset = dataset
        self.prompt_keys = tuple(prompt_keys)
        label, default_size = DATASET_SPECS[dataset]
        self.image_size = image_size or default_size
        csv_path = os.path.join(os.path.abspath(datasets_root), label, "files", test_csv_name)
        self.img_dir = os.path.join(os.path.abspath(datasets_root), label, "img")
        self.rows: List[dict] = pd.read_csv(csv_path).to_dict("records")
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        top_id = str(row.get("tshirt") or row.get("top_id"))
        bottom_id = str(row.get("positive_pant") or row.get("bottom_id"))
        top_image = Image.open(os.path.join(self.img_dir, f"{top_id}.jpg")).convert("RGB")
        item: Dict[str, object] = {
            "top_id": top_id,
            "bottom_id": bottom_id,
            "top": self.transform(top_image),
        }
        for key in self.prompt_keys:
            item[key] = _resolve_prompt(row, key)
        return item
