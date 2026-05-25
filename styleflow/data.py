"""Dataset utilities for StyleFlow training and evaluation.

Two datasets are exposed:

* :class:`FashionDataset` — joint training set over FashionVC,
  ExpReduced, and FashionTaobao-TB. The shipped ``train.csv`` files
  store one row per (top, bottom) pair with five instruction columns
  (``detailed / medium / low / empty / dif``). At load time we
  *explode* this wide table into a long table with one row per
  (top, bottom, prompt-level) triple, so a single epoch traverses each
  pair under every instruction granularity exactly once. ``__getitem__``
  therefore reads the per-row prompt directly — no stochastic prompt
  choice inside the dataloader.

* :class:`FashionPromptDataset` — evaluation dataset for a single
  benchmark. ``test.csv`` keeps the wide format so each pair is iterated
  once and a generation is produced for every requested prompt level
  inside ``generate.py``.

The two CSV files (``train.csv`` and ``test.csv``) shipped under
``data/<DatasetLabel>/files/`` match the schema documented in the
top-level README.
"""

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

# Map a logical prompt key to the column name in the wide CSV. ``dif``
# may or may not be present as a literal column: in ``train.csv`` it is
# (a pre-rendered template string); in ``test.csv`` only ``Type_New`` is
# stored and we render the template on the fly.
PROMPT_COLUMNS: Dict[str, Optional[str]] = {
    "detailed": "detailed",
    "medium":   "medium",
    "low":      "low",
    "empty":    "empty",
    "dif":      "dif",
}

DIF_TEMPLATE = "A photo of a {category}, on white background, high quality."
ALL_PROMPT_LEVELS: tuple = ("detailed", "medium", "low", "empty", "dif")


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


def _format_dif(row) -> str:
    """Build the DiFashion-style template prompt from the row's category."""
    cat = str(row.get("Type_New", row.get("Type_Only", "garment"))).strip() or "garment"
    return DIF_TEMPLATE.format(category=cat)


def _resolve_test_prompt(row, prompt_key: str) -> str:
    """Resolve a single prompt for the wide-format ``test.csv``.

    For ``dif`` we render the template from ``Type_New``. For the other
    keys we read the column directly and substitute a single-space
    placeholder if it is empty or missing (so ``empty`` correctly
    propagates as an unconditional prompt).
    """
    if prompt_key == "dif":
        return _format_dif(row)
    col = PROMPT_COLUMNS[prompt_key]
    val = row.get(col, "")
    if not isinstance(val, str) or not val.strip():
        return " "
    return val


# ---------------------------------------------------------------------------
# Training dataset (long format; one row per (pair, prompt-level))
# ---------------------------------------------------------------------------
class FashionDataset(Dataset):
    """Joint training set over the three CIG benchmarks, exploded by
    prompt level.

    The wide CSV columns ``detailed / medium / low / empty / dif`` are
    melted into a long table:

    ``tshirt | positive_pant | prompt_level | prompt | data_name | root_path``

    Each (top, bottom) pair therefore yields ``len(prompt_levels)`` rows
    (5 by default). ``max_examples_per_dataset`` is applied to the
    *pair* count before exploding, so the same subset of pairs is used
    across every prompt level.
    """

    def __init__(
        self,
        datasets_root: str,
        dataset_names: Sequence[str] = ("fashionvc", "expreduced", "fashiontaobaotb"),
        train_csv_name: str = "train.csv",
        prompt_levels: Sequence[str] = ALL_PROMPT_LEVELS,
        max_examples_per_dataset: Optional[int] = None,
        subset_seed: int = 0,
    ):
        super().__init__()
        self.prompt_levels = tuple(prompt_levels)
        datasets_root = os.path.abspath(datasets_root)

        long_frames: List[pd.DataFrame] = []
        for name in dataset_names:
            label, _ = DATASET_SPECS[name]
            csv_path = os.path.join(datasets_root, label, "files", train_csv_name)
            df = pd.read_csv(csv_path)
            if max_examples_per_dataset and len(df) > max_examples_per_dataset:
                df = df.sample(n=max_examples_per_dataset, random_state=subset_seed).reset_index(drop=True)
            df["data_name"] = name
            df["root_path"] = os.path.join(datasets_root, label)
            long_frames.append(self._explode(df, self.prompt_levels))

        self.data = pd.concat(long_frames, ignore_index=True)
        self.data = self.data.sample(frac=1, random_state=subset_seed).reset_index(drop=True)

    @staticmethod
    def _explode(df: pd.DataFrame, prompt_levels: Sequence[str]) -> pd.DataFrame:
        """Wide->long melt that handles missing columns and the ``dif``
        template gracefully. Falls back to a single space for empty /
        missing prompts so ``empty`` remains a valid unconditional cue."""
        rows: List[Dict[str, object]] = []
        has_dif_column = "dif" in df.columns
        for record in df.to_dict(orient="records"):
            for level in prompt_levels:
                if level == "dif":
                    text = record.get("dif", "") if has_dif_column else ""
                    if not isinstance(text, str) or not text.strip():
                        text = _format_dif(record)
                else:
                    col = PROMPT_COLUMNS[level]
                    text = record.get(col, "")
                    if not isinstance(text, str) or not text.strip():
                        text = " "
                rows.append({
                    "tshirt": record["tshirt"],
                    "positive_pant": record["positive_pant"],
                    "prompt_level": level,
                    "prompt": text,
                    "data_name": record["data_name"],
                    "root_path": record["root_path"],
                })
        return pd.DataFrame(rows)

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
        prompt = row["prompt"] if isinstance(row["prompt"], str) and row["prompt"].strip() else " "
        return {
            "pixel_values": img,
            "conditioning_pixel_values": cond,
            "prompt": prompt,
            "prompt_level": row["prompt_level"],
            "data_name": row["data_name"],
        }


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(memory_format=torch.contiguous_format).float()
    cond = torch.stack([b["conditioning_pixel_values"] for b in batch]).to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": cond,
        "captions": [b["prompt"] for b in batch],
        "prompt_levels": [b["prompt_level"] for b in batch],
        "data_name": [b["data_name"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Evaluation dataset (wide format; one row per (top, bottom) pair)
# ---------------------------------------------------------------------------
class FashionPromptDataset(Dataset):
    """Per-dataset test-time loader exposing the seed top and one or
    more instruction strings per pair.

    Returns a dict with ``top_id``, ``bottom_id``, ``top`` (seed
    tensor), and one entry per requested prompt key. ``test.csv`` is
    read in its wide format and each prompt level is resolved per
    request via :func:`_resolve_test_prompt`.
    """

    def __init__(
        self,
        datasets_root: str,
        dataset: str,
        prompt_keys: Sequence[str],
        image_size: Optional[int] = None,
        test_csv_name: str = "test.csv",
    ):
        super().__init__()
        self.dataset = dataset
        self.prompt_keys = tuple(prompt_keys)
        label, default_size = DATASET_SPECS[dataset]
        self.image_size = image_size or default_size
        root = os.path.abspath(datasets_root)
        csv_path = os.path.join(root, label, "files", test_csv_name)
        self.img_dir = os.path.join(root, label, "img")
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
            item[key] = _resolve_test_prompt(row, key)
        return item
