# Baselines

This folder ships the reference implementations of the baselines
compared against \styleflow in the paper, together with our text-
extension variants used in the ablations. The folders are:

| Folder | Type | Conditioning | Notes |
|---|---|---|---|
| `DiFashion/` | Latent diffusion | seed top + free-form instruction | LDM baseline; closest to \styleflow in modelling family. |
| `GeCo/` | GAN + retrieval head | seed top → bottom | Joint generation/compatibility loss. |
| `MGCM_text/` | GAN (compatibility-guided) | seed top + free-form instruction | Text-augmented variant of MGCM. |
| `Pix2PixCM/` | Image-to-image GAN | seed top → bottom | Text-free baseline; CLIP-Score reported as `--`. |
| `custom_gan_text/` | GAN | seed top + free-form instruction | CLIP-text-conditioned GAN. |

All scripts read the multi-granularity instructions from the bundled
`../data/` folder using the standardized `train.csv` and `test.csv`
filenames (same as `styleflow/data.py`), and write generated images
under `baseline_outputs/<Baseline>/<DatasetLabel>/test/prompt_<level>/`
so they can be evaluated with the same protocol as \styleflow
(`../eval/*` modules).

## Common setup

Every baseline expects:

* the bundled `../data/` folder (or another `--datasets_root`-compatible
  path) containing `<DatasetLabel>/files/{train.csv, test.csv}` and a
  populated `<DatasetLabel>/img/` directory of `.jpg` images (download
  the upstream images per the top-level README);
* a Python environment with `torch`, `torchvision`, `pandas`, `Pillow`,
  `transformers`, and `open_clip_torch` (already installed by the
  repository's `requirements.txt`).

Working directory matters: the scripts resolve paths relative to
`os.getcwd()`, so run them from the repository root:

```bash
cd /path/to/Grounding-Fashion-CIG
```

---

## DiFashion

A latent-diffusion baseline. We train a LoRA adapter and a free-form-
instruction conditioning head, then generate per prompt level.

### Train

```bash
python -m baselines.DiFashion.train \
  --dataset fashionvc \
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --img_size 512 \
  --output_dir ./baselines/DiFashion/checkpoint
```

**Tunable knobs:** `--learning_rate`, `--epochs`, `--batch_size`,
`--gradient_accumulation_steps`, `--img_size`, `--guidance_scale`,
`--max_train_steps`. The provided values match the configuration used
in the paper.

### Generate

```bash
python -m baselines.DiFashion.generate \
  --dataset fashionvc \
  --mode test \
  --weights_dir ./baselines/DiFashion/checkpoint \
  --save_dir ./baselines/baseline_outputs/DiFashion \
  --batch_size 1 --img_size 512
```

### Catalog-alignment retrieval (optional)

```bash
python -m baselines.DiFashion.catalog_alignment --dataset fashionvc
```

---

## GeCo

A GAN-based CIG baseline trained jointly with a retrieval/compatibility
loss. Three CLI scripts:

### Train (sweeps over $\alpha, \beta, \gamma, \tau$)

```bash
python -m baselines.GeCo.train_geco \
  --dataset fashionvc \
  --alpha_values 0.5 \
  --beta_values 1.0 \
  --gamma_values 0.01 \
  --tau_values 0.1 \
  --num_epochs 50 \
  --train_batch_size 64 \
  --emb_dim 128 \
  --img_size 128 \
  --learning_rate 1e-4
```

`--alpha_values / --beta_values / --gamma_values / --tau_values` accept
a list (`nargs='+'`) so a full grid sweep can be launched with one
invocation. Best (alpha, beta, gamma, tau) for each dataset are stored
implicitly in the checkpoint filename; pick the best per dataset on
validation FID and feed the path to `test.py` / `eval.py`.

### Generate / Test

```bash
python -m baselines.GeCo.test \
  --dataset fashionvc \
  --emb_dim 128 \
  --img_size 128
```

Edit `weight_path` and `generator_path` near the top of `test.py` to
point at the checkpoints produced by the training sweep (the script
contains commented examples for each dataset).

### Catalog-alignment retrieval

```bash
python -m baselines.GeCo.eval --dataset fashionvc --emb_dim 128
```

---

## MGCM_text

GAN trained with a compatibility-guided loss and free-form instruction
conditioning (frozen CLIP text encoder + small projection trained jointly).

### Train

```bash
python -m baselines.MGCM_text.train_mgcm \
  --dataset fashionvc \
  --alpha_values 1 --beta_values 0.01 --mi_values 0.1 --ni_values 0.01 \
  --epochs 60 --batch_size 420 --learning_rate 2e-4 \
  --img_size 64 --out_csv ./baselines/MGCM_text/out.csv
```

`alpha / beta / mi / ni` weight the four loss terms (BPR compatibility,
adversarial, mutual-information, identity); start from the listed
defaults and grid-search around them per dataset. Per-run results are
appended to `--out_csv`.

### Test

```bash
python -m baselines.MGCM_text.test --dataset fashionvc --img_size 64
```

---

## Pix2PixCM

A text-free image-to-image GAN (Pix2Pix + compatibility head). Trained
with the same loss-weight CLI as `MGCM_text`.

```bash
python -m baselines.Pix2PixCM.train_pix2pixcm \
  --dataset fashionvc \
  --alpha_values 1 --beta_values 0.01 --mi_values 0.1 --ni_values 0.01 \
  --epochs 60 --batch_size 420 --learning_rate 2e-4 \
  --img_size 64 --out_csv ./baselines/Pix2PixCM/out.csv
```

Outputs are at 64×64 by design; upsample at evaluation time only.

---

## custom_gan_text

CLIP-text-conditioned generator/discriminator pair used in the
text-augmented ablation.

```bash
python -m baselines.custom_gan_text.train_cigm \
  --dataset fashionvc \
  --num_epochs 200 \
  --train_batch_size 64 \
  --learning_rate 2e-4 \
  --img_size 128 \
  --beta1 0.5 \
  --L1Lambda 100 \
  --weights_dir ./baselines/custom_gan_text/weights
```

`--L1Lambda` weights the pixel-reconstruction loss against the GAN
adversarial signal; the default `100` mirrors Pix2Pix's recommended
value.

---

## Evaluating baselines with the StyleFlow metric scripts

Once a baseline has produced its output folder, run the same four
metrics used for \styleflow:

```bash
GEN_DIR=baselines/baseline_outputs/DiFashion/FashionVC/test/prompt_detailed

python -m eval.fid_kid \
  --reference_dir ./data/FashionVC/img \
  --generated_dir ${GEN_DIR}

python -m eval.lpips_paired \
  --reference_dir ./data/FashionVC/img \
  --generated_dir ${GEN_DIR}

python -m eval.clip_score \
  --datasets_root ./data \
  --dataset fashionvc \
  --generated_dir ${GEN_DIR} \
  --prompt_column detailed

python -m eval.siglip_catalog_alignment \
  --datasets_root ./data \
  --dataset fashionvc \
  --generated_dir ${GEN_DIR}
```

`../scripts/eval_baselines_example.sh` automates this sweep across
baselines × datasets × prompt levels, skipping CLIP-Score for the
text-free `Pix2PixCM`.
